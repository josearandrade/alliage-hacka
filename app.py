"""Gradio interface for local or remote panoramic positioning inference."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import gradio as gr
import httpx
import joblib
import torch
from PIL import Image

from fusion_model import FUSION_DIR, load_models, predict_fusion_image
from pipeline import (
    ARTIFACTS,
    CLASS_NAMES,
    CLASSES,
    IMAGES,
    load_backbone,
    load_json,
    predict_image,
)

TEST_IMAGE = Path(__file__).resolve().parent / "assets" / "imagem-teste.jpg"

RESPONSIVE_CSS = """
.gradio-container { width: 100%; }
.report-table { min-width: 0 !important; overflow-x: auto; }
.report-table table { font-size: 0.875rem; }
@media (max-width: 767px) {
    .gradio-container { padding: 12px !important; }
    .gradio-container h1 { font-size: 1.5rem !important; line-height: 1.25 !important; }
    .gradio-container h3 { font-size: 1.125rem !important; }
    .mobile-stack { flex-direction: column !important; }
    .mobile-stack > * { min-width: 0 !important; width: 100% !important; flex: auto !important; }
    .gradio-container .prose { overflow-wrap: anywhere; }
    .gradio-container [role="tablist"] { flex-wrap: wrap; gap: 4px; }
    .gradio-container .tab-container button { min-height: 44px; padding: 8px; font-size: 0.8125rem; }
    #test-image-button, #classify-button { min-height: 48px; width: 100%; }
    #input-radiograph, #heatmap { height: 240px !important; min-width: 0 !important; }
    #example-gallery .grid-container { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
}
@media (max-width: 380px) {
    #example-gallery .grid-container { grid-template-columns: minmax(0, 1fr) !important; }
}
"""


def load_test_image():
    with Image.open(TEST_IMAGE) as source:
        image = source.convert("RGB")
    return image, "", None, None


def summary_table(report: dict, title: str) -> str:
    lines = [
        f"### {title}",
        f"**{report['n']} imagens · Acurácia {report['accuracy']:.1%} · F1 macro {report['macro_f1']:.1%}**",
        "| Classe | Precision | Recall | F1 | Imagens |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in CLASSES:
        row = report["classification_report"][label]
        lines.append(f"| {CLASS_NAMES[label]} | {row['precision']:.1%} | {row['recall']:.1%} | {row['f1-score']:.1%} | {int(row['support'])} |")
    return "\n".join(lines)


def choose_examples(report: dict):
    examples = []
    for label in CLASSES:
        matches = [row for row in report["predictions"] if row["true"] == label]
        correct = [row for row in matches if row["predicted"] == label]
        picked = max(correct or matches, key=lambda row: row["confidence"])
        caption = f"Real: {CLASS_NAMES[label]} · Prevista: {CLASS_NAMES[picked['predicted']]} · Escore: {picked['confidence']:.0%}"
        examples.append((str(IMAGES / f"{picked['hash']}.jpg"), caption))
    return examples


def top_confusions(report: dict) -> str:
    rows = report["confusions"][:3]
    if not rows:
        return "Nenhuma confusão neste conjunto."
    return "Confusões mais frequentes: " + "; ".join(f"{CLASS_NAMES[row['real']]} → {CLASS_NAMES[row['predicted']]} ({row['count']})" for row in rows)


def remote_predict(image: Image.Image, url: str, api_key: str) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = httpx.post(f"{url}/predict", files={"file": ("radiografia.png", buffer.getvalue(), "image/png")}, headers=headers, timeout=180)
        response.raise_for_status()
        payload = response.json()
        return {
            "label": payload["label"],
            "confidence": payload["confidence"],
            "scores": payload["scores"],
            "elapsed_ms": payload["elapsed_ms"],
            "overlay": Image.open(io.BytesIO(base64.b64decode(payload["overlay_png"]))).convert("RGB"),
        }
    except httpx.HTTPError as exc:
        raise gr.Error(f"Falha no processamento remoto: {exc}") from exc
    except (KeyError, ValueError, OSError) as exc:
        raise gr.Error(f"Resposta inválida da API remota: {exc}") from exc


def build_app() -> gr.Blocks:
    inference_url = os.environ.get("INFERENCE_URL", "").rstrip("/")
    api_key = os.environ.get("ALLIAGE_API_KEY", "")
    selected_model = os.environ.get("ALLIAGE_MODEL", "fusion" if (FUSION_DIR / "training.json").exists() else "baseline")
    if selected_model not in {"fusion", "baseline"}:
        raise ValueError("ALLIAGE_MODEL deve ser 'fusion' ou 'baseline'.")
    report_dir = FUSION_DIR if selected_model == "fusion" else ARTIFACTS
    raw = load_json(report_dir / "test_raw.json")
    clean = load_json(report_dir / "test_clean.json")
    benchmark = load_json(report_dir / "inference_benchmark.json")
    training = load_json(report_dir / "training.json")
    feature_mode = training["feature_mode"]
    backbone_key = training["backbone_key"]
    tta = training.get("tta", False)
    classifier = joblib.load(report_dir / "classifier.joblib") if not inference_url else None
    device = "cloud" if inference_url else ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = (load_models(device, training["backbones"]) if selected_model == "fusion" else load_backbone(device, backbone_key)) if not inference_url else None

    def local_predict(image: Image.Image) -> dict:
        if selected_model == "fusion":
            return predict_fusion_image(image, classifier, backbone, device)
        return predict_image(image, classifier, backbone, device, feature_mode, backbone_key, tta)

    def classify(image: Image.Image):
        if image is None:
            raise gr.Error("Selecione uma radiografia JPG ou PNG.")
        result = remote_predict(image, inference_url, api_key) if inference_url else local_predict(image)
        scores = {CLASS_NAMES[label]: result["scores"][label] for label in CLASSES}
        headline = f"### {CLASS_NAMES[result['label']]}\nEscore do classificador: **{result['confidence']:.1%}** · Inferência: **{result['elapsed_ms']:.0f} ms** ({device.upper()})"
        return headline, scores, result["overlay"]

    examples = choose_examples(clean)
    errors = [row for row in clean["predictions"] if row["true"] != row["predicted"]]
    if not errors:
        errors = [row for row in raw["predictions"] if row["true"] != row["predicted"]]
    error = max(errors, key=lambda row: row["confidence"]) if errors else None

    with gr.Blocks(title="Controle de posicionamento panorâmico", css=RESPONSIVE_CSS) as app:
        gr.Markdown("# Controle de posicionamento em radiografias panorâmicas\nClassificação automática em seis condições de aquisição. O escore indica a preferência do modelo entre as classes; não mede a chance real de acerto. Esta ferramenta não faz diagnóstico.")
        if inference_url:
            gr.Markdown(f"Processamento remoto habilitado: `{inference_url}`. A imagem será enviada à API configurada.")
        with gr.Tab("Classificar"):
            with gr.Row(elem_classes="mobile-stack"):
                input_image = gr.Image(label="Radiografia panorâmica", type="pil", sources=["upload"], elem_id="input-radiograph")
                with gr.Column(min_width=0):
                    test_image_button = gr.Button("imagem teste", elem_id="test-image-button")
                    gr.Markdown("Carregue uma imagem teste e clique em Classificar.")
                    classify_button = gr.Button("Classificar", variant="primary", elem_id="classify-button")
                    headline = gr.Markdown()
                    scores = gr.Label(label="Escores por classe", num_top_classes=6)
            heatmap = gr.Image(label="Regiões que contribuíram para a classe prevista", type="pil", elem_id="heatmap")
            gr.Markdown("O mapa mostra a contribuição aproximada dos blocos visuais para diferenciar as duas classes mais prováveis. As representações têm contexto global; o mapa não deve ser interpretado como localização clínica precisa.")
            classify_button.click(classify, inputs=input_image, outputs=[headline, scores, heatmap])
            test_image_button.click(load_test_image, inputs=[], outputs=[input_image, headline, scores, heatmap])
        with gr.Tab("Resultados do teste"):
            gr.Markdown("O teste fornecido tem 81 arquivos. Onze pares contêm a mesma imagem com rótulos diferentes.")
            with gr.Row(elem_classes="mobile-stack"):
                gr.Markdown(summary_table(raw, "Teste completo"), elem_classes="report-table")
                gr.Markdown(summary_table(clean, "Teste sem conflitos de rótulo"), elem_classes="report-table")
            gr.Markdown(top_confusions(raw))
            gr.Markdown(f"Tempo de inferência após aquecimento: mediana **{benchmark['median_ms']:.0f} ms**, p95 **{benchmark['p95_ms']:.0f} ms** em {benchmark['device'].upper()}.")
            with gr.Row(elem_classes="mobile-stack"):
                gr.Image(value=str(report_dir / "test_raw_confusion.png"), label="Matriz — 81 arquivos", interactive=False)
                gr.Image(value=str(report_dir / "test_clean_confusion.png"), label="Matriz — 59 imagens", interactive=False)
        with gr.Tab("Exemplos"):
            gr.Markdown("### Um exemplo de cada classe do teste sem conflito")
            gr.Gallery(value=examples, label="Seis classes", columns=3, object_fit="contain", height=280, elem_id="example-gallery")
            if error:
                gr.Markdown(f"### Um erro real do modelo\nClasse real: **{CLASS_NAMES[error['true']]}** · Prevista: **{CLASS_NAMES[error['predicted']]}** · Escore: **{error['confidence']:.1%}**.")
                with gr.Row(elem_classes="mobile-stack"):
                    gr.Image(value=str(IMAGES / f"{error['hash']}.jpg"), label="Imagem classificada incorretamente", interactive=False)
                    if inference_url:
                        gr.Markdown("Envie esta imagem na aba Classificar para obter o mapa pela API remota.")
                    else:
                        with Image.open(IMAGES / f"{error['hash']}.jpg") as source:
                            error_overlay = local_predict(source.copy())["overlay"]
                        gr.Image(value=error_overlay, label="Contribuições à classe prevista", interactive=False)
    return app


if __name__ == "__main__":
    if not os.environ.get("INFERENCE_URL") and not (ARTIFACTS / "classifier.joblib").exists():
        raise SystemExit("Treine o classificador primeiro: python pipeline.py train")
    auth_user = os.environ.get("GRADIO_AUTH_USER")
    auth_password = os.environ.get("GRADIO_AUTH_PASSWORD")
    if bool(auth_user) != bool(auth_password):
        raise SystemExit("Defina GRADIO_AUTH_USER e GRADIO_AUTH_PASSWORD juntos.")
    build_app().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
        auth=(auth_user, auth_password) if auth_user else None,
        allowed_paths=[str(IMAGES), str(ARTIFACTS)],
    )

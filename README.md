# Controle de posicionamento em radiografias panorâmicas

O projeto classifica uma radiografia panorâmica em seis condições de posicionamento. Ele pode funcionar de três formas:

- interface local em um container Linux;
- API em uma máquina ou serviço de nuvem, usando CPU ou GPU;
- interface local enviando as imagens para essa API remota.

O resultado é uma classificação de posicionamento e um mapa visual aproximado. A ferramenta não faz diagnóstico nem decide se uma exposição deve ser repetida.

## Executar diretamente no Windows

### Recursos locais necessários

O repositório versiona código, infraestrutura e o registro dos experimentos. Dados, pesos, classificadores, relatórios gerados, caches e certificados locais não acompanham o clone. Antes de executar a aplicação ou construir a imagem Docker, prepare:

- `data/images/` e `data/splits.json`, incluindo as imagens referenciadas nos relatórios;
- `data/hf_cache/`, com os modelos e revisões definidos em `pipeline.py`;
- `artifacts/training.json` e `artifacts/classifier.joblib` para o baseline;
- para a fusão, `artifacts/fusion/training.json` e `artifacts/fusion/classifier.joblib`;
- no diretório do modelo usado pela interface: `test_raw.json`, `test_clean.json`, `inference_benchmark.json`, `test_raw_confusion.png` e `test_clean_confusion.png`.

Os recursos devem corresponder à mesma preparação e treinamento. A interface lê os relatórios inclusive no modo remoto. Seu launcher também verifica o classificador baseline no modo local, mesmo quando a fusão está selecionada. A API não exige os cinco arquivos de avaliação para inferência.

O ambiente local validado usa PyTorch `2.11.0+cu128` e RTX 5060. `requirements.txt` e o Dockerfile fixam PyTorch `2.6.0`; essa configuração não foi validada nesta máquina. O comando abaixo instala a versão CUDA local depois dos demais requisitos.

Com Python 3.12 instalado, use a RTX 5060 sem Docker:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe pipeline.py prepare
.\.venv\Scripts\python.exe pipeline.py train --backbone rad-dino --batch-size 2
.\.venv\Scripts\python.exe pipeline.py benchmark --batch-size 2
.\.venv\Scripts\python.exe experiments/compare_frozen_features.py
.\.venv\Scripts\python.exe fusion_model.py train --batch-size 2
.\.venv\Scripts\python.exe app.py
```

A interface abre em `http://127.0.0.1:7860`. Após o treino, basta executar `app.py` para iniciar a demonstração novamente.

Na aba Classificar, o botão **imagem teste** carrega o exemplo distribuído em `assets/imagem-teste.jpg`; depois clique em Classificar. Esse arquivo de demonstração acompanha o repositório e a imagem Docker, ao contrário do dataset de treinamento.

## Demonstração online

Link temporário criado em 24/09/2026: [abrir a aplicação](https://sign-scanned-contain-programmer.trycloudflare.com).

O acesso público e uma classificação completa foram verificados pelo link. A aplicação usa a GPU do computador local; não foi implantada em um servidor permanente. O computador, a aplicação e o túnel precisam permanecer ligados. Ao reiniciar o túnel, o endereço poderá mudar.

Para abrir outro túnel com `cloudflared` instalado e a aplicação em execução:

```bash
cloudflared tunnel --url http://127.0.0.1:7860 --no-autoupdate
```

O Gradio aceita `GRADIO_SHARE=true`, mas a criação desse link falhou no ambiente local. O acesso atual usa Cloudflare Tunnel. Para autenticação na interface, defina `GRADIO_AUTH_USER` e `GRADIO_AUTH_PASSWORD` juntos.

## Executar localmente com Docker

Requisitos mínimos:

- Linux x86_64;
- Docker Engine e Docker Compose;
- acesso ao daemon do Docker;
- espaço para a imagem Docker, os dados e os pesos preparados localmente.

Python, CUDA e bibliotecas científicas não precisam estar instalados no host. Entre na pasta do projeto e execute:

```bash
./run.sh
```

O script verifica o daemon Docker, tenta detectar uma GPU NVIDIA e inicia automaticamente em GPU ou CPU. A interface fica em `http://localhost:7860`. Para trocar a porta:

```bash
ALLIAGE_PORT=9000 ./run.sh
```

O launcher solicita GPU quando `nvidia-smi` está disponível e responde. Esse modo também exige driver e runtime NVIDIA configurados para Docker; uma falha desse runtime não provoca nova tentativa automática em CPU. Para iniciar explicitamente sem solicitar GPU:

```bash
docker compose -f docker-compose.yml up --build
```

O build precisa dos recursos locais listados acima. O container define `HF_HOME=/app/data/hf_cache` e `TRANSFORMERS_OFFLINE=1`; prepare os pesos antes do build. O volume `alliage-hf-cache` é persistente e precisa conter as revisões necessárias ao modelo selecionado.

## Processamento em nuvem

O serviço HTTP pode ser executado em qualquer ambiente que aceite uma imagem Docker Linux. O ambiente de nuvem não precisa de GPU: o container usa CPU quando CUDA não está disponível.

Para testar a API localmente:

```bash
./run-cloud.sh
```

Ela ficará em `http://localhost:8080`. Verifique o serviço:

```bash
curl http://localhost:8080/healthz
```

Envie uma imagem:

```bash
curl -X POST http://localhost:8080/predict \
  -F file=@radiografia.jpg \
  -o resultado.json
```

O endpoint retorna classe, escores, tempo, dispositivo usado e o mapa visual em PNG codificado em base64. O limite de upload é 25 MB e os formatos aceitos são JPG e PNG.

Para proteger a API, defina uma chave:

```bash
ALLIAGE_API_KEY='uma-chave-longa' ./run-cloud.sh
```

O cliente deverá enviar essa chave no cabeçalho `X-API-Key`. Sem uma chave configurada, o endpoint fica sem autenticação; isso só é adequado para uma rede controlada.

Depois de publicar a API em um endereço HTTPS, a interface pode encaminhar o processamento para ela:

```bash
INFERENCE_URL=https://api.exemplo.com \
ALLIAGE_API_KEY='uma-chave-longa' \
python app.py
```

Nesse modo, a imagem é enviada ao endereço configurado. A interface ainda precisa dos relatórios e exemplos locais. O Compose atual não encaminha `INFERENCE_URL` nem `ALLIAGE_API_KEY` ao serviço de interface; configure-as explicitamente no container se usar esse modo.

## Modo offline

Para uma banca sem internet, prepare uma imagem Docker exportada em uma máquina com acesso à rede:

```bash
docker compose build
docker save alliage-hacka:latest -o alliage-hacka.tar
```

Copie `alliage-hacka.tar` e o projeto para o computador de execução e rode:

```bash
./run-offline.sh alliage-hacka.tar
```

Os pesos do modelo e os artefatos da demonstração ficam dentro da imagem. O ZIP de treinamento não é necessário para classificar imagens.

## Reconstruir o modelo

O treinamento é opcional e precisa do ZIP original na raiz com o nome esperado pelo pipeline:

```bash
python pipeline.py prepare
python pipeline.py train --backbone rad-dino --batch-size 2
```

Execute no ambiente Python local com acesso à rede para obter os pesos. Esse fluxo pode exigir mais memória, espaço e tempo que a demonstração. A ausência do ZIP não impede o uso do classificador já empacotado.

## Experimentos de melhoria

O modelo principal concatena as representações congeladas de [RAD-DINO](https://huggingface.co/microsoft/rad-dino) e [DINOv2-small](https://huggingface.co/facebook/dinov2-small), normalizadas separadamente. Uma regressão logística classifica as seis posições. Os pesos dos extratores são fixados por revisão. O modelo anterior, com apenas RAD-DINO, permanece em `artifacts/` e pode ser executado com `ALLIAGE_MODEL=baseline`.

Para comparar DINOv2-small e Rad-DINO congelados:

```bash
python pipeline.py benchmark --batch-size 4
```

O relatório fica em `artifacts/experiments.json`. A comparação das representações e o treinamento final são reproduzidos com:

```bash
python experiments/compare_frozen_features.py
python fusion_model.py train --batch-size 2
```

Os experimentos e as mudanças estão em [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md); os relatórios do modelo principal ficam em `artifacts/fusion/`. A seleção usa apenas validação cruzada no treino. O teste fornecido já foi consultado em avaliações anteriores; não é um novo conjunto independente.

### Comparações adicionais de 24/09/2026

Nenhuma alternativa satisfez o critério de aumentar a acurácia média sem reduzir F1 macro nos mesmos cinco folds. O modelo ativo permanece **RAD-DINO + DINOv2-small, regressão logística balanceada (`C=0,1`)**. As candidatas são salvas separadamente e não substituem automaticamente o modelo ativo.

| Experimento | Comando | Melhor acurácia CV | F1 macro correspondente |
| --- | --- | ---: | ---: |
| DINOv2-base e padronização | `python experiments/compare_larger_backbone.py --evaluate-selected` | 51,93% (controle) | 49,24% |
| Adaptação do último bloco | `python experiments/adapt_last_block.py --evaluate-selected` | 43,96% | 40,66% |
| Resolução maior | `python experiments/compare_resolution.py` | 48,38% | 45,74% |
| SVM sobre vetores normalizados | `python experiments/compare_kernel.py` | 50,83% | 47,96% |
| Normalização por região | `python experiments/compare_region_normalization.py` | 51,93% | 49,22% |
| Combinação de classificadores | `python experiments/compare_ensemble.py` | 51,38% | 48,58% |

Foram 60 configurações adicionais. Os relatórios detalhados são gerados nas subpastas `larger_backbone`, `adapted`, `resolution`, `kernel`, `region_normalization` e `ensemble` de `artifacts/`, fora do Git. Os caches ficam em `data/`.

O DINOv3 não foi avaliado: o download oficial retornou `GatedRepoError`, exigindo liberação de acesso. O reaproveitamento das 63 imagens com rótulos conflitantes foi apenas proposto; o trabalho foi interrompido antes da implementação.

## Ampliação do treino com normalização

Para comparar o controle sem variações com uma e três variações por imagem:

```bash
python experiments/compare_augmentation.py --batch-size 2 --train-candidate
```

O experimento usa as mesmas cinco partes do treino (`seed=42`), com brilho,
contraste, nitidez, desfoque e ruído reproduzíveis. As variações de uma imagem
ficam somente no treino do respectivo fold; a validação usa imagens originais.
Não são aplicados cortes, rotações ou deformações. Os dois extratores recebem
exatamente a mesma variação.

Originais e variações usam RGB, redimensionamento proporcional com preenchimento,
pixels em `[0,1]` e média/desvio próprios do extrator. Os vetores recebem
normalização L2 por extrator e `StandardScaler` ajustado apenas no treino de cada
fold. O cache em `data/augmentation_cache/` identifica imagens, sementes,
transformações, normalização e revisões dos modelos.

O relatório `artifacts/augmentation/comparison.json` registra métricas por fold e
classe, previsões, matrizes de confusão e tempos. A seleção exige maior acurácia
média sem queda de F1 macro médio em relação ao controle; empates favorecem menos
variações. Se houver ganho, `--train-candidate` salva e avalia uma candidata em
`artifacts/augmentation/candidate/`, incluindo uma inferência com mapa visual.
O modelo padrão não é substituído. Use `--output-dir` para separar outras execuções.

As 364 imagens originais continuam sendo 364 exemplos independentes, mesmo com
1.456 amostras de treino ao usar três variações. O teste já foi consultado em
experimentos anteriores: seus resultados são comparativos e não orientam a seleção.

Na execução de 24/09/2026, a acurácia média de validação foi 51,93% sem variações,
51,11% com uma e 49,73% com três. O F1 macro também caiu; nenhuma candidata foi
selecionada e o modelo padrão foi mantido. Detalhes em `EXPERIMENT_LOG.md`.

## Situações adversas

- Docker ausente: instale Docker Engine/Compose ou use uma máquina preparada com a imagem offline.
- Daemon parado: inicie o serviço Docker e execute `./run.sh` novamente.
- GPU ausente: o launcher usa CPU; se houver NVIDIA detectada mas o runtime Docker falhar, use o comando explícito de CPU acima.
- Porta ocupada: defina `ALLIAGE_PORT` ou `ALLIAGE_API_PORT`.
- Internet indisponível: use a imagem exportada e `./run-offline.sh`.
- ZIP ausente: a demonstração funciona; somente o treinamento fica indisponível.
- API remota indisponível: remova `INFERENCE_URL` para processar localmente, desde que a máquina tenha recursos para isso.
- Chave inválida: a API retorna `401`; confira `ALLIAGE_API_KEY` no servidor e no cliente.
- Imagem inválida ou formato não aceito: envie JPG ou PNG válido com até 25 MB.

## Integridade dos dados

O pipeline calcula SHA-256 das imagens, remove imagens do teste do treinamento e elimina conflitos de rótulo. Os relatórios registram acurácia, F1 macro, métricas por classe, matriz de confusão e previsões. A validação cruzada da fusão atingiu **49,2% de F1 macro** em 364 imagens únicas de treino. Foram examinadas 144 configurações, portanto o melhor resultado de validação pode estar otimista.

A separação é por imagem, sem garantia de separação por paciente. A acurácia média do controle na validação é **51,93%**; os experimentos adicionais não demonstraram melhoria.

No teste recebido, a acurácia foi **34,6%** e o F1 macro **33,9%** nos 81 arquivos. Nos 59 exemplos sem rótulos conflitantes, foram **32,2%** e **33,5%**, respectivamente. A mediana de inferência com mapa visual foi **76 ms** na RTX 5060. O ganho no teste completo frente ao primeiro baseline é mínimo; o desempenho continua insuficiente para decisões operacionais sem revisão humana.

## Verificação

```bash
python -m unittest discover -p "test_*.py" -v
```

Os testes verificam separação treino/teste, reprodução das transformações, normalização, cache, resolução e equivalência entre treino com tokens em cache e inferência. Não baixam pesos nem executam treinamento completo.

Em 24/09/2026 foram verificados os 14 testes locais, a inferência na RTX 5060 e a classificação pela interface HTTP pública. Docker, containers, API FastAPI independente e execução offline não foram validados de ponta a ponta. A verificação local não valida a versão de PyTorch fixada no Dockerfile.

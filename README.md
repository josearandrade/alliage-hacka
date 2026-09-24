# Controle de posicionamento em radiografias panorâmicas

Aplicação de demonstração com interface Gradio e API HTTP para classificar seis condições de posicionamento. O mapa visual é aproximado; a ferramenta não faz diagnóstico.

## Recursos locais necessários

Este repositório versiona o código e a infraestrutura. Antes de construir a imagem, forneça no diretório do projeto:

- `data/images/` e `data/splits.json`, preparados pelo pipeline, incluindo as imagens referenciadas nos relatórios de avaliação;
- `data/hf_cache/`, com os modelos e revisões definidos em `pipeline.py`;
- `artifacts/training.json` e `artifacts/classifier.joblib` para o modelo baseline;
- para a interface com baseline, também `artifacts/test_raw.json`, `artifacts/test_clean.json`, `artifacts/inference_benchmark.json`, `artifacts/test_raw_confusion.png` e `artifacts/test_clean_confusion.png`;
- opcionalmente, `artifacts/fusion/training.json` e `artifacts/fusion/classifier.joblib` para a fusão, acompanhados dos pesos dos dois extratores no cache e dos mesmos cinco arquivos de avaliação acima, dentro de `artifacts/fusion/`.

Os relatórios, classificadores e imagens devem corresponder à mesma execução de preparação e treinamento. A interface lê os relatórios ao iniciar, inclusive no modo de API remota; apenas `training.json` e `classifier.joblib` não bastam. O launcher Python da interface também verifica a existência de `artifacts/classifier.joblib` no modo local, mesmo quando a fusão está selecionada. A API não exige os cinco arquivos de avaliação para inferência.

Dados, pesos, relatórios e caches não acompanham o commit. Um clone isolado não contém os recursos necessários ao build e à inferência. O container define `HF_HOME=/app/data/hf_cache` e `TRANSFORMERS_OFFLINE=1`; prepare o cache antes do build. A fusão é selecionada quando seu relatório de treinamento existe; caso contrário, usa-se o baseline.

## Linux e Docker

Requisitos: Docker Engine, plugin Docker Compose e acesso ao daemon. O build instala dependências pela rede e inclui os recursos locais na imagem.

```bash
./run.sh
```

A interface abre em `http://localhost:7860`. Para outra porta, use `ALLIAGE_PORT=9000 ./run.sh`.

O launcher seleciona GPU quando `nvidia-smi` está disponível e responde. Esse modo também exige driver e runtime NVIDIA configurados para Docker; erros desse runtime não provocam uma nova tentativa automática em CPU. Para iniciar explicitamente sem solicitar GPU:

```bash
docker compose -f docker-compose.yml up --build
```

## API

Inicie o serviço em um terminal:

```bash
./run-cloud.sh
```

Após a inicialização, execute em outro terminal, substituindo `radiografia.jpg` por uma imagem existente:

```bash
curl http://localhost:8080/healthz
curl -X POST http://localhost:8080/predict -F file=@radiografia.jpg -o resultado.json
```

A API aceita JPG ou PNG de até 25 MB e retorna classe, escores, tempo, dispositivo e mapa em PNG/base64. Configure `ALLIAGE_API_PORT` para alterar a porta e `ALLIAGE_API_KEY` para exigir o cabeçalho `X-API-Key`. Sem chave, o serviço não exige autenticação.

A interface Python aceita `INFERENCE_URL` e `ALLIAGE_API_KEY` para acessar uma API remota. O Compose atual não encaminha essas variáveis ao serviço de interface; configure-as explicitamente no container se usar esse modo.

## Execução offline

Prepare e exporte a imagem em uma máquina com acesso à rede e com os recursos locais:

```bash
docker compose -f docker-compose.yml build
docker save alliage-hacka:latest -o alliage-hacka.tar
```

Transfira o arquivo e os arquivos Compose/scripts para o destino:

```bash
./run-offline.sh alliage-hacka.tar
```

O script carrega a imagem e inicia a interface sem reconstruí-la.

## Testes

Em um ambiente Python com as dependências de `requirements.txt` instaladas:

```bash
python -m unittest test_pipeline -v
```

Os testes verificam a separação dos dados de treino/teste e a reprodutibilidade das transformações, sem baixar modelos nem executar treinamento.

### Estado da validação (24/09/2026)

- Os dois testes passaram no ambiente Python disponível, com PyTorch `2.11.0+cu128`. Isso não valida a versão `2.6.0` fixada em `requirements.txt` e no Dockerfile.
- A sintaxe dos scripts Bash e dos arquivos YAML foi verificada; isso não equivale à validação do Docker Compose ou do build.
- Foi reproduzida uma falha de inicialização por ausência de `test_raw.json` ao fornecer somente os dois artefatos baseline. A lista de pré-requisitos acima foi corrigida após esse teste.
- Build, inicialização dos containers e inferência HTTP ainda não foram validados: Docker estava indisponível, as instalações WSL não iniciavam e não havia uma imagem `alliage-hacka.tar` para testar o modo offline.

# Registro dos experimentos

As comparações de modelo usam somente as 364 imagens únicas do conjunto de treino, com validação cruzada estratificada em cinco partes (`seed=42`). O teste fornecido fica reservado para avaliar uma candidata selecionada sem consultar seus rótulos durante a escolha.

| Etapa | Mudança | Validação F1 macro | Teste completo F1 macro | Teste sem conflitos F1 macro | Decisão |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline inicial | RAD-DINO congelado + SVM RBF | não registrado | 0,338 | 0,286 | Referência |
| Trabalho paralelo existente | RAD-DINO congelado + regressão logística, `mean_spatial`, `C=1`, pesos balanceados | 0,471 | 0,317 | 0,285 | Não superou o teste do baseline; manter como experimento |

## 2026-09-24 — comparação independente de representações

- Material do desafio relido: apenas seis classes; teste fora do treinamento; métricas por classe, matriz de confusão, tempo e exemplos visuais necessários. O guia recomenda que erros de posicionamento não impliquem repetição automática do exame.
- A comparação em `experiments/compare_frozen_features.py` examina RAD-DINO, DINOv2-small e concatenação de ambos, com as mesmas cinco partes do treino. Resultados detalhados serão gravados em `artifacts/feature_comparison.json`.
- Resultado: a concatenação L2-normalizada de todas as oito regiões/representações de RAD-DINO e DINOv2-small, seguida de padronização e regressão logística balanceada (`C=0,1`), atingiu F1 macro de **0,4924 ± 0,0233** e acurácia de **0,5193** na validação cruzada. Superou a seleção anterior em quatro dos cinco folds (diferenças: +0,0395, +0,0124, −0,0382, +0,0421, +0,0516). Foram comparadas 144 configurações; esse número torna a estimativa do melhor resultado otimista.
- A candidata foi então treinada nas 364 imagens únicas de treino e avaliada uma vez no teste reservado por `experiments/evaluate_fusion_candidate.py`. O resultado completo ficou em `artifacts/fusion_candidate.json`.

| Etapa | Mudança | Validação F1 macro | Teste completo F1 macro | Teste sem conflitos F1 macro | Decisão |
| --- | --- | ---: | ---: | ---: | --- |
| Candidata de fusão | RAD-DINO + DINOv2-small congelados; regressão logística balanceada, `C=0,1` | 0,492 | 0,339 | 0,335 | Integrar e testar inferência; ganho no teste completo é pequeno |

A acurácia da candidata foi 34,6% no teste completo e 32,2% no teste sem conflitos. O teste não participou do ajuste dos pesos nem da seleção dos hiperparâmetros. O ganho de F1 macro no teste completo em relação ao baseline inicial é de cerca de 0,1 ponto percentual, portanto não deve ser apresentado como melhoria robusta nesse recorte.

## 2026-09-24 — ampliação por variações e normalização

- Experimento: `python experiments/compare_augmentation.py --batch-size 2 --train-candidate`.
- Controle: fusão congelada RAD-DINO + DINOv2-small, todas as representações, regressão logística balanceada com `C=0,1`.
- Comparação restrita a zero, uma ou três variações por original: 364, 728 ou 1.456 amostras no treino completo, sempre 364 imagens independentes. Brilho, contraste, nitidez, desfoque e ruído seguem os intervalos da função existente; nenhuma transformação geométrica foi acrescentada.
- As sementes derivam de SHA-256 da semente global, hash da imagem e índice da variação. Ambos os extratores recebem a mesma imagem transformada. O cache registra a implementação das transformações e da normalização, revisões dos modelos, hashes e sementes.
- Normalização: RGB, encaixe proporcional em 518 × 252, pixels divididos por 255, média/desvio do backbone, L2 por backbone e `StandardScaler` ajustado apenas no treino de cada fold. Vetores nulos permanecem nulos na normalização L2.
- Seleção: cinco folds estratificados (`seed=42`) separados por imagem original; validação somente com originais. Exigir acurácia média maior sem redução de F1 macro médio frente ao controle desta execução; empatar pelo menor número de variações. Não usar os testes para selecionar a configuração.
- Os testes fornecidos já foram consultados em experimentos anteriores. A avaliação adicional de uma candidata é comparativa, sem iniciar novos ajustes com base nesses resultados. O modelo padrão permanece preservado.

| Variações por original | Amostras no treino completo | Acurácia média CV | F1 macro médio CV |
| --- | ---: | ---: | ---: |
| 0 (controle) | 364 | 51,93% | 49,24% |
| 1 | 728 | 51,11% | 48,53% |
| 3 | 1.456 | 49,73% | 47,06% |

Resultado: nenhuma ampliação satisfez o critério de seleção. O controle reproduziu a comparação anterior, mas as variações reduziram ambas as métricas. Portanto, nenhuma candidata foi treinada e o teste reservado não foi reavaliado nesta execução. O modelo padrão foi mantido. A extração e os 15 ajustes de validação levaram aproximadamente 720 segundos na execução local.

Relatório completo: `artifacts/augmentation/comparison.json`, com previsões, composição dos folds, métricas por classe, matrizes de confusão, tempos e identificadores de cache. Os oito testes automatizados de integridade, normalização e aumento de dados passaram. A ampliação está implementada e reproduzível, mas não demonstrou ganho de acurácia nesta configuração.

A inferência completa do modelo atual também passou: os escores calculados a partir da imagem concordaram com os calculados usando as características originais em cache (`rtol=1e-3`, `atol=1e-4`). O resultado e o mapa visual estão em `artifacts/augmentation/current_model_smoke.json` e `current_model_smoke_overlay.png`.

## 2026-09-24 — comparação adicional com prazo de 30 minutos

O controle continuou sendo a fusão RAD-DINO + DINOv2-small, regressão logística balanceada (`C=0,1`), com acurácia média de 51,93% e F1 macro médio de 49,24% nos mesmos cinco folds. A condição de seleção foi acurácia maior sem queda de F1 macro. Nenhuma candidata satisfez essa condição; o teste fornecido não foi novamente avaliado e nenhum classificador ativo foi substituído.

| Hipótese | Configurações | Melhor acurácia CV | F1 macro correspondente |
| --- | ---: | ---: | ---: |
| DINOv2-base e padronização de vetores | 18, incluindo controle | 51,93% (controle) | 49,24% |
| Ajuste do último bloco DINOv2-small | 3 checkpoints: 10, 20 e 30 épocas | 43,96% | 40,66% |
| Resolução 1.036 × 504 e combinações com resolução original | 10 | 48,38% | 45,74% |
| SVM RBF sem padronização por coordenada | 18 | 50,83% | 47,96% |
| L2 independente por bloco espacial/global | 8 | 51,93% | 49,22% |
| Média de classificadores com regularização/balanceamento diferentes | 3 | 51,38% | 48,58% |

São 60 configurações adicionais, além dos experimentos anteriores; o máximo observado em validação não deve ser tratado como estimativa independente de generalização. A comparação de DINOv2-base também registrou uma segunda partição (`seed=17`), sem usá-la para escolher outra configuração.

O DINOv3 oficial retornou `GatedRepoError` durante a tentativa de obter `config.json`; os pesos não foram baixados e não há resultado de DINOv3. A proposta de incluir imagens com rótulos conflitantes com peso reduzido foi cancelada antes da implementação. As 63 imagens continuam excluídas do treino principal.

Os scripts de reprodução e os diretórios de resultados estão listados no README. Os resultados gerados ficam locais, fora do Git. A aplicação pública manteve o modelo original; acesso e inferência pelo túnel Cloudflare foram verificados após o encerramento dos experimentos.

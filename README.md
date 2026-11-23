# Compressão = previsão: transformers minúsculos em nível de caractere

Um modelo de linguagem treinado com perda de entropia cruzada é
um compressor: a codificação aritmética guiada pelas probabilidades do modelo transforma −log2 q(x) bits por
símbolo em um arquivo de verdade. Treinamos transformers pequenos, escritos do zero em PyTorch, na Wikipédia em
português e em inglês, e medimos isso de ponta a ponta.

**Resultados completos, com tabelas, figuras e a descrição de cada medida: [RESULTS.md](RESULTS.md).**

## Em poucas linhas

Bits por caractere no conjunto de teste (10 milhões de caracteres por língua, alfabeto de 49 símbolos):

| | gzip -9 | xz -9e | modelo 0,1M | modelo 1M | modelo 10M |
|---|---|---|---|---|---|
| português | 2,89 | 2,21 | 2,37 | 1,80 | 1,52 |
| inglês | 2,87 | 2,19 | 2,43 | 1,85 | 1,56 |

- O codificador aritmético decodifica tudo de volta bit a bit, e o tamanho real do arquivo fica a cerca de 0,02
  bit/caractere da soma ideal Σ −log2 q.
- Usar o modelo da outra língua custa caro, e de forma assimétrica: inglês codificado pelo modelo português custa
  2,31 bits/caractere (10M), português codificado pelo modelo inglês custa 3,05.
- Também medimos calibração, perda por posição no contexto, identificação de língua (previsão pelo lema de Stein
  contra erro medido) e comprimento de descrição (modelo compartilhado, duas partes e prequencial).

## Como reproduzir

Precisa de [uv](https://docs.astral.sh/uv/), uma GPU com CUDA (usamos uma RTX 3060 de 12 GB, compartilhada) e
cerca de 5 GB de disco. O modelo de 10M leva cerca de 1h20 por língua; os de 0,1M e 1M, minutos.

```bash
bash setup.sh                                   # .venv com torch, numpy, pyarrow, matplotlib
bash download.sh                                # Wikipédia 2023-11-01 (parquet) em raw/
.venv/bin/python prepare_data.py raw data       # normaliza e divide treino/validação/teste por artigo
.venv/bin/python ngram_entropy.py data results/ngram.json
bash compressors.sh data > results/compressors.tsv
bash run_gpu.sh > run_gpu.log 2>&1              # treina 0,1M/1M/10M em pt e en, avalia, codifica, identifica língua
.venv/bin/python mdl.py runs results/mdl.json
.venv/bin/python figures.py results figures     # depois de copiar runs/ (logs, eval/, coder/, lid.json) para results/runs/
bash make_results.sh                            # reescreve as tabelas de RESULTS.md a partir de results/
```

Verificação opcional no text8: `bash text8_data.sh data`, depois `.venv/bin/python train.py text8 1M data runs`.

## Arquivos

Um script por etapa, cerca de 1.000 linhas no total.

| arquivo | o que faz |
|---|---|
| `download.sh`, `prepare_data.py` | baixa a Wikipédia e produz o texto normalizado e os conjuntos |
| `ngram_entropy.py`, `compressors.sh` | escada de entropia por n-gramas; gzip, xz e zstd |
| `model.py`, `train.py` | o transformer e o treino de uma época, com comprimento de código prequencial |
| `evaluate.py` | entropia cruzada no teste, matriz entre línguas, calibração, perda por posição |
| `coder.py` | codificador aritmético inteiro próprio e teste de ida e volta bit a bit |
| `lid.py` | identificação de língua: previsão e medida |
| `mdl.py` | comprimento de descrição |
| `tables.py`, `figures.py`, `make_results.sh` | geram RESULTS.md e `figures/` a partir de `results/` |
| `results/` | os resultados brutos (JSON, TSV e logs) usados nas tabelas |

Pesos dos modelos e textos não estão no repositório; os scripts os reconstroem.

## Dados e créditos

Texto da Wikipédia (dataset [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), dumps
de 2023-11-01), licenciado sob CC BY-SA 3.0 e GFDL. Este repositório não redistribui o texto, apenas números
derivados dele. O text8 é de Matt Mahoney.

O código, a execução dos experimentos e o texto de RESULTS.md foram produzidos com a ajuda do Claude (Anthropic),
um assistente de IA.

## Licença

Código sob a licença MIT (veja [LICENSE](LICENSE)).

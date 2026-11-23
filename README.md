# Compressão é previsão: transformers character-level na Wikipédia

Um modelo de linguagem treinado com entropia cruzada também é um compressor. Se o modelo dá probabilidade q(x) ao
próximo caractere, um codificador aritmético gasta cerca de −log2 q(x) bits para guardá-lo. Ou seja, o arquivo
comprimido fica com o tamanho da loss do modelo.

Neste projeto, treinamos transformers pequenos do zero, em PyTorch, com a Wikipédia em português e em inglês, e
verificamos essa ideia na prática: do treino até o arquivo comprimido, e de volta.

**Os resultados completos, com tabelas, figuras e a explicação de cada medida, estão em [RESULTS.md](RESULTS.md).**

## Em poucas linhas

Bits por caractere (bpc) no conjunto de teste, com 10 milhões de caracteres por língua e um alfabeto de 49 símbolos:

| | gzip -9 | xz -9e | modelo 0,1M | modelo 1M | modelo 10M |
|---|---|---|---|---|---|
| português | 2,89 | 2,21 | 2,37 | 1,80 | 1,52 |
| inglês | 2,87 | 2,19 | 2,43 | 1,85 | 1,56 |

- **A compressão funciona de verdade:** o codificador aritmético recupera o texto original bit a bit, e o arquivo
  fica só cerca de 0,02 bpc acima do tamanho ideal Σ −log2 q.
- **Usar o modelo da outra língua custa caro, e o custo não é simétrico:** com os modelos de 10M, o inglês codificado
  pelo modelo português custa 2,31 bpc, e o português codificado pelo modelo inglês custa 3,05 bpc.
- **Também medimos:** calibração, loss por posição no contexto, identificação de língua (uma previsão pelo lema de
  Stein comparada com o erro medido) e comprimento de descrição (modelo compartilhado, two-part code e prequential
  code length).

## Como rodar

Você vai precisar do [uv](https://docs.astral.sh/uv/) para criar o ambiente Python, de uma GPU com CUDA e de cerca de
5 GB de disco. Nós usamos uma RTX 3060 de 12 GB, compartilhada com outros trabalhos. O modelo de 10M leva cerca de
1h20 por língua; os de 0,1M e 1M levam minutos.

Rode os comandos na raiz do repositório, nesta ordem:

```bash
bash src/setup.sh                                   # cria .venv com torch, numpy, pyarrow e matplotlib
bash src/download.sh                                # baixa os dumps da Wikipédia de 2023-11-01 (parquet) em raw/
.venv/bin/python src/prepare_data.py raw data       # normaliza o texto e divide treino/validação/teste por artigo
.venv/bin/python src/ngram_entropy.py data results/ngram.json
bash src/compressors.sh data > results/compressors.tsv
bash src/run_gpu.sh > run_gpu.log 2>&1              # treina 0,1M/1M/10M em pt e en, avalia, codifica, identifica língua
.venv/bin/python src/mdl.py runs results/mdl.json
.venv/bin/python src/figures.py results figures     # antes, copie runs/ (logs, eval/, coder/, lid.json) para results/runs/
bash src/make_results.sh                            # reescreve as tabelas de RESULTS.md a partir de results/
```

Há também uma verificação de sanidade opcional no text8, um benchmark clássico de texto em inglês: rode
`bash src/text8_data.sh data` e depois `.venv/bin/python src/train.py text8 1M data runs`.

## Estrutura

Todo o código fica em `src/`, com um script por etapa e cerca de 1.000 linhas no total.

- `src/download.sh`, `src/prepare_data.py`: baixam a Wikipédia, normalizam o texto e montam os conjuntos de treino,
  validação e teste;
- `src/ngram_entropy.py`, `src/compressors.sh`: baselines sem rede neural (entropia por n-gramas; gzip, xz e zstd);
- `src/model.py`, `src/train.py`: o transformer e o treino de uma época, com prequential code length;
- `src/evaluate.py`: entropia cruzada no teste, matriz entre línguas, calibração e loss por posição;
- `src/coder.py`: o codificador aritmético e o teste de round trip;
- `src/lid.py`: identificação de língua, com previsão e medida;
- `src/mdl.py`: comprimento de descrição;
- `src/tables.py`, `src/figures.py`, `src/make_results.sh`: geram RESULTS.md e as figuras a partir de `results/`;
- `results/`: os resultados brutos (JSON, TSV e logs) usados nas tabelas;
- `figures/`: as figuras usadas em RESULTS.md.

Os pesos dos modelos e o texto não estão no repositório; os scripts reconstroem tudo.

## Dados e créditos

O texto vem da Wikipédia, pelo dataset [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
(dumps de 2023-11-01), sob as licenças CC BY-SA 3.0 e GFDL. Este repositório não redistribui o texto, só números
calculados a partir dele. O text8 é de Matt Mahoney.

O código, a execução dos experimentos e o texto de RESULTS.md foram produzidos com a ajuda do Claude (Anthropic), um
assistente de IA.

## Licença

O código está sob a licença MIT (veja [LICENSE](LICENSE)).

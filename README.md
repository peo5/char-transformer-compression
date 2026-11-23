# Compressão = previsão: transformers minúsculos em nível de caractere

Um modelo de linguagem treinado com perda de entropia cruzada é um compressor. A codificação aritmética, guiada pelas
probabilidades q(x) do modelo, transforma −log2 q(x) bits por símbolo em um arquivo de verdade. Escrevemos
transformers pequenos do zero em PyTorch, treinamos esses modelos na Wikipédia em português e em inglês e medimos o
processo de ponta a ponta.

**Os resultados completos, com tabelas, figuras e a descrição de cada medida, estão em [RESULTS.md](RESULTS.md).**

## Em poucas linhas

Bits por caractere no conjunto de teste (10 milhões de caracteres por língua, alfabeto de 49 símbolos):

| | gzip -9 | xz -9e | modelo 0,1M | modelo 1M | modelo 10M |
|---|---|---|---|---|---|
| português | 2,89 | 2,21 | 2,37 | 1,80 | 1,52 |
| inglês | 2,87 | 2,19 | 2,43 | 1,85 | 1,56 |

- O codificador aritmético recupera o texto original bit a bit. O tamanho real do arquivo fica cerca de 0,02 bit por
  caractere acima da soma ideal Σ −log2 q.
- Usar o modelo da outra língua custa caro, e o custo não é simétrico. Com o modelo de 10M, o texto em inglês
  codificado pelo modelo português custa 2,31 bits por caractere. O texto em português codificado pelo modelo inglês
  custa 3,05.
- Também medimos a calibração dos modelos, a perda por posição no contexto, a identificação de língua (uma previsão
  pelo lema de Stein comparada com o erro medido) e o comprimento de descrição (modelo compartilhado, código em duas
  partes e código prequencial).

## Como reproduzir

É preciso ter o [uv](https://docs.astral.sh/uv/), um gerenciador de ambientes Python, uma GPU com CUDA e cerca de
5 GB de disco. Usamos uma RTX 3060 de 12 GB, compartilhada com outros trabalhos. O modelo de 10M leva cerca de 1h20
por língua; os de 0,1M e 1M levam minutos.

Rode os comandos abaixo na raiz do repositório, nesta ordem:

```bash
bash src/setup.sh                                   # cria .venv com torch, numpy, pyarrow e matplotlib
bash src/download.sh                                # baixa a Wikipédia de 2023-11-01 (parquet) em raw/
.venv/bin/python src/prepare_data.py raw data       # normaliza e divide treino/validação/teste por artigo
.venv/bin/python src/ngram_entropy.py data results/ngram.json
bash src/compressors.sh data > results/compressors.tsv
bash src/run_gpu.sh > run_gpu.log 2>&1              # treina 0,1M/1M/10M em pt e en, avalia, codifica, identifica língua
.venv/bin/python src/mdl.py runs results/mdl.json
.venv/bin/python src/figures.py results figures     # antes, copie runs/ (logs, eval/, coder/, lid.json) para results/runs/
bash src/make_results.sh                            # reescreve as tabelas de RESULTS.md a partir de results/
```

Há também uma verificação opcional no text8, um conjunto padrão de texto em inglês: rode
`bash src/text8_data.sh data` e depois `.venv/bin/python src/train.py text8 1M data runs`.

## Arquivos

Todos os scripts ficam em `src/`, um script por etapa, com cerca de 1.000 linhas no total.

| arquivo | o que faz |
|---|---|
| `download.sh`, `prepare_data.py` | baixam a Wikipédia e produzem o texto normalizado e os conjuntos de treino, validação e teste |
| `ngram_entropy.py`, `compressors.sh` | calculam a escada de entropia por n-gramas e rodam gzip, xz e zstd |
| `model.py`, `train.py` | definem o transformer e o treinam por uma época, medindo o comprimento de código prequencial |
| `evaluate.py` | mede a entropia cruzada no teste, a matriz entre línguas, a calibração e a perda por posição |
| `coder.py` | implementa o codificador aritmético e testa a ida e volta bit a bit |
| `lid.py` | faz a identificação de língua: a previsão e a medida |
| `mdl.py` | calcula o comprimento de descrição |
| `tables.py`, `figures.py`, `make_results.sh` | geram RESULTS.md e as figuras a partir de `results/` |

A pasta `results/` guarda os resultados brutos (JSON, TSV e logs) usados nas tabelas. Os pesos dos modelos e os textos
não estão no repositório; os scripts os reconstroem.

## Dados e créditos

O texto vem da Wikipédia, pelo dataset [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
(dumps de 2023-11-01), sob as licenças CC BY-SA 3.0 e GFDL. Este repositório não redistribui o texto, só números
calculados a partir dele. O text8 é de Matt Mahoney.

O código, a execução dos experimentos e o texto de RESULTS.md foram produzidos com a ajuda do Claude (Anthropic), um
assistente de IA.

## Licença

O código está sob a licença MIT (veja [LICENSE](LICENSE)).

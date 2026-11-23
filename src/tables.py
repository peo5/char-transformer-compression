"""Print the generated part of RESULTS.md from the result files, in Portuguese. Each section gets a short
description of what was measured and how (the TEXT dictionary below), then its tables and figures. Every number in
the tables is read from a result file, never typed by hand.

Usage: python src/tables.py RESULTS_DIR > tables.md        (make_results.sh does this)
"""
import csv, json, sys
from pathlib import Path

SIZES = ("0.1M", "1M", "10M")

# Prose of each RESULTS.md section (pt-BR). {n_symbols} and {min_chars} are filled from results/stats.json.
TEXT = {
"dados": """\
Usamos os dumps da Wikipédia de 1º de novembro de 2023 publicados no Hugging Face (`wikimedia/wikipedia`). Do
português (`20231101.pt`), usamos os 6 arquivos Parquet; do inglês (`20231101.en`), os arquivos 0, 8, 16, 24, 32 e 40
de 41.

A normalização perde informação de propósito, para que as duas línguas usem o mesmo alfabeto. Aplicamos a
normalização Unicode NFKD, que separa cada letra do seu acento, e jogamos os acentos fora (assim "ç" vira "c").
Depois passamos tudo para minúsculas, trocamos aspas curvas e « » por `"` ou `'` e travessões por `-`, e descartamos
qualquer caractere fora de a–z, 0–9, espaço, quebra de linha e `. , ; : ! ? ' " - ( )`.

Sobram {n_symbols} símbolos. Sem modelo nenhum, com todos os símbolos igualmente prováveis, cada caractere custaria
log2({n_symbols}) ≈ 5,61 bits. Esse é o teto de referência para todos os números abaixo.

No texto final, cada linha é um parágrafo, e uma linha vazia separa um artigo do outro. Descartamos os artigos com
menos de {min_chars} caracteres depois da normalização: em geral, são stubs ou artigos criados por bots. Também
descartamos duplicatas exatas e quase-duplicatas, ou seja, artigos que só diferem nos dígitos.

Para dividir o corpus, ordenamos os artigos pelo hash MD5 de "língua:id" e os distribuímos, nessa ordem, para teste,
validação e treino. Assim, nenhum artigo aparece em dois conjuntos, e o treino é uma amostra aleatória do que
sobrou. Cada conjunto tem exatamente o mesmo número de caracteres nas duas línguas; para isso, cortamos o último
artigo de cada um.""",

"referencias": """\
**Entropia condicional por tamanho de contexto** (*entropy ladder*; `ngram_entropy.py`, `results/ngram.json`). A
pergunta aqui é: quanto custa o próximo caractere se conhecemos os k anteriores? Contamos n-gramas (sequências de n
caracteres) nos primeiros N caracteres do treino, com N = 10M e N = 100M, e calculamos
H(X_n | k caracteres anteriores) = H((k+1)-gramas) − H(k-gramas).

Essa é a estimativa plug-in: ela usa as frequências observadas como se fossem as probabilidades verdadeiras. Quando
há muitos contextos diferentes em comparação com N, a maioria aparece poucas vezes, e a estimativa fica baixa
demais. Comparar N = 10M com N = 100M mostra a partir de qual k isso começa a acontecer.

**Compressores** (`compressors.sh`, `results/compressors.tsv`). Comprimimos os arquivos de teste normalizados (10 MB
por língua) com gzip, xz e zstd. Se um arquivo de N caracteres vira B bytes, o custo é 8B/N bpc.""",

"modelos": """\
O modelo é um transformer decoder-only pequeno, escrito do zero em PyTorch. Cada token é um caractere, e o contexto
tem 256 caracteres. Treinamos três tamanhos por língua: cerca de 0,1M, 1M e 10M de parâmetros.

Cada modelo treina por uma única época. Cada trecho de 256 caracteres do corpus de treino (400M de caracteres)
aparece uma vez, em ordem aleatória (seed 0). Usamos batches de 64 trechos, AdamW, learning rate com warmup e cosine
decay, e bf16.

**Prequential code length.** Antes de atualizar os pesos com um batch, medimos a loss do modelo nesse batch, que ele
ainda não viu. Somando essas losses no treino inteiro, temos o tamanho de um código para os dados de treino que não
paga nada pelos pesos. Isso vale porque um decodificador que conhece a seed e a receita de treino consegue refazer
as mesmas atualizações.

A validação usa os 5M de caracteres de val.txt. As curvas de validação da figura usam só os primeiros 0,5M; os
números finais da tabela usam os 5M.""",

"teste": """\
Cortamos cada arquivo de teste (10M de caracteres) em trechos de 256 caracteres, sem sobreposição. Cada trecho é
avaliado sozinho, sem o texto que vem antes dele: a posição i prevê o caractere i+1 a partir dos caracteres 0..i.
Aqui tudo roda em float32, sem mixed precision.

**Calibração.** O modelo também diz o quanto está incerto: é a entropia da distribuição que ele prevê. Comparamos a
média dessa entropia com a entropia cruzada medida de fato. Quando as duas coincidem, a incerteza que o modelo
declara é a incerteza real. Quando a entropia prevista é menor, o modelo está confiante demais.

**Loss por posição.** Calculamos a entropia cruzada média em cada posição do trecho. Isso mostra quanto o modelo
ganha a cada caractere a mais de contexto.""",

"codificacao": """\
Implementamos nosso próprio codificador aritmético, com inteiros de 32 bits, no esquema clássico de Witten, Neal e
Cleary. A cada passo, o modelo dá uma distribuição sobre os 49 símbolos, e o codificador gasta cerca de −log2 q bits
no caractere que de fato aparece.

O ponto delicado é que o codificador e o decodificador precisam ver exatamente as mesmas probabilidades. Uma
diferença no último bit de um float já corrompe todo o resto do arquivo. Por isso, os dois rodam o mesmo laço, em
float32 e com algoritmos determinísticos. As probabilidades viram frequências inteiras que somam 2^16, com
f = 1 + piso(p·(2^16 − 49)), e a sobra vai para o símbolo mais provável.

O texto é dividido em 512 streams contíguos, processados em paralelo, um caractere por passo. Cada stream usa uma
janela deslizante com os últimos 256 caracteres. Para testar, codificamos o primeiro 1 MB do teste, decodificamos e
comparamos com o original (round trip).

O tamanho ideal é a soma Σ −log2 q_quantizada. O tamanho real é o tamanho do arquivo, incluindo um cabeçalho de 2056
bytes (8 bytes mais 4 bytes por stream). A diferença entre os dois é o overhead do codificador.

Aqui o contexto vem de uma janela deslizante (até 255 caracteres antes), e não de trechos independentes como na
seção 4. Por isso, os números desta seção não são exatamente iguais aos de lá.""",

"matriz": """\
Esta seção reorganiza os números da seção 4 em uma matriz. Cada texto de teste, com o mesmo tamanho nas duas
línguas, é codificado pelo modelo da própria língua e pelo modelo da outra. Por exemplo, "en pelo pt" é o texto em
inglês codificado pelo modelo treinado em português.

A intuição vem da identidade H(P,Q) = H(P) + KL(P‖Q): usar o modelo errado custa bits extras. O excesso
CE(en | modelo pt) − CE(en | modelo en), em que CE é a entropia cruzada, estima KL(p_en‖q_pt) − KL(p_en‖q_en). Aqui
p_en é a distribuição verdadeira do inglês e q é a distribuição de cada modelo. Mostramos as duas direções porque a
divergência KL não é simétrica.""",

"lid": """\
A ideia é usar os dois modelos para descobrir em que língua um trecho está. Para um trecho escrito na língua A,
calculamos para cada caractere a razão de log-verossimilhança L_i = log2 q_A(x_i) − log2 q_B(x_i), em que B é a outra
língua. A regra classifica o trecho como A quando Σ L_i > 0, ou seja, quando o modelo de A comprime o trecho melhor.

Primeiro fazemos uma previsão. Na primeira metade dos trechos de teste, medimos a média D e o desvio padrão de L_i, e
estimamos quantos caracteres n são necessários para errar com probabilidade eps:

- **lema de Stein:** n ≈ log2(1/eps)/D;
- **teorema central do limite (TCL):** tratando os L_i como independentes, n ≈ (z_eps · desvio / D)², em que z_eps é
  o quantil da normal padrão para 1 − eps.

Depois medimos. Na segunda metade, que é texto diferente, calculamos o erro real da regra para cada n de 1 a 256,
usando os primeiros n caracteres de cada trecho. O "n medido" é o menor n com erro ≤ eps; "—" quer dizer que esse
erro não foi alcançado em 256 caracteres.

Olhamos os trechos de teste com mais erros (modelos de 0,1M). No texto em português, são listas de anos sem palavras
("1834 1835 1836 ..."). No texto em inglês, são listas de títulos de livros em português dentro de artigos da
Wikipédia em inglês.""",

"mdl": """\
Uma objeção natural é que o modelo também precisa ser enviado para que o arquivo possa ser decodificado. Por isso,
contamos os bpc de cada modelo de quatro maneiras, com cada peso custando 32 bits:

- **Modelo compartilhado:** a entropia cruzada no teste. É o custo quando quem recebe o arquivo já tem o modelo, como
  acontece com um dicionário.
- **Two-part code, teste:** (32 × parâmetros + 10M × bpc no teste) / 10M. É o custo de enviar os pesos junto com o
  arquivo de teste de 10 MB.
- **Two-part code, treino:** a mesma conta para os 400M caracteres de treino. Como loss no treino, usamos a loss de
  validação final, já que o modelo viu cada caractere do treino uma única vez.
- **Prequential code length:** o valor da seção 3, que não paga nada pelos pesos.

Como referência, citamos sem executar o recorde do Hutter Prize: 110.793.128 bytes para o enwik9 (fx2-cmix, setembro
de 2024), ou 0,886 bits por byte de XML bruto, já contando o descompressor. Esse número não é comparável com os
nossos, porque o texto, o alfabeto e o tamanho são outros.""",

"text8": """\
O text8 é um benchmark clássico: os primeiros 10^8 caracteres do enwik9 (um trecho da Wikipédia em inglês), limpos
para conter só a–z e espaço. A divisão padrão usa 90M de caracteres para treino, 5M para validação e 5M para teste.

Treinamos o modelo de 1M de parâmetros por uma época nos 90M de treino, com a mesma receita dos outros modelos. O
número "teste" abaixo é a avaliação final nos 5M de teste, guardados como val.txt.

Resultados publicados no text8 em character-level ficam em torno de 1,1 a 1,5 bpc, mas vêm de modelos maiores,
treinados por muitas épocas. Este teste serve só como verificação de sanidade do nosso pipeline.""",
}


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def thousands(n):
    return f"{n:,}".replace(",", ".")


def num(x, digits=4):
    return f"{x:.{digits}f}".replace(".", ",")


def section(title, text):
    print(f"## {title}\n\n{text}\n")


def main():
    res = Path(sys.argv[1])
    runs = res / "runs"
    stats, ngram = load(res / "stats.json"), load(res / "ngram.json")
    comp = list(csv.DictReader(open(res / "compressors.tsv"), delimiter="\t"))
    logs = {p.name.removesuffix(".log.json"): json.loads(p.read_text()) for p in runs.glob("*.log.json")}
    summary, lid, mdl = load(runs / "eval/summary.json"), load(runs / "lid.json"), load(res / "mdl.json")
    coder = {p.stem: json.loads(p.read_text()) for p in (runs / "coder").glob("*.json")}
    text8 = load(res / "runs_text8/text8_1M.log.json")
    models = [f"{lang}_{size}" for size in SIZES for lang in ("pt", "en") if f"{lang}_{size}" in logs]

    section("1. Dados (`prepare_data.py`, `results/stats.json`)",
            TEXT["dados"].format(n_symbols=len(stats["recipe"]["alphabet"]),
                                   min_chars=stats["recipe"]["min_chars"]))
    rows = [["artigos lidos"] + [thousands(stats[l]['articles']) for l in ("pt", "en")],
            ["descartados, < 2000 caracteres"] + [thousands(stats[l]['short']) for l in ("pt", "en")],
            ["descartados, duplicata"] + [thousands(stats[l]['duplicate']) for l in ("pt", "en")],
            ["mantidos (caracteres)"] + [f"{thousands(stats[l]['kept'])} ({thousands(stats[l]['kept_chars'])})" for l in ("pt", "en")]]
    for split, name in (("test", "teste"), ("val", "validação"), ("train", "treino")):
        rows.append([f"{name}: caracteres (artigos)"] +
                    [f"{thousands(stats[l][split]['chars'])} ({thousands(stats[l][split]['articles'])})" for l in ("pt", "en")])
    table(["", "pt", "en"], rows)

    section("2. Baselines sem rede neural",
            TEXT["referencias"])
    table(["k", "pt, N = 10M", "pt, N = 100M", "en, N = 10M", "en, N = 100M"],
          [[k] + [num(ngram[f"{l}_{n}"][k], 3) for l in ("pt", "en") for n in ("10M", "100M")] for k in range(6)])
    table(["compressor", "pt bytes", "pt bpc", "en bytes", "en bpc"],
          [[f"`{c}`"] + [x for l in ("pt", "en") for r in comp if r["lang"] == l and r["compressor"] == c
                         for x in (thousands(int(r['bytes'])), num(float(r["bits_per_char"]), 3))]
           for c in dict.fromkeys(r["compressor"] for r in comp)])
    print("![entropia por tamanho de contexto](figures/entropy_ladder.png)\n")

    section("3. Modelos (`model.py`, `train.py`, `runs/*.log.json`)",
            TEXT["modelos"])
    table(["modelo", "parâmetros", "prequential code length, bpc (400M de treino)", "validação final, bpc",
           "minutos", "VRAM máx. (GB)"],
          [[m, thousands(logs[m]['params']), num(logs[m]["prequential_bpc"]), num(logs[m]["val_bpc"]),
            f"{logs[m]['seconds'] / 60:.0f}", num(logs[m]["max_vram_gb"], 2)] for m in models])
    print("![curvas de treino](figures/training_curves.png)\n")

    section("4. Entropia cruzada no teste, calibração e loss por posição (`evaluate.py`, `runs/eval/summary.json`)",
            TEXT["teste"])
    table(["modelo", "texto de teste", "bpc", "entropia média prevista", "entropia − bpc"],
          [[m, t, num(summary[f"{m}_on_{t}"]["bpc"]), num(summary[f"{m}_on_{t}"]["mean_entropy"]),
            num(summary[f"{m}_on_{t}"]["mean_entropy"] - summary[f"{m}_on_{t}"]["bpc"])]
           for m in models for t in ("pt", "en") if f"{m}_on_{t}" in summary])
    print("![loss por posição](figures/loss_by_position.png)\n")

    section("5. Codificação aritmética (`coder.py`, `runs/coder/*.json`)",
            TEXT["codificacao"])
    table(["modelo", "round trip", "bytes do arquivo", "real, bpc", "ideal, bpc",
           "real − ideal", "codificação (s)", "decodificação (s)"],
          [[m, "OK" if coder[m]["round_trip_ok"] else "FALHOU", thousands(coder[m]['file_bytes']),
            num(coder[m]["real_bpc"]), num(coder[m]["ideal_bpc"]), num(coder[m]["real_bpc"] - coder[m]["ideal_bpc"]),
            f"{coder[m]['encode_seconds']:.0f}", f"{coder[m]['decode_seconds']:.0f}"] for m in models if m in coder])

    section("6. Matriz entre línguas (`evaluate.py`, `runs/eval/summary.json`)",
            TEXT["matriz"])
    rows = []
    for size in SIZES:
        k = lambda model, text: summary.get(f"{model}_{size}_on_{text}", {}).get("bpc")
        pp, ee, pe, ep = k("pt", "pt"), k("en", "en"), k("pt", "en"), k("en", "pt")
        if None not in (pp, ee, pe, ep):
            rows.append([size, num(pp), num(ee), num(pe), num(ep), num(pe - ee), num(ep - pp)])
    table(["tamanho", "pt pelo pt", "en pelo en", "en pelo pt", "pt pelo en",
           "excesso no texto en (modelo pt − modelo en)", "excesso no texto pt (modelo en − modelo pt)"], rows)

    section("7. Identificação de língua: uma previsão testada (`lid.py`, `runs/lid.json`)",
            TEXT["lid"])
    table(["tamanho", "texto", "D (bpc)", "desvio", "eps", "n previsto, Stein", "n previsto, TCL",
           "n medido", "erro medido com n = 256"],
          [[key.split("_")[0], key.split("_")[1], num(r["D_bits"], 3), num(r["std_bits"], 3), str(eps).replace(".", ","),
            num(p["n_stein"], 1), num(p["n_clt"], 1), p["n_measured"] or "—", num(r["error_by_n"][-1])]
           for key, r in lid.items() for eps, p in r["eps"].items()])
    print("![identificação de língua](figures/language_id.png)\n")

    section("8. Comprimento de descrição (`mdl.py`, `results/mdl.json`)",
            TEXT["mdl"])
    table(["modelo", "pesos (Mbit)", "modelo compartilhado", "two-part, 10M de teste", "two-part, 400M de treino",
           "prequential, 400M de treino"],
          [[m, num(mdl[m]["weights_bits"] / 1e6, 1), num(mdl[m]["shared"]), num(mdl[m]["two_part_test"]),
            num(mdl[m]["two_part_train"]), num(mdl[m]["prequential"])] for m in models if m in mdl])

    if text8:
        section("9. Verificação de sanidade: text8 (`text8_data.sh`, `results/runs_text8/text8_1M.log.json`)",
                TEXT["text8"])
        table(["modelo", "parâmetros", "prequential code length, bpc (90M de treino)", "teste, bpc (5M)",
               "minutos"],
              [["text8_1M", thousands(text8['params']), num(text8["prequential_bpc"]), num(text8["val_bpc"]),
                f"{text8['seconds'] / 60:.0f}"]])


if __name__ == "__main__":
    main()

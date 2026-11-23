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
português (`20231101.pt`) usamos os 6 arquivos Parquet; do inglês (`20231101.en`), os arquivos 0, 8, 16, 24, 32 e 40
de 41.

A normalização perde informação de propósito. Decompomos o texto com a normalização Unicode NFKD, que separa cada
letra do seu acento, e descartamos os acentos (assim "ç" vira "c"). Passamos tudo para minúsculas, trocamos aspas
curvas e « » por `"` ou `'` e travessões por `-`. Por fim, descartamos todo caractere fora de a–z, 0–9, espaço,
quebra de linha e `. , ; : ! ? ' " - ( )`.

Sobram {n_symbols} símbolos. Um modelo que não sabe nada e dá a mesma probabilidade a todos eles gasta
log2({n_symbols}) ≈ 5,61 bits por caractere: esse é o teto de referência. No texto final, cada linha é um parágrafo e
uma linha vazia separa um artigo do outro.

Descartamos os artigos com menos de {min_chars} caracteres depois da normalização, que em geral são esboços ou foram
criados por robôs. Também descartamos duplicatas exatas e quase-duplicatas, isto é, artigos que ficam iguais quando
trocamos todo dígito por 0.

Para dividir os dados, ordenamos os artigos pelo hash MD5 do texto "língua:id" e os distribuímos primeiro para o
teste, depois para a validação e por fim para o treino. Assim, nenhum artigo aparece em dois conjuntos. Cada
conjunto tem exatamente o mesmo número de caracteres nas duas línguas; para isso, o último artigo de cada conjunto
é cortado.""",

"referencias": """\
**Escada de entropia** (`ngram_entropy.py`, `results/ngram.json`). Contamos n-gramas (sequências de n caracteres)
nos primeiros N caracteres do treino, com N = 10M e N = 100M. Com essas contagens calculamos a entropia condicional
H(X_n | k caracteres anteriores) = H((k+1)-gramas) − H(k-gramas). Chamamos essa estimativa de "plug-in" porque ela
usa as frequências observadas como se fossem as probabilidades verdadeiras.

A estimativa plug-in fica abaixo do valor verdadeiro quando há muitos contextos distintos em comparação com N, pois
muitos contextos aparecem poucas vezes. Comparar N = 10M com N = 100M mostra a partir de qual k isso acontece.

**Compressores** (`compressors.sh`, `results/compressors.tsv`). Comprimimos os arquivos de teste normalizados
(10 MB por língua) com gzip, xz e zstd. Os bits por caractere são 8 × bytes comprimidos / caracteres.""",

"modelos": """\
O modelo é um transformer mínimo, só com decodificador (*decoder-only*), escrito à mão em PyTorch. Cada token é um
caractere, e o modelo enxerga até 256 caracteres de contexto. Treinamos três tamanhos por língua, com cerca de 0,1M,
1M e 10M de parâmetros.

Cada modelo foi treinado por exatamente uma época, ou seja, viu cada trecho de 256 caracteres do treino (400M de
caracteres) uma única vez, em ordem aleatória fixada pela semente 0. Usamos lotes de 64 trechos e o otimizador
AdamW (Adam com decaimento de pesos). A taxa de aprendizado sobe no início (aquecimento) e depois cai seguindo um
cosseno. Os cálculos de treino usam bf16, um formato de ponto flutuante de 16 bits.

**Comprimento de código prequencial.** Avaliamos cada lote ANTES de atualizar o modelo com ele. A soma dessas perdas
é um comprimento de código para os dados de treino que não cobra nada pelos pesos. Isso vale porque um
decodificador que conhece a semente e a receita de treino pode refazer as mesmas atualizações.

A validação usa os 5M de caracteres de val.txt. As curvas de validação da figura usam só os primeiros 0,5M de
caracteres de val.txt.""",

"teste": """\
Cortamos cada arquivo de teste (10M de caracteres) em trechos de 256 caracteres sem sobreposição. Cada trecho é
avaliado sem nenhum contexto anterior a ele: a posição i prevê o caractere i+1 a partir dos caracteres 0..i. Aqui
os cálculos são em float32, sem precisão mista automática (AMP).

**Calibração.** Comparamos a entropia média da distribuição que o modelo prevê com a entropia cruzada medida. Se as
duas coincidem, o modelo sabe o quanto não sabe.

**Perda por posição.** Calculamos a entropia cruzada média em cada posição do trecho. Isso mostra como a perda cai
conforme o modelo tem mais caracteres de contexto.""",

"codificacao": """\
Escrevemos nosso próprio codificador aritmético com inteiros de 32 bits, no esquema de Witten, Neal e Cleary. As
probabilidades do modelo são convertidas em frequências inteiras que somam 2^16, com f = 1 + piso(p·(2^16 − 49)). O
que sobra da soma vai para o símbolo mais provável.

O texto é dividido em 512 fluxos contíguos, que avançam juntos, um caractere por passo. Cada fluxo usa como contexto
uma janela deslizante com os últimos 256 caracteres. O codificador e o decodificador rodam exatamente o mesmo laço,
em float32 e com algoritmos determinísticos. Por isso, os dois veem frequências idênticas bit a bit.

Para testar, codificamos o primeiro 1 MB do teste, decodificamos o arquivo e comparamos com o original. O tamanho
"ideal" é a soma Σ −log2 q_quantizada. O tamanho "real" é o tamanho do arquivo, incluindo o cabeçalho de 2056 bytes
(8 bytes mais 4 bytes por fluxo).

Esta forma de dar contexto (janela deslizante, até 255 caracteres antes) é diferente da usada na seção 4 (trechos
independentes). Por isso, os números desta seção não são exatamente iguais aos de lá.""",

"matriz": """\
Esta seção reorganiza os números da seção 4 como uma matriz. Cada texto de teste (do mesmo tamanho nas duas línguas)
é codificado pelo modelo da própria língua e pelo modelo da outra. Por exemplo, "en pelo pt" é o texto de teste em
inglês codificado pelo modelo treinado em português.

O excesso de entropia cruzada (CE) CE(en | modelo pt) − CE(en | modelo en) estima a diferença entre duas divergências
KL: KL(p_en‖q_pt) − KL(p_en‖q_en). Aqui p_en é a distribuição verdadeira do inglês e q é a distribuição de cada
modelo. Mostramos as duas direções porque elas não precisam ser iguais.""",

"lid": """\
Para um trecho escrito na língua A, calculamos para cada caractere a razão de log-verossimilhança
L_i = log2 q_A(x_i) − log2 q_B(x_i), onde B é a outra língua. Classificamos o trecho como A quando Σ L_i > 0.

Na primeira metade dos trechos de teste, medimos a média D e o desvio padrão de L_i. Com eles, prevemos quantos
caracteres n são necessários para errar com probabilidade eps. Pelo lema de Stein, n ≈ log2(1/eps)/D. Pelo teorema
central do limite (TCL), tratando os L_i como independentes, n ≈ (z_eps · desvio / D)², onde z_eps é o quantil da
normal padrão para eps.

Na segunda metade, que é texto diferente, medimos o erro real da regra para cada n de 1 a 256. Como amostra, usamos
os primeiros n caracteres de cada trecho de 256. O "n medido" é o menor n com erro ≤ eps; "—" quer dizer que esse
erro não foi alcançado em 256 caracteres.

Inspecionamos os trechos de teste com mais erros (modelos de 0,1M). No texto em português, eles são listas de anos
sem palavras ("1834 1835 1836 ..."). No texto em inglês, são listas de títulos de livros em português dentro de
artigos da Wikipédia em inglês.""",

"mdl": """\
Contamos os bits por caractere de cada modelo de quatro maneiras, com cada peso custando 32 bits:

- **Modelo compartilhado**: a entropia cruzada no teste. É o custo quando o receptor já tem o modelo, como um
  dicionário.
- **Duas partes, teste**: (32 × parâmetros + 10M × bits por caractere no teste) / 10M. É o custo de mandar os pesos
  junto com o arquivo de teste de 10 MB.
- **Duas partes, treino**: a mesma conta para os 400M de caracteres de treino. Como perda no treino, usamos a perda
  de validação final, já que o modelo viu cada caractere do treino uma única vez.
- **Prequencial**: o comprimento de código da seção 3, que não cobra nada pelos pesos.

Como referência, citamos sem executar o recorde do Hutter Prize: 110.793.128 bytes para o enwik9 (fx2-cmix,
setembro de 2024), ou 0,886 bits por byte de XML bruto, já contando o descompressor. Esse número não é comparável
com os nossos, porque o texto, o alfabeto e o tamanho são outros.""",

"text8": """\
O text8 é formado pelos primeiros 10^8 caracteres do enwik9 (um trecho da Wikipédia em inglês), limpos para conter
só a–z e espaço. A divisão padrão usa 90M de caracteres para treino, 5M para validação e 5M para teste.

Treinamos o modelo de 1M de parâmetros por uma época nos 90M de treino, com a mesma receita dos outros modelos. O
número "teste" abaixo é a avaliação final nos 5M de teste, guardados como val.txt.

Os resultados publicados no text8 em nível de caractere ficam em torno de 1,1 a 1,5 bits por caractere. Eles vêm de
modelos maiores, treinados por muitas épocas.""",
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

    section("2. Referências sem rede neural",
            TEXT["referencias"])
    table(["k", "pt, N = 10M", "pt, N = 100M", "en, N = 10M", "en, N = 100M"],
          [[k] + [num(ngram[f"{l}_{n}"][k], 3) for l in ("pt", "en") for n in ("10M", "100M")] for k in range(6)])
    table(["compressor", "pt bytes", "pt bits/caractere", "en bytes", "en bits/caractere"],
          [[f"`{c}`"] + [x for l in ("pt", "en") for r in comp if r["lang"] == l and r["compressor"] == c
                         for x in (thousands(int(r['bytes'])), num(float(r["bits_per_char"]), 3))]
           for c in dict.fromkeys(r["compressor"] for r in comp)])
    print("![escada de entropia](figures/entropy_ladder.png)\n")

    section("3. Modelos (`model.py`, `train.py`, `runs/*.log.json`)",
            TEXT["modelos"])
    table(["modelo", "parâmetros", "prequencial, bits/caractere (400M de treino)", "validação final, bits/caractere",
           "minutos", "VRAM máx. (GB)"],
          [[m, thousands(logs[m]['params']), num(logs[m]["prequential_bpc"]), num(logs[m]["val_bpc"]),
            f"{logs[m]['seconds'] / 60:.0f}", num(logs[m]["max_vram_gb"], 2)] for m in models])
    print("![curvas de treino](figures/training_curves.png)\n")

    section("4. Entropia cruzada no teste, calibração e perda por posição (`evaluate.py`, `runs/eval/summary.json`)",
            TEXT["teste"])
    table(["modelo", "texto de teste", "bits/caractere", "entropia média prevista", "entropia − bits/caractere"],
          [[m, t, num(summary[f"{m}_on_{t}"]["bpc"]), num(summary[f"{m}_on_{t}"]["mean_entropy"]),
            num(summary[f"{m}_on_{t}"]["mean_entropy"] - summary[f"{m}_on_{t}"]["bpc"])]
           for m in models for t in ("pt", "en") if f"{m}_on_{t}" in summary])
    print("![perda por posição](figures/loss_by_position.png)\n")

    section("5. Codificação aritmética (`coder.py`, `runs/coder/*.json`)",
            TEXT["codificacao"])
    table(["modelo", "ida e volta", "bytes do arquivo", "real, bits/caractere", "ideal, bits/caractere",
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
    table(["tamanho", "texto", "D (bits/caractere)", "desvio", "eps", "n previsto, Stein", "n previsto, TCL",
           "n medido", "erro medido com n = 256"],
          [[key.split("_")[0], key.split("_")[1], num(r["D_bits"], 3), num(r["std_bits"], 3), str(eps).replace(".", ","),
            num(p["n_stein"], 1), num(p["n_clt"], 1), p["n_measured"] or "—", num(r["error_by_n"][-1])]
           for key, r in lid.items() for eps, p in r["eps"].items()])
    print("![identificação de língua](figures/language_id.png)\n")

    section("8. Comprimento de descrição (`mdl.py`, `results/mdl.json`)",
            TEXT["mdl"])
    table(["modelo", "pesos (Mbit)", "modelo compartilhado", "duas partes, 10M de teste", "duas partes, 400M de treino",
           "prequencial, 400M de treino"],
          [[m, num(mdl[m]["weights_bits"] / 1e6, 1), num(mdl[m]["shared"]), num(mdl[m]["two_part_test"]),
            num(mdl[m]["two_part_train"]), num(mdl[m]["prequential"])] for m in models if m in mdl])

    if text8:
        section("9. Verificação de sanidade: text8 (`text8_data.sh`, `results/runs_text8/text8_1M.log.json`)",
                TEXT["text8"])
        table(["modelo", "parâmetros", "prequencial, bits/caractere (90M de treino)", "teste, bits/caractere (5M)",
               "minutos"],
              [["text8_1M", thousands(text8['params']), num(text8["prequential_bpc"]), num(text8["val_bpc"]),
                f"{text8['seconds'] / 60:.0f}"]])


if __name__ == "__main__":
    main()

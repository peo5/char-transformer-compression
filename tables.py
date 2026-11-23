"""Print the generated part of RESULTS.md (pt-BR markdown) from the result files: one short paragraph per section
saying what was measured and how, then its tables and figures. No number is typed by hand.

Usage: python tables.py RESULTS_DIR > tables.md        (make_results.sh does this)
"""
import csv, json, sys
from pathlib import Path

SIZES = ("0.1M", "1M", "10M")


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
            "Wikipédia da Hugging Face (`wikimedia/wikipedia`), dumps `20231101.pt` (os 6 arquivos parquet) e "
            "`20231101.en` (arquivos 0, 8, 16, 24, 32 e 40 de 41). Normalização (com perda, de propósito): NFKD, "
            "acentos e outras marcas combinantes removidos (ç → c), tudo em minúsculas, aspas curvas e « » viram "
            "`\"` ou `'`, travessões viram `-`, e todo caractere fora de "
            "a–z, 0–9, espaço, quebra de linha e `. , ; : ! ? ' \" - ( )` é descartado "
            f"({len(stats['recipe']['alphabet'])} símbolos; "
            "log2(49) = 5,61 bits/caractere é o teto \"sem modelo\"). Um parágrafo por linha, artigos separados por "
            f"uma linha vazia. Filtros: artigos com menos de {stats['recipe']['min_chars']} caracteres normalizados "
            "são descartados (esboços e a maioria dos artigos feitos por robôs); duplicatas exatas e quase-duplicatas "
            "(iguais depois de trocar todo dígito por 0) também. Os artigos são ordenados por md5(\"língua:id\") e "
            "distribuídos para teste, depois validação, depois treino: os conjuntos são disjuntos por artigo, e cada "
            "um tem exatamente o mesmo tamanho nas duas línguas (o último artigo é cortado).")
    rows = [["artigos lidos"] + [thousands(stats[l]['articles']) for l in ("pt", "en")],
            ["descartados, < 2000 caracteres"] + [thousands(stats[l]['short']) for l in ("pt", "en")],
            ["descartados, duplicata"] + [thousands(stats[l]['duplicate']) for l in ("pt", "en")],
            ["mantidos (caracteres)"] + [f"{thousands(stats[l]['kept'])} ({thousands(stats[l]['kept_chars'])})" for l in ("pt", "en")]]
    for split, name in (("test", "teste"), ("val", "validação"), ("train", "treino")):
        rows.append([f"{name}: caracteres (artigos)"] +
                    [f"{thousands(stats[l][split]['chars'])} ({thousands(stats[l][split]['articles'])})" for l in ("pt", "en")])
    table(["", "pt", "en"], rows)

    section("2. Referências sem rede neural",
            "**Escada de entropia** (`ngram_entropy.py`, `results/ngram.json`): entropia condicional \"plug-in\" "
            "H(X_n | k caracteres anteriores) = H((k+1)-gramas) − H(k-gramas), contando n-gramas nos primeiros N "
            "caracteres do treino, para N = 10M e N = 100M. A estimativa plug-in é viesada para baixo quando há muitos "
            "contextos distintos em relação a N; comparar os dois N mostra a partir de qual k isso acontece. "
            "**Compressores** (`compressors.sh`, `results/compressors.tsv`): gzip, xz e zstd nos arquivos de teste "
            "normalizados (10 MB por língua); bits/caractere = 8 × bytes comprimidos / caracteres.")
    table(["k", "pt, N = 10M", "pt, N = 100M", "en, N = 10M", "en, N = 100M"],
          [[k] + [num(ngram[f"{l}_{n}"][k], 3) for l in ("pt", "en") for n in ("10M", "100M")] for k in range(6)])
    table(["compressor", "pt bytes", "pt bits/caractere", "en bytes", "en bits/caractere"],
          [[f"`{c}`"] + [x for l in ("pt", "en") for r in comp if r["lang"] == l and r["compressor"] == c
                         for x in (thousands(int(r['bytes'])), num(float(r["bits_per_char"]), 3))]
           for c in dict.fromkeys(r["compressor"] for r in comp)])
    print("![escada de entropia](figures/entropy_ladder.png)\n")

    section("3. Modelos (`model.py`, `train.py`, `runs/*.log.json`)",
            "Transformer só-decodificador mínimo, escrito à mão em PyTorch: tokens são caracteres, contexto de 256, "
            "três tamanhos por língua (cerca de 0,1M, 1M e 10M parâmetros). Cada modelo é treinado por exatamente uma "
            "época: cada trecho de 256 caracteres do treino (400M caracteres) é visto uma vez, em ordem aleatória "
            "(semente 0), lotes de 64 trechos, AdamW, taxa de aprendizado com aquecimento e decaimento cosseno, bf16. "
            "**Comprimento de código prequencial:** cada lote é avaliado ANTES da atualização feita com ele; a soma "
            "dessas perdas é o comprimento de código dos dados de treino sem cobrar nada pelos pesos (um decodificador "
            "que conhece a semente e a receita refaz as mesmas atualizações). Validação: os 5M caracteres de val.txt. "
            "As curvas de validação na figura usam só os primeiros 0,5M caracteres de val.txt.")
    table(["modelo", "parâmetros", "prequencial, bits/caractere (400M de treino)", "validação final, bits/caractere",
           "minutos", "VRAM máx. (GB)"],
          [[m, thousands(logs[m]['params']), num(logs[m]["prequential_bpc"]), num(logs[m]["val_bpc"]),
            f"{logs[m]['seconds'] / 60:.0f}", num(logs[m]["max_vram_gb"], 2)] for m in models])
    print("![curvas de treino](figures/training_curves.png)\n")

    section("4. Entropia cruzada no teste, calibração e perda por posição (`evaluate.py`, `runs/eval/summary.json`)",
            "Cada arquivo de teste (10M caracteres) é cortado em trechos de 256 caracteres sem sobreposição, e cada "
            "trecho é avaliado sem nenhum contexto anterior a ele: a posição i prevê o caractere i+1 a partir dos "
            "caracteres 0..i (float32, sem AMP). **Calibração:** a entropia média da distribuição prevista pelo "
            "modelo comparada com a entropia cruzada real; se as duas coincidem, o modelo sabe o quanto não sabe. "
            "**Perda por posição:** entropia cruzada média em cada posição do trecho, ou seja, em função de quantos "
            "caracteres de contexto o modelo tinha.")
    table(["modelo", "texto de teste", "bits/caractere", "entropia média prevista", "entropia − bits/caractere"],
          [[m, t, num(summary[f"{m}_on_{t}"]["bpc"]), num(summary[f"{m}_on_{t}"]["mean_entropy"]),
            num(summary[f"{m}_on_{t}"]["mean_entropy"] - summary[f"{m}_on_{t}"]["bpc"])]
           for m in models for t in ("pt", "en") if f"{m}_on_{t}" in summary])
    print("![perda por posição](figures/loss_by_position.png)\n")

    section("5. Codificação aritmética (`coder.py`, `runs/coder/*.json`)",
            "Codificador aritmético inteiro próprio (registradores de 32 bits, Witten–Neal–Cleary). As probabilidades "
            "do modelo são quantizadas em frequências inteiras somando 2^16: f = 1 + piso(p·(2^16 − 49)), o resto vai "
            "para o símbolo mais provável. O texto é dividido em 512 fluxos contíguos que avançam juntos, um "
            "caractere por passo, cada um com uma janela deslizante dos últimos 256 caracteres. Codificador e "
            "decodificador rodam exatamente o mesmo laço (float32, algoritmos determinísticos), então veem "
            "frequências idênticas bit a bit. Teste: codificar e decodificar o primeiro 1 MB do teste e comparar. "
            "\"Ideal\" = Σ −log2 q_quantizada; \"real\" = tamanho do arquivo, incluindo o cabeçalho (8 bytes + 4 bytes "
            "por fluxo = 2056 bytes). Esta política de contexto (janela deslizante, até 255 caracteres de contexto) é "
            "diferente da política por trechos da seção 4, então os números não são idênticos aos de lá.")
    table(["modelo", "ida e volta", "bytes do arquivo", "real, bits/caractere", "ideal, bits/caractere",
           "real − ideal", "codificação (s)", "decodificação (s)"],
          [[m, "OK" if coder[m]["round_trip_ok"] else "FALHOU", thousands(coder[m]['file_bytes']),
            num(coder[m]["real_bpc"]), num(coder[m]["ideal_bpc"]), num(coder[m]["real_bpc"] - coder[m]["ideal_bpc"]),
            f"{coder[m]['encode_seconds']:.0f}", f"{coder[m]['decode_seconds']:.0f}"] for m in models if m in coder])

    section("6. Matriz entre línguas (`evaluate.py`, `runs/eval/summary.json`)",
            "Os mesmos números da seção 4, organizados como matriz: cada texto de teste (mesmo tamanho nas duas "
            "línguas) codificado pelo modelo da própria língua e pelo da outra. \"en pelo pt\" = texto de teste em "
            "inglês codificado pelo modelo português. O excesso CE(en | modelo pt) − CE(en | modelo en) estima "
            "KL(p_en‖q_pt) − KL(p_en‖q_en); as duas direções são mostradas porque não precisam ser iguais.")
    rows = []
    for size in SIZES:
        k = lambda model, text: summary.get(f"{model}_{size}_on_{text}", {}).get("bpc")
        pp, ee, pe, ep = k("pt", "pt"), k("en", "en"), k("pt", "en"), k("en", "pt")
        if None not in (pp, ee, pe, ep):
            rows.append([size, num(pp), num(ee), num(pe), num(ep), num(pe - ee), num(ep - pp)])
    table(["tamanho", "pt pelo pt", "en pelo en", "en pelo pt", "pt pelo en",
           "excesso no texto en (modelo pt − modelo en)", "excesso no texto pt (modelo en − modelo pt)"], rows)

    section("7. Identificação de língua: uma previsão testada (`lid.py`, `runs/lid.json`)",
            "Para um trecho na língua A, a razão de log-verossimilhança por caractere é "
            "L_i = log2 q_A(x_i) − log2 q_B(x_i) (B = a outra língua); o trecho é classificado como A quando "
            "Σ L_i > 0. Na primeira metade dos trechos de teste medimos a média D e o desvio padrão de L_i e "
            "prevemos quantos caracteres n são necessários para errar com probabilidade eps: pelo lema de Stein, "
            "n ≈ log2(1/eps)/D; pelo teorema central do limite (tratando os L_i como independentes), "
            "n ≈ (z_eps · desvio / D)². A segunda metade (texto disjunto) dá o erro medido da regra para cada "
            "n = 1..256, usando como amostra os primeiros n caracteres de cada trecho de 256. \"n medido\" = menor n "
            "com erro ≤ eps; — = não alcançado em 256 caracteres. Inspeção dos trechos de teste com maior erro "
            "(modelos de 0,1M): no texto pt, listas de anos sem palavras (\"1834 1835 1836 ...\"); no texto en, "
            "listas de títulos de livros em português dentro de artigos da Wikipédia em inglês.")
    table(["tamanho", "texto", "D (bits/caractere)", "desvio", "eps", "n previsto, Stein", "n previsto, TCL",
           "n medido", "erro medido com n = 256"],
          [[key.split("_")[0], key.split("_")[1], num(r["D_bits"], 3), num(r["std_bits"], 3), str(eps).replace(".", ","),
            num(p["n_stein"], 1), num(p["n_clt"], 1), p["n_measured"] or "—", num(r["error_by_n"][-1])]
           for key, r in lid.items() for eps, p in r["eps"].items()])
    print("![identificação de língua](figures/language_id.png)\n")

    section("8. Comprimento de descrição (`mdl.py`, `results/mdl.json`)",
            "Quatro maneiras de contar bits por caractere para o mesmo modelo, com pesos a 32 bits cada: "
            "**modelo compartilhado** = entropia cruzada no teste (o receptor já tem o modelo, como um dicionário); "
            "**duas partes, teste** = (32 × parâmetros + 10M × bits/caractere no teste) / 10M, isto é, mandar os "
            "pesos junto com o arquivo de teste de 10 MB; **duas partes, treino** = o mesmo para os 400M caracteres "
            "de treino, usando a perda de validação final como perda no treino (uma época, cada caractere visto uma "
            "vez); **prequencial** = seção 3, sem custo pelos pesos. Referência citada, não executada: o recorde do "
            "Hutter Prize é 110.793.128 bytes para o enwik9 (fx2-cmix, set/2024) = 0,886 bits por byte de XML bruto, "
            "descompressor incluído; não é comparável com estes números (outro texto, outro alfabeto, outro tamanho).")
    table(["modelo", "pesos (Mbit)", "modelo compartilhado", "duas partes, 10M de teste", "duas partes, 400M de treino",
           "prequencial, 400M de treino"],
          [[m, num(mdl[m]["weights_bits"] / 1e6, 1), num(mdl[m]["shared"]), num(mdl[m]["two_part_test"]),
            num(mdl[m]["two_part_train"]), num(mdl[m]["prequential"])] for m in models if m in mdl])

    if text8:
        section("9. Verificação de sanidade: text8 (`text8_data.sh`, `results/runs_text8/text8_1M.log.json`)",
                "text8 = primeiros 10^8 caracteres do enwik9 limpos para a–z e espaço; divisão padrão 90M treino / "
                "5M validação / 5M teste. O modelo de 1M foi treinado uma época nos 90M de treino com a mesma receita; "
                "o número \"teste\" abaixo é a avaliação final nos 5M de teste (guardados como val.txt). Resultados "
                "publicados no text8 em nível de caractere ficam em torno de 1,1–1,5 bits/caractere, com modelos "
                "maiores treinados por muitas épocas.")
        table(["modelo", "parâmetros", "prequencial, bits/caractere (90M de treino)", "teste, bits/caractere (5M)",
               "minutos"],
              [["text8_1M", thousands(text8['params']), num(text8["prequential_bpc"]), num(text8["val_bpc"]),
                f"{text8['seconds'] / 60:.0f}"]])


if __name__ == "__main__":
    main()

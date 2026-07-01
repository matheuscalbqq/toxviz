"""
ToxViz — limpeza e consolidação das 4 fontes de toxicidade peptídica.

Fontes:
  - NTxPred2 (independent + cross_val)  -> neurotoxin
  - hemolytik.fasta                     -> hemotoxin
  - epitope_table_export (IEDB)         -> immunotoxin
  - ToxProt.fasta                       -> cytotoxin

Saída: sequence, neurotoxin, hemotoxin, immunotoxin, cytotoxin (0/1)
multitoxin é deliberadamente OMITIDO aqui — é derivado em tempo de
visualização (soma das 4 flags > 1), conforme decidido na consolidação
do schema.
"""
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent      # .../toxviz/src
PROJECT_ROOT = SCRIPT_DIR.parent                    # .../toxviz
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "raw_data"

MIN_LEN, MAX_LEN = 5, 50
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def is_clean_sequence(seq: str) -> bool:
    """Sequência só com aminoácidos padrão e dentro do range de tamanho."""
    if not (MIN_LEN <= len(seq) <= MAX_LEN):
        return False
    return set(seq.upper()) <= STANDARD_AA


def parse_fasta(path: str) -> list[tuple[str, str]]:
    recs, header, seq = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    recs.append((header, "".join(seq)))
                header, seq = line[1:], []
            else:
                seq.append(line)
        if header is not None:
            recs.append((header, "".join(seq)))
    return recs


# ---------------------------------------------------------------- neurotoxin
def load_neurotoxin() -> set[str]:
    ind = pd.read_csv(DB_DIR / "NTxPred2_independent_dataset.csv")
    cv = pd.read_csv(DB_DIR / "NTxPred2_cross_val_dataset.csv")
    combined = pd.concat([ind, cv], ignore_index=True)
    positives = combined.loc[combined.Label == 1, "Sequence"].astype(str)
    clean = {s for s in positives if is_clean_sequence(s)}
    print(f"[neurotoxin] brutos positivos={len(positives)} | limpos={len(clean)}")
    return clean


# ------------------------------------------------------------------ hemotoxin
def load_hemotoxin() -> set[str]:
    recs = parse_fasta(DB_DIR / "hemolytik.fasta")
    seqs = [s for _, s in recs]
    clean = {s for s in seqs if is_clean_sequence(s)}
    print(f"[hemotoxin] brutos={len(seqs)} (únicos brutos={len(set(seqs))}) | limpos={len(clean)}")
    return clean


# --------------------------------------------------------------- immunotoxin
def load_immunotoxin() -> set[str]:
    df = pd.read_csv(DB_DIR / "epitope_table_export_1782853329.csv", header=1)
    seqs = df["Name"].astype(str)
    # descarta entradas com anotação de modificação colada na sequência
    # (ex.: "AVWRIDTPDKLT + ACET(A1)") — não são string de aminoácido pura
    no_mod = seqs[~seqs.str.contains(r"\+", regex=True)]
    clean = {s for s in no_mod if is_clean_sequence(s)}
    print(f"[immunotoxin] brutos={len(seqs)} | sem modificação={len(no_mod)} | limpos={len(clean)}")
    return clean


# ----------------------------------------------------------------- cytotoxin
def load_cytotoxin() -> set[str]:
    recs = parse_fasta(DB_DIR / "ToxProt.fasta")
    # descarta fragmentos (pedaço de enzima grande, não peptídeo curto real)
    non_fragment = [(h, s) for h, s in recs if "(Fragment)" not in h]
    seqs = [s for _, s in non_fragment]
    clean = {s for s in seqs if is_clean_sequence(s)}
    print(f"[cytotoxin] brutos={len(recs)} | não-fragmento={len(non_fragment)} | limpos={len(clean)}")
    return clean


def main():
    neuro = load_neurotoxin()
    hemo = load_hemotoxin()
    immuno = load_immunotoxin()
    cyto = load_cytotoxin()

    all_seqs = neuro | hemo | immuno | cyto
    print(f"\nTotal de sequências únicas (união das 4 fontes): {len(all_seqs)}")

    rows = []
    for s in sorted(all_seqs):
        rows.append({
            "sequence": s,
            "neurotoxin": int(s in neuro),
            "hemotoxin": int(s in hemo),
            "immunotoxin": int(s in immuno),
            "cytotoxin": int(s in cyto),
        })
    df = pd.DataFrame(rows)

    # quantas sequências aparecem em mais de uma fonte (preview do multitoxin)
    flag_sum = df[["neurotoxin", "hemotoxin", "immunotoxin", "cytotoxin"]].sum(axis=1)
    print(f"Sequências em >1 categoria (preview multitoxin): {(flag_sum > 1).sum()}")

    df.to_csv(DATA_DIR / "toxviz_consolidated.csv", index=False)
    print("\nSalvo em data/toxviz_consolidated.csv")
    print(df["sequence"].apply(len).describe())


if __name__ == "__main__":
    main()

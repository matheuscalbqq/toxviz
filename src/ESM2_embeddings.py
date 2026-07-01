"""
ToxViz — extração de embeddings ESM-2 35M para a base consolidada,
usando exatamente a mesma biblioteca e checkpoint do pipeline NoiTox:

    biblioteca:  fair-esm  (pip install fair-esm==1.0.3)
    checkpoint:  esm2_t12_35M_UR50D  (12 camadas, 480d)
    *** ATENÇÃO: o pacote correto é "fair-esm", não "esm" (esm é a API
        do ESM3, incompatível) — mesmo problema que vocês já resolveram
        no pipeline do NoiTox.

Agregação: mean-pooling sobre os tokens de resíduo, excluindo
<cls>/<bos> e <eos> — mesma lógica usada para o modelo MLP do NoiTox
(que também usa ESM-2 diretamente, sem grafo).

Saída: toxviz_with_embeddings.parquet
"""
import numpy as np
import pandas as pd
import torch

MODEL_NAME = "esm2_t12_35M_UR50D"   # mesma checkpoint do NoiTox
REPR_LAYER = 12                      # última camada do t12 (480d)
BATCH_SIZE = 16                      # reduza se tiver OOM, aumente se tiver GPU grande


def carregar_modelo_esm(nome_modelo: str = MODEL_NAME, device: str = None):
    """Carrega o modelo ESM-2 e o tokenizador (batch_converter) via fair-esm."""
    try:
        import esm as esm_lib
    except ImportError as e:
        raise ImportError(
            "Biblioteca ESM não encontrada ou incorreta.\n"
            "Instale com: pip install fair-esm==1.0.3\n"
            "(NÃO instale apenas 'esm' — esse é o pacote do ESM3, incompatível)"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[ESM-2] Carregando modelo: {nome_modelo}")
    print(f"[ESM-2] Device: {device}")

    model, alphabet = esm_lib.pretrained.load_model_and_alphabet(nome_modelo)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    embed_dim = model.embed_dim
    print(f"[ESM-2] Dimensão do embedding (D): {embed_dim}")
    print(f"[ESM-2] Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    return model, alphabet, batch_converter, device


@torch.no_grad()
def embed_batch(seqs: list[str], model, batch_converter, device: str) -> np.ndarray:
    """
    Mean-pooling sobre os resíduos para um batch de sequências.
    fair-esm formata cada sequência como [<cls>, resíduo_1, ..., resíduo_N, <eos>],
    então descartamos a posição 0 (<cls>) e a última posição válida (<eos>)
    de cada sequência antes de fazer a média.
    """
    data = [(str(i), seq) for i, seq in enumerate(seqs)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    reps = out["representations"][REPR_LAYER]  # (batch, seq_len, 480)

    pooled = []
    for i, seq in enumerate(seqs):
        # tokens[i] = [<cls>, r1, ..., rN, <eos>, <pad>, ...]
        # resíduos válidos ficam em [1, len(seq)+1)
        residue_reps = reps[i, 1 : len(seq) + 1]
        pooled.append(residue_reps.mean(dim=0))

    return torch.stack(pooled).cpu().numpy()


def main():
    df = pd.read_csv("toxviz_consolidated.csv")
    sequences = df["sequence"].tolist()
    print(f"Total de sequências a embedar: {len(sequences)}")

    model, alphabet, batch_converter, device = carregar_modelo_esm()

    embeddings = []
    for i in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[i : i + BATCH_SIZE]
        emb = embed_batch(batch, model, batch_converter, device)
        embeddings.append(emb)
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  {i + len(batch)}/{len(sequences)}")

    embeddings = np.vstack(embeddings)
    print("Shape final dos embeddings:", embeddings.shape)  # (N, 480)

    df["embedding"] = list(embeddings)
    df.to_parquet("toxviz_with_embeddings.parquet", index=False)
    print("Salvo em toxviz_with_embeddings.parquet")


if __name__ == "__main__":
    main()

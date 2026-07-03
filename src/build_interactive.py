import math
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, beta as beta_dist
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parent 
PROJECT_ROOT = SCRIPT_DIR.parent                  
DATA_DIR = PROJECT_ROOT / "data"

alt.data_transformers.disable_max_rows()

FLAGS = ["neurotoxin", "hemotoxin", "immunotoxin", "cytotoxin"]

WONG = {
    "immunotoxin": "#0072B2",
    "hemotoxin": "#E69F00",
    "neurotoxin": "#009E73",
    "cytotoxin": "#CC79A7",
}

COMBO_COLORS = {
    "immunotoxin": WONG["immunotoxin"],
    "hemotoxin": WONG["hemotoxin"],
    "neurotoxin": WONG["neurotoxin"],
    "cytotoxin": WONG["cytotoxin"],
    "hemotoxin+cytotoxin": "#D55E00",
    "neurotoxin+cytotoxin": "#56B4E9",
    "neurotoxin+hemotoxin": "#F5C710",
    "neurotoxin+hemotoxin+cytotoxin": "#474747FF",
}

MAXBINS = 30
PERCENTILE_TRIM = 2     # corta os 2% mais extremos de cada lado, por eixo
EDGE_PADDING_FRAC = 0.03
K_SAFETY_MARGIN = 1.5      # margem sobre o k mínimo estatístico por categoria
ALPHA = 0.001


def compute_k_per_target(df: pd.DataFrame, alpha: float, margin: float):
    """
    k específico por categoria-alvo, derivado do mínimo estatístico necessário
    para a categoria conseguir, em tese, atingir significância — em vez de um
    k fixo igual pra todo mundo. Sem isso, categorias de alta taxa-base (ex:
    immunotoxin, 76% da base) ficam matematicamente impossibilitadas de
    aparecer como candidato com k pequeno (efeito-teto: mesmo 100% dos
    vizinhos batendo não basta para superar uma base já alta).
    """
    n = len(df)
    base_rate = {f: df[f].sum() / n for f in FLAGS}
    k_per_target = {}
    for f in FLAGS:
        p0 = base_rate[f]
        k = 1
        while True:
            res = binomtest(k, k, p0, alternative="greater")
            if res.pvalue < alpha:
                break
            k += 1
        k_per_target[f] = math.ceil(k * margin)
    return k_per_target, base_rate


def build_combo_column(df: pd.DataFrame) -> pd.DataFrame:
    def combo_str(row):
        active = [f for f in FLAGS if row[f] == 1]
        return "+".join(active)

    df = df.copy()
    df["combo"] = df.apply(combo_str, axis=1)
    df["is_multitoxin"] = df["combo"].str.contains(r"\+")
    return df


def build_matrix_df(combo_order: list[str]) -> pd.DataFrame:
    rows = []
    for combo in combo_order:
        active_flags = set(combo.split("+"))
        for flag in FLAGS:
            rows.append({"combo": combo, "flag": flag, "active": flag in active_flags})
    return pd.DataFrame(rows)


PRIOR_STRENGTH = 4  # "tamanho amostral efetivo" do prior — quanto maior, mais o prior pesa contra a evidência local


def posterior_prob_enriched(k_obs: int, k_used: int, p0: float, prior_strength: float) -> float:
    """
    P(taxa_local > taxa_base | vizinhos observados), via prior Beta centrado
    na taxa-base populacional + atualização conjugada pela evidência local.

    prior: Beta(a0, b0) com média = p0 e "peso" = prior_strength
    posterior: Beta(a0 + k_obs, b0 + (k_used - k_obs))
    probabilidade reportada: 1 - CDF_Beta(p0; posterior) = P(p_local > p0 | dados)
    """
    a0 = prior_strength * p0
    b0 = prior_strength * (1 - p0)
    a1 = a0 + k_obs
    b1 = b0 + (k_used - k_obs)
    return float(beta_dist.sf(p0, a1, b1))


def compute_candidates(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    k_per_target, base_rate = compute_k_per_target(df, ALPHA, K_SAFETY_MARGIN)
    print("k por categoria-alvo (mínimo estatístico x margem de segurança):")
    for f in FLAGS:
        print(f"  {f:12s} k={k_per_target[f]}")

    pool_size = max(k_per_target.values())  # maior k necessário entre as 4 categorias
    nn = NearestNeighbors(n_neighbors=pool_size + 1, metric="cosine")
    nn.fit(embeddings)
    _, idx = nn.kneighbors(embeddings)
    idx_no_self = idx[:, 1:]  # já ordenado por distância crescente

    combos = df["combo"].to_numpy()
    single_mask = ~df["is_multitoxin"].to_numpy()

    rows = []
    for i in np.where(single_mask)[0]:
        own = combos[i]
        own_set = set(own.split("+"))
        for f in FLAGS:
            if f in own_set:
                continue
            k_f = k_per_target[f]
            # só os k_f vizinhos mais próximos (o pool já está ordenado por
            # distância) — cada categoria-alvo usa sua própria janela
            nbrs_f = combos[idx_no_self[i, :k_f]]
            k_obs = sum(1 for nb in nbrs_f if f in nb.split("+"))
            if k_obs == 0:
                continue
            res = binomtest(k_obs, k_f, base_rate[f], alternative="greater")
            if res.pvalue < ALPHA:
                prob = posterior_prob_enriched(k_obs, k_f, base_rate[f], PRIOR_STRENGTH)
                rows.append({
                    "sequence": df["sequence"].iloc[i],
                    "source_combo": own,
                    "target_flag": f,
                    "k_used": k_f,
                    "k_obs": k_obs,
                    "pvalue": res.pvalue,
                    "probability": prob,
                })
    return pd.DataFrame(rows)


def compute_domain(df: pd.DataFrame, must_include: pd.DataFrame, axis: str) -> list[float]:
    """Percentil nos dados todos, mas nunca corta o que está em must_include."""
    lo_p, hi_p = np.percentile(df[axis], [PERCENTILE_TRIM, 100 - PERCENTILE_TRIM])
    lo = min(lo_p, must_include[axis].min())
    hi = max(hi_p, must_include[axis].max())
    span = hi - lo
    pad = span * EDGE_PADDING_FRAC
    return [lo - pad, hi + pad]


def build_density_grid(sub: pd.DataFrame, x_edges: np.ndarray, y_edges: np.ndarray) -> pd.DataFrame:
    """Histograma 2D + normalização pelo PICO da própria categoria (max-norm)."""
    H, _, _ = np.histogram2d(sub["umap_x"], sub["umap_y"], bins=[x_edges, y_edges])
    peak = H.max()
    if peak == 0:
        peak = 1  # evita divisão por zero (categoria sem nenhum ponto na grade)
    rows = []
    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            count = H[ix, iy]
            if count == 0:
                continue
            rows.append({
                "x0": x_edges[ix], "x1": x_edges[ix + 1],
                "y0": y_edges[iy], "y1": y_edges[iy + 1],
                "pct_of_peak": count / peak,
            })
    return pd.DataFrame(rows)


UMAP_CACHE_PATH = DATA_DIR / "toxviz_umap.csv"
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)


def load_or_compute_umap(emb_df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Calcula o UMAP em tempo de execução SE não houver cache em disco; se
    'toxviz_umap.csv' já existir, carrega direto (evita recalcular ~14k
    pontos toda vez que só um parâmetro visual muda, ex: cor, bins).

    Para forçar recálculo (ex: depois de mudar UMAP_PARAMS), apague o
    toxviz_umap.csv ou chame com force=True.
    """
    if os.path.exists(UMAP_CACHE_PATH):
        print(f"[UMAP] cache encontrado em {UMAP_CACHE_PATH} — carregando (não recalculado)")
        cached = pd.read_csv(UMAP_CACHE_PATH)
        if len(cached) == len(emb_df) and (cached["sequence"].values == emb_df["sequence"].values).all():
            return cached
        print("[UMAP] cache desalinhado com o parquet atual — recalculando...")

    try:
        import umap
    except ImportError as e:
        raise ImportError("Biblioteca umap-learn não encontrada.\nInstale com: pip install umap-learn") from e

    print(f"[UMAP] calculando projeção 480d -> 2D para {len(embeddings)} sequências (pode levar alguns minutos)...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    emb2d = reducer.fit_transform(embeddings)

    df = emb_df.drop(columns=["embedding"]).copy()
    df["umap_x"] = emb2d[:, 0]
    df["umap_y"] = emb2d[:, 1]

    df.to_csv(UMAP_CACHE_PATH, index=False)
    print(f"[UMAP] concluído — cache salvo em {UMAP_CACHE_PATH} para as próximas execuções")
    return df

def canonical_combo(flags_set) -> str:
    return "+".join(f for f in FLAGS if f in flags_set)

def main():
    emb_df = pd.read_parquet(DATA_DIR/"toxviz_with_embeddings.parquet")
    embeddings = np.stack(emb_df["embedding"].values)

    df = load_or_compute_umap(emb_df, embeddings)
    df = build_combo_column(df)

    cand_df = compute_candidates(df, embeddings)
    print(f"Candidatos encontrados (p<{ALPHA}): {len(cand_df)}")

    # combo HIPOTÉTICO do candidato: source_combo + target_flag combinados
    cand_df["combo"] = cand_df.apply(
        lambda r: canonical_combo(set(r["source_combo"].split("+")) | {r["target_flag"]}), axis=1
    )

    # "porcentagem dos candidatos" (k_obs/k_used) formatada + texto pronto pra lista
    cand_df["pct_label"] = (cand_df["k_obs"] / cand_df["k_used"] * 100).round(0).astype(int).astype(str) + "%"
    cand_points = cand_df.merge(df[["sequence", "umap_x", "umap_y"]], on="sequence", how="left")

    combo_counts = df["combo"].value_counts()
    combo_order = combo_counts.index.tolist()

    # estende o domínio do eixo X com combos hipotéticos que só existem via candidato
    cand_combo_counts = cand_df["combo"].value_counts()
    new_combos = [c for c in cand_combo_counts.index if c not in combo_order]
    new_combos_sorted = cand_combo_counts.loc[new_combos].sort_values(ascending=False).index.tolist()
    combo_order = combo_order + new_combos_sorted
    domain = combo_order

    EXTRA_COLORS = ["#8B4513", "#4B0082", "#2F4F4F", "#1B7837"]

    def color_for(combo: str) -> str:
        if combo in COMBO_COLORS:
            return COMBO_COLORS[combo]
        return EXTRA_COLORS[new_combos_sorted.index(combo) % len(EXTRA_COLORS)]

    range_ = [color_for(c) for c in combo_order]

    # dados do empilhamento: y0 = total confirmado, y1 = total + candidatos
    cand_stack = cand_df.groupby("combo").size().rename("n_candidates").reset_index()
    real_totals = combo_counts.rename("real_total").reset_index()
    real_totals.columns = ["combo", "real_total"]
    stack_df = cand_stack.merge(real_totals, on="combo", how="left")
    stack_df["real_total"] = stack_df["real_total"].fillna(0)
    stack_df["y0"] = stack_df["real_total"].clip(lower=1)
    stack_df["y1"] = stack_df["y0"] + stack_df["n_candidates"]
    stack_df["fill_color"] = stack_df["combo"].map(color_for)

    matrix_df = build_matrix_df(combo_order)
    bar_summary = combo_counts.rename("total").reset_index().rename(columns={"index": "combo"})

    # linha conectando os pontos ativos de cada combo (só combos com 2+ flags)
    connector_rows = []
    for combo in combo_order:
        active_flags = [f for f in FLAGS if f in combo.split("+")]
        if len(active_flags) >= 2:
            connector_rows.append({
                "combo": combo,
                "flag_start": active_flags[0],
                "flag_end": active_flags[-1],
            })
    matrix_connectors = pd.DataFrame(connector_rows)

    # ----------------------------------------------------- domínio dos eixos
    must_include = pd.concat([
        df[df["is_multitoxin"]][["umap_x", "umap_y"]],
        cand_points[["umap_x", "umap_y"]],
    ])
    x_domain = compute_domain(df, must_include, "umap_x")
    y_domain = compute_domain(df, must_include, "umap_y")

    X_SCALE = alt.Scale(domain=x_domain, nice=False, zero=False)
    Y_SCALE = alt.Scale(domain=y_domain, nice=False, zero=False)

    x_edges = np.linspace(x_domain[0], x_domain[1], MAXBINS + 1)
    y_edges = np.linspace(y_domain[0], y_domain[1], MAXBINS + 1)

    # --------------------------------------------------- grades de densidade
    # (calculadas sobre TODOS os pontos de cada categoria, não só os que
    # caem dentro do domínio cortado — só o desenho é que é cortado)
    grids = {}
    for flag in FLAGS:
        sub = df[df["combo"] == flag]
        grids[flag] = build_density_grid(sub, x_edges, y_edges)

    # ---------------------------------------------------- seleções/parâmetros
    brush = alt.selection_interval(name="brush", encodings=["x", "y"], empty=False)
    combo_select = alt.selection_point(name="combo_select", fields=["combo"], empty=False, on="click")
    show_candidates = alt.param(
        name="show_candidates",
        value=False,
    )
    reveal_test = brush | combo_select

    # --------------------------------------------- UMAP: densidade por categoria
    density_layers = []
    for flag in FLAGS:
        layer = (
            alt.Chart(grids[flag])
            .mark_rect(clip=True)
            .encode(
                x=alt.X("x0:Q", scale=X_SCALE, title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
                x2="x1:Q",
                y=alt.Y("y0:Q", scale=Y_SCALE, title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
                y2="y1:Q",
                opacity=alt.Opacity("pct_of_peak:Q", scale=alt.Scale(domain=[0, 1], range=[0.0, 0.8]), legend=None),
                color=alt.value(WONG[flag]),
                tooltip=[alt.Tooltip("pct_of_peak:Q", title=f"{flag} (% do pico de densidade)", format=".0%")],
            )
        )
        density_layers.append(layer)

    # pontos individuais (single-category): só aparecem com brush ou clique
    reveal_single = (
        alt.Chart(df[~df["is_multitoxin"]])
        .mark_circle(size=44, stroke="white", strokeWidth=0.4, clip=True)
        .encode(
            x=alt.X("umap_x:Q", scale=X_SCALE),
            y=alt.Y("umap_y:Q", scale=Y_SCALE),
            color=alt.Color("combo:N", scale=alt.Scale(domain=domain, range=range_), legend=None),
            opacity=alt.condition(reveal_test, alt.value(0.95), alt.value(0.0)),
            tooltip=["sequence:N", "combo:N"],
        )
        .add_params(brush, combo_select)
    )

    # pontos multitóxicos: mesma lógica de revelação, losango maior
    reveal_multitoxin = (
        alt.Chart(df[df["is_multitoxin"]])
        .mark_point(shape="diamond", size=150, filled=True, stroke="black", strokeWidth=1.2, clip=True)
        .encode(
            x=alt.X("umap_x:Q", scale=X_SCALE),
            y=alt.Y("umap_y:Q", scale=Y_SCALE),
            color=alt.Color("combo:N", scale=alt.Scale(domain=domain, range=range_), legend=None),
            opacity=alt.condition(reveal_test, alt.value(0.95), alt.value(0.0)),
            tooltip=["sequence:N", "combo:N"],
        )
    )

    # candidatos: losango vazado, controlado só pelo checkbox (independente
    # do brush/clique — comportamento já validado na v4, mantido aqui)
    candidate_overlay = (
        alt.Chart(cand_points)
        .mark_point(shape="cross", filled=True, stroke="black", strokeWidth=0.8, clip=True)
        .encode(
            x=alt.X("umap_x:Q", scale=X_SCALE),
            y=alt.Y("umap_y:Q", scale=Y_SCALE),
            size=alt.value(70),
            color=alt.Color("combo:N", scale=alt.Scale(domain=FLAGS, range=[WONG[f] for f in FLAGS]), legend=None),
            opacity=alt.condition(
                        {"test": {"and": [{"param": "show_candidates"},
                            {"or": [{"param": "brush", "empty": False},
                                    {"param": "combo_select", "empty": False}]}]}},     # type: ignore[arg-type]
                        alt.value(0.95), alt.value(0.0),),                              # type: ignore[arg-type]
            tooltip=["sequence:N", "source_combo:N", "target_flag:N", "k_obs:Q",
                     alt.Tooltip("probability:Q", title="P(enriquecido)", format=".1%"),
                     alt.Tooltip("pvalue:Q", format=".2e")],
        )
        .add_params(show_candidates)
    )
    # legenda pequena embutida no canto inferior-esquerdo do UMAP —
    # em coordenadas de dados (mesma X_SCALE/Y_SCALE), não em pixel fixo
    row_step = (y_domain[1] - y_domain[0]) * 0.045
    inline_legend_df = pd.DataFrame({
        "flag": FLAGS,
        "lx": x_domain[0] + (x_domain[1] - x_domain[0]) * 0.03,
        "ly": [y_domain[0] + (y_domain[1] - y_domain[0]) * 0.03 + i * row_step for i in range(len(FLAGS))],
    })

    inline_legend_squares = (
        alt.Chart(inline_legend_df)
        .mark_square(size=90, filled=True, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("lx:Q", scale=X_SCALE),
            y=alt.Y("ly:Q", scale=Y_SCALE),
            color=alt.Color("flag:N", scale=alt.Scale(domain=FLAGS, range=[WONG[f] for f in FLAGS]), legend=None),
        )
    )
    inline_legend_text = (
        alt.Chart(inline_legend_df)
        .mark_text(align="left", dx=8, fontSize=10, color="#333")
        .encode(
            x=alt.X("lx:Q", scale=X_SCALE),
            y=alt.Y("ly:Q", scale=Y_SCALE),
            text="flag:N",
        )
    )

    umap_panel = alt.layer(
        *density_layers, reveal_single, reveal_multitoxin, candidate_overlay,
        inline_legend_squares, inline_legend_text,
    ).properties(
        width=760,
        height=560,
        title=[
            "UMAP",
        ],
    )

    # ------------------------------------------------------------ UpSet bars
    x_log = alt.X("count():Q", title="Total (log scale)", scale=alt.Scale(type="log", domain=[1, 20000]),
                  axis=alt.Axis(grid=False))
    UPSET_ROW_H = 54
    combo_abbrev_expr = " : ".join(
        f"datum.value === '{c}' ? '{'+'.join(f[0].upper() for f in c.split('+'))}'" for c in combo_order
    ) + " : datum.value"
    flag_abbrev_expr = " : ".join(
        f"datum.value === '{f}' ? '{f[0].upper()}'" for f in FLAGS
    ) + " : datum.value"

    bg_bars = (
        alt.Chart(df)
        .mark_bar(opacity=0.4)
        .encode(
            x=x_log,
            y=alt.Y("combo:N", sort=combo_order, title=None, axis=alt.Axis(labelExpr=combo_abbrev_expr)),
            color=alt.Color("combo:N", scale=alt.Scale(domain=domain, range=range_), legend=None),
        )
        .properties(width=360, height=UPSET_ROW_H*len(combo_order))
    )

    bg_labels = (
        alt.Chart(bar_summary)
        .mark_text(fontSize=11, fontWeight="bold", align="left", color="black")
        .encode(
            y=alt.Y("combo:N", sort=combo_order),
            x=alt.value(6),  # pixels fixos da borda esquerda — fica sempre colado no eixo Y, dentro da barra
            text="total:Q",
            opacity=alt.condition(combo_select, alt.value(1), alt.value(0)),
        )
    )

    fg_bars = (
        alt.Chart(df)
        .transform_filter(brush)
        .mark_bar()
        .encode(
            x=x_log,
            y=alt.Y("combo:N", sort=combo_order),
            color=alt.Color("combo:N", scale=alt.Scale(domain=domain, range=range_), legend=None),
        )
    )

    combo_scaffold = pd.DataFrame({"combo": combo_order})
    click_target = (
        alt.Chart(combo_scaffold)
        .mark_rect(opacity=0.001)
        .encode(y=alt.Y("combo:N", sort=combo_order))
        .add_params(combo_select)
        .properties(width=360)
    )

    candidate_bars = (
    alt.Chart(stack_df)
        .mark_bar(strokeWidth=1.5, stroke="black", fillOpacity=0.55)
        .encode(
            x=alt.X("y0:Q", scale=alt.Scale(type="log", domain=[1, 20000])),
            y=alt.Y("combo:N", sort=combo_order, title=None),
            x2="y1:Q",
            color=alt.Color("fill_color:N", scale=None, legend=None),
            opacity=alt.condition(show_candidates, alt.value(0.9), alt.value(0.0)),
            tooltip=["combo:N", "real_total:Q", "n_candidates:Q"],
            )
    )

    COL_SEQ_X, COL_SRC_X, COL_TGT_X, COL_PCT_X = 0, 360, 445, 520
    LIST_WIDTH = 600

    selected_combo_param = alt.param(name="selected_combo", value="")
    show_candidates_table = alt.param(name="show_candidates", value=False)

    def candidate_col_standalone(field, x, bold=False, color_enc=None):
        enc = dict(y=alt.Y("rank:O", axis=None), x=alt.value(x), text=alt.Text(f"{field}:N"))
        if color_enc is not None:
            enc["color"] = color_enc
        mark_kwargs = dict(align="left", fontSize=10, fontWeight="bold" if bold else "normal")
        if field == "sequence":
            mark_kwargs["font"] = "monospace"
        return (
            alt.Chart(cand_points)
            .transform_filter("datum.combo === selected_combo && show_candidates")
            .transform_window(rank="row_number()")
            .transform_filter("datum.rank <= 10")
            .mark_text(**mark_kwargs)
            .encode(**enc)
            .add_params(selected_combo_param, show_candidates_table)
        )

    candidate_list_body = alt.layer(
        candidate_col_standalone("sequence", COL_SEQ_X),
        candidate_col_standalone("source_combo", COL_SRC_X,
                    color_enc=alt.Color("source_combo:N", scale=alt.Scale(domain=FLAGS, range=[WONG[f] for f in FLAGS]), legend=None)),
        candidate_col_standalone("target_flag", COL_TGT_X,
                    color_enc=alt.Color("target_flag:N", scale=alt.Scale(domain=FLAGS, range=[WONG[f] for f in FLAGS]), legend=None)),
        candidate_col_standalone("pct_label", COL_PCT_X, bold=True),
    ).properties(width=LIST_WIDTH, height=210)


    header_df = pd.DataFrame({
        "label": ["Sequence", "Original Label", "Possible Label", "k-NN %"],
        "x": [COL_SEQ_X, COL_SRC_X, COL_TGT_X, COL_PCT_X],
    })
    candidate_list_header = (
        alt.Chart(header_df)
        .mark_text(align="left", fontSize=10, fontWeight="bold", color="#555")
        .encode(x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, LIST_WIDTH])), text="label:N")
        .properties(width=LIST_WIDTH, height=14)
    )
    
    table_chart = alt.vconcat(candidate_list_header, candidate_list_body, spacing=2).properties(
        title="Examples of potential candidates"
    )

    bars = alt.layer(bg_bars, bg_labels, fg_bars, 
                     click_target, candidate_bars
                     ).resolve_scale(x="shared", y="shared"
                                     ).properties(
            title="UpSet"
            )

    FLAG_SCALE = alt.Scale(domain=FLAGS)  # domínio explícito — evita a ordem inverter ao mesclar com dot_connectors

    dot_connectors = (
        alt.Chart(matrix_connectors)
        .mark_rule(stroke="#333333", strokeWidth=1.5)
        .encode(
            y=alt.Y("combo:N", sort=combo_order),
            x=alt.X("flag_start:N", scale=FLAG_SCALE, sort=FLAGS),
            x2="flag_end:N",
        )
    )

    dots = (
        alt.Chart(matrix_df)
        .mark_circle(size=140)
        .encode(
            y=alt.Y("combo:N", sort=combo_order, title=None, axis=alt.Axis(labels=False, ticks=False)),
            x=alt.X("flag:N", scale=FLAG_SCALE, sort=FLAGS, title=None,
                    axis=alt.Axis(orient="top", labelAngle=0, labelExpr=flag_abbrev_expr)),
            color=alt.condition(
                "datum.active",
                alt.Color("flag:N", scale=alt.Scale(domain=FLAGS, range=[WONG[f] for f in FLAGS]), legend=None),
                alt.value("#e6e6e6"),
            ),
        )
        .properties(width=90, height=UPSET_ROW_H * len(combo_order))
    )

    dots_with_lines = alt.layer(dot_connectors, dots).resolve_scale(x="shared", y="shared")

    upset = alt.hconcat(bars, dots_with_lines).resolve_scale(y="shared")

    left_column = umap_panel 

    main_chart = alt.hconcat(left_column, upset, spacing=10).resolve_scale(color="independent")
    n_sequences = len(df)
    n_candidates = len(cand_df)
    main_spec_json = main_chart.to_json()
    table_spec_json = table_chart.to_json()

    html_out = f"""<!DOCTYPE html>
        <html lang="pt-BR">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>ToxViz</title>
        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
        <style>
        :root {{
            --bg: #f4f5f7;
            --card-bg: #ffffff;
            --border: #e2e4e8;
            --text: #1f2328;
            --text-muted: #5b6270;
            --accent: #0072B2;
        }}

        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            padding: 0;
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
        }}

        .page {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 24px 48px;
        }}

        header.hero {{
            margin-bottom: 28px;
        }}

        header.hero h1 {{
            font-size: 26px;
            font-weight: 700;
            margin: 0 0 6px;
            text-align: center;
        }}

        header.hero p.subtitle {{
            font-size: 15px;
            color: var(--text-muted);
            margin: 0 0 16px;
            max-width: 780px;
            text-align: center;
            margin-left: auto;
            margin-right: auto;
        }}

        .badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 5px 12px;
            font-size: 12.5px;
            color: var(--text-muted);
        }}

        .badge strong {{
            color: var(--text);
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
            padding: 20px;
            margin-bottom: 20px;
            max-width: 100%;
            overflow-x: auto;
        }}

        .card-title {{
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: var(--text-muted);
            margin: 0 0 12px;
        }}

        #candidates-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13.5px;
            padding: 8px 12px;
            background: #f7f8fa;
            border: 1px solid var(--border);
            border-radius: 8px;
            width: fit-content;
            margin-bottom: 14px;
        }}

        #candidates-toggle input[type="checkbox"] {{
            width: 16px;
            height: 16px;
            accent-color: var(--accent);
            cursor: pointer;
        }}

        #candidates-toggle label {{
            cursor: pointer;
            user-select: none;
        }}

        #table-wrapper {{
            display: none;
        }}

        footer.methodology {{
            margin-top: 12px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
            font-size: 12.5px;
            color: var(--text-muted);
        }}

        footer.methodology p {{
            margin: 4px 0;
            max-width: 1350px;
        }}
        </style>
        </head>
        <body>
        <div class="page">

        <header class="hero">           
            <h1>Interactive Exploration of Peptide Toxicity</h1>
            <p class="subtitle">
            Visual comparison of peptide toxicity datasets, 
            showing the density of each isolated category and 
            candidates for unannotated multitoxicity identified 
            by proximity within the ESM-2 embedding space.
            </p>
            <div class="badges">
            <span class="badge"><strong>{n_sequences:,}</strong> unique sequences</span>
            <span class="badge">5 public databases</span>
            <span class="badge">4 toxicity categories</span>
            <span class="badge"><strong>{n_candidates}</strong> multitoxicity candidates</span>
            </div>
            
        </header>

        <div class="card">
            <div id="vis-main"></div>
            <div id="candidates-toggle">
            <input type="checkbox" id="candidates-checkbox">
            <label for="candidates-checkbox">Possible Candidates</label>
            </div>
            <div id="table-wrapper">
            <div id="vis-table"></div>
        </div>

        <footer class="methodology">
            <p><strong>Methodology summary:</strong> UMAP (480d → 2D, ESM-2) with density normalized 
            by the peak of each category. Candidates calculated via k-NN in the original 
            embedding space, with k derived statistically for each target category and a 
            binomial test against the population baseline rate to correct for the ceiling 
            effect of dominant categories.</p>
        </footer>

        </div>

        <script type="text/javascript">
        var mainSpec = {main_spec_json};
        var tableSpec = {table_spec_json};

        Promise.all([
            vegaEmbed("#vis-main", mainSpec, {{actions: false}}),
            vegaEmbed("#vis-table", tableSpec, {{actions: false}})
        ]).then(function(results) {{
            var mainView = results[0].view;
            var tableView = results[1].view;
            var checkbox = document.getElementById("candidates-checkbox");
            var tableWrapper = document.getElementById("table-wrapper");

            var hasSelection = false;

            function updateTableVisibility() {{
            tableWrapper.style.display = (hasSelection && checkbox.checked) ? "block" : "none";
            }}

            mainView.addDataListener("combo_select_store", function(name, value) {{
            hasSelection = value && value.length > 0;
            var comboVal = hasSelection ? value[0].values[0] : "";
            tableView.signal("selected_combo", comboVal).run();
            updateTableVisibility();
            }});

            checkbox.addEventListener("change", function() {{
            mainView.signal("show_candidates", checkbox.checked).run();
            tableView.signal("show_candidates", checkbox.checked).run();
            updateTableVisibility();
            }});
        }});
        </script>
        </body>
        </html>
        """

    with open(PROJECT_ROOT/"index.html", "w", encoding="utf-8") as f:
        f.write(html_out)

    cand_df.to_csv(DATA_DIR / "candidates.csv", index=False)
    print("Salvo em toxviz_interactive.html e candidates.csv")


if __name__ == "__main__":
    main()
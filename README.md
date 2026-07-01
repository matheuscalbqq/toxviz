# ToxViz — Exploração Interativa de Toxicidade Peptídica

**Trabalho prático — INF 723 Data Visualization 2026/01 (UFV)**
Proposta de visualização interativa aplicada a pesquisa própria (toxicidade de peptídeos), seguindo as diretrizes do IEEE VIS Bio+MedVis Challenge 2026.

🔗 **Demo ao vivo:** https://matheuscalbqq.github.io/toxviz/

---

## O problema

Bases de dados de toxicidade peptídica (hemolítica, citotóxica, neurotóxica, imunotóxica) são publicadas de forma fragmentada, cada uma com critérios próprios de anotação. Isso dificulta responder perguntas simples como: *quanto essas categorias realmente se sobrepõem?* e *quantos peptídeos podem ser multitóxicos sem que isso esteja anotado em nenhuma base individual?*

## A proposta

Consolidei 4 bases públicas (Hemolytik2, NTxPred2, IEDB, Tox-Prot) em **14.389 sequências únicas**, com embeddings ESM-2 (480d) por sequência, e construímos um protótipo interativo com duas visualizações coordenadas:

- **UMAP** — densidade espacial de cada categoria isolada (normalizada pelo pico da própria categoria, para não ser dominada pela classe majoritária), com interseções confirmadas e candidatos a multitoxicidade destacados.
- **UpSet plot** — interseções entre as 4 categorias (escala log), com candidatos empilhados sobre as barras confirmadas.

As duas visões são ligadas por *brushing & linking*: selecionar uma área no UMAP ou clicar numa barra do UpSet revela os pontos correspondentes em ambos.

### Candidatos a multitoxicidade não anotada
Além do dado confirmado, o protótipo identifica **candidatos estatísticos**: sequências anotadas com uma única categoria, mas cuja vizinhança no espaço de embedding ESM-2 (480 dimensões — não no UMAP 2D, que não preserva distância entre clusters distantes de forma confiável) é estatisticamente enriquecida para outra categoria. O método usa:
- k-NN com **k derivado estatisticamente por categoria-alvo**, o quecorrige um efeito-teto que tornava impossível detectar candidatos para a categoria majoritária com k fixo;
- teste binomial contra a taxa-base populacional (p < 0,001);
- probabilidade posterior via atualização Bayesiana (prior Beta centrado na taxa-base).

## Estrutura do repositório

```
toxviz/
├── index.html              # visualização final (abrir direto ou via GitHub Pages)
├── README.md
├── src/
│   ├── ESM2_embeddings.py           # extrai embeddings ESM-2 (roda em cluster/GPU)
│   └── build_interactive.py         # calcula UMAP + candidatos, gera index.html
└── data/
    ├── toxviz_with_embeddings.parquet
    ├── toxviz_umap.csv              # cache da projeção UMAP
    └── candidates.csv               # candidatos a multitoxicidade (saída final)
```

## Como rodar localmente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install altair pandas numpy scipy scikit-learn umap-learn pyarrow fair-esm torch

python3 src/clean_and_consolidate.py      # gera data/toxviz_consolidated.csv
python3 src/ESM2_embeddings.py            # gera data/toxviz_with_embeddings.parquet
python3 src/build_interactive.py          # gera index.html + data/candidates.csv
```

`index.html` é autocontido (usa Vega/Vega-Lite/Vega-Embed via CDN) — basta abrir num navegador com internet.

## Design e limitações (resumo — detalhes no documento do trabalho)

- **Paleta colorblind-safe** (Wong, 2011) em toda a visualização.
- **Densidade normalizada por categoria**, não por contagem absoluta — evita que a categoria majoritária (`immunotoxin`, 76% da base) domine visualmente as demais.
- **Limitação documentada**: medimos que 13 dos 27 peptídeos multitóxicos confirmados ficam visualmente "absorvidos" por uma única região no UMAP (densidade zero da segunda categoria na posição do ponto) — evidência concreta de que posição 2D não deve ser lida como prova de relação entre categorias distantes. Por isso, a detecção de candidatos usa o espaço de embedding original (480d), não o UMAP.

## Uso de Inteligência Artificial

Este projeto foi desenvolvido com assistência de IA (Claude, Anthropic) nas seguintes etapas:
- Geração e iteração do código de visualização (Python/Altair/Vega-Lite) e do HTML/CSS/JS de integração final.
- Discussão e formalização das decisões estatísticas do módulo de detecção de candidatos (correção de efeito-teto via k por categoria-alvo, estimação da probabilidade posterior Bayesiana).
- Depuração de bugs de composição de gráficos e sincronização de estado entre visualizações.
- Revisão de texto e organização deste README.

Todas as decisões de design, os dados utilizados, as escolhas metodológicas e a validação dos resultados foram definidas e revisadas pelos autores.

## Autores

[Nome 1], [Nome 2] — Departamento de Ciência da Computação, Universidade Federal de Viçosa
Disciplina: INF 723 — Data Visualization (Profa. Sabrina de Azevedo Silveira)

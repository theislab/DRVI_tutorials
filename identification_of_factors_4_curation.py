# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: drvi_tutorials
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Curating DRVI factor annotations
#
# The integrative step: for each factor-direction, compare **all** tools' outputs and assign a
# final label.
#
# This notebook reloads `embed` from disk and reads whatever the other notebooks stored in 
# `embed.uns` / `embed.var`. Tools whose notebook you did not run simply show "(not run)".
#
# For each factor: `inspect(...)` prints all evidence, `plot_factor_summary(...)` shows its UMAP
# pattern and top genes, `inspect_marker_gene(...)` tests a specific gene. Record decisions in
# `MANUAL_LABELS`.
#
# > This is one of four companion notebooks. Run whichever of
# > [cell types](./identification_of_factors_1_cell_types.html),
# > [biological processes](./identification_of_factors_2_biological_processes.html), and
# > [LLM tools](./identification_of_factors_3_llm_tools.html) you need first — they each store
# > results into the same `embed.h5ad` that this notebook reads.

# %% [markdown]
# ## Contact
#
# Questions: [scverse discourse](https://discourse.scverse.org/). Bugs:
# [issue tracker](https://github.com/theislab/drvi/issues).

# %% [markdown]
# ## Prerequisites
#
# Run one or more of the companion notebooks first so that `embed.h5ad` contains their stored
# results. This notebook needs only the base `drvi-py` package (`pip install drvi-py`, which
# provides `drvi` and `scanpy`) — none of the tutorial extras (LLM or enrichment) are required.

# %% [markdown]
# ## Imports

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import scanpy as sc
import pandas as pd
import anndata as ad
import drvi
from drvi.model import DRVI
from pathlib import Path

# %% [markdown]
# ## Setup

# %%
# Input/output directory holding the trained model and embeddings. Update accordingly.
io_dir = Path("./tmp_io/drvi_immune_128/").resolve()

embed_path = io_dir / "embed.h5ad"
adata = sc.read_h5ad(io_dir / "adata_preprocesses.h5ad")
embed = sc.read_h5ad(embed_path)
model = DRVI.load(io_dir / "drvi_model", adata)
scores_df = model.get_interpretability_scores(embed, adata, key="OOD_combined")

drvi_score_cutoff = 0.1
annot_col = "final_annotation"

# %% [markdown]
# ## Config: which stored results to show

# %%
# Top-N terms shown per enrichment tool.
CURATION_TOP_N = {"gseapy_enrichr": 3, "gprofiler": 5, "decoupler": 1}

ENRICHMENT_TOOLS = [
    ("gseapy_enrichr", "gseapy_enrichr_results", "Term"),
    ("gprofiler",      "gprofiler_results",      "name"),
    ("decoupler",      "decoupler_results",      "term"),
]

# LLM tools — populated only if you ran the LLM notebook; otherwise shown as None.
LLM_TOOLS = [
    ("llm celltype",   "llm_direct_results", "cell_type"),
    ("llm process",    "llm_direct_results", "biological_process"),
    ("cassia general", "cassia_results",     "Predicted General Cell Type"),
    ("cassia detailed", "cassia_results",    "Predicted Detailed Cell Type"),
    ("gs2txt",         "gs2txt_results",     "description"),
]

# Cell-type SMI matches written by the cell-types notebook (matching column names).
VAR_ANNOTATIONS = [
    ("known (SMI)", "positive_direction_match_with_user_annotations",
                    "negative_direction_match_with_user_annotations"),
    ("celltypist",  "positive_direction_match_with_celltypist",
                    "negative_direction_match_with_celltypist"),
]


# %% [markdown]
# ## Inspection helpers

# %%
def _var_value(embed, col, factor_title):
    if col not in embed.var.columns:
        return None
    row = embed.var.loc[embed.var["title"] == factor_title, col].dropna()
    return row.iloc[0] if not row.empty else None


def _uns_value(embed, key, col, factor, direction):
    df = embed.uns.get(key, pd.DataFrame())
    if df.empty or col not in df.columns:
        return None
    if "direction" in df.columns:
        mask = (df["factor"] == factor) & (df["direction"] == direction)
    else:
        mask = df["factor"] == f"{factor}{direction}"
    sub = df.loc[mask, col].dropna()
    return sub.iloc[0] if not sub.empty else None


# %%
def inspect(factor_label, embed, scores_df=scores_df, cutoff=drvi_score_cutoff):
    """Print a summary of all annotation tools for a single factor-direction."""
    factor, direction = factor_label[:-1].strip(), factor_label[-1]
    suf = "positive" if direction == "+" else "negative"
    print(f"\n{'=' * 60}\n  {factor_label}\n{'=' * 60}")

    print(f"\n- Manual -\n  {_var_value(embed, f'{suf}_direction_manual_label', factor)}")

    n_passing = (scores_df[factor_label] >= cutoff).sum()
    if n_passing == 0:
        print("\nNo genes above cutoff - uninformative direction.")
        return

    print(f"\nTop 10 genes ({n_passing} above cutoff):")
    for g, s in scores_df[factor_label].sort_values(ascending=False).head(10).items():
        print(f"  {g:<15} {s:.3f}")

    print("\n- Known / CellTypist -")
    for label, pos_col, neg_col in VAR_ANNOTATIONS:
        print(f"  {label:<18} {_var_value(embed, pos_col if direction == '+' else neg_col, factor)}")

    print("\n- LLM tools -")
    for label, key, col in LLM_TOOLS:
        print(f"  {label:<18} {_uns_value(embed, key, col, factor, direction)}")

    print("\n- Enrichment tools -")
    for label, key, term_col in ENRICHMENT_TOOLS:
        df = embed.uns.get(key, pd.DataFrame())
        if df.empty or term_col not in df.columns:
            print(f"  {label}: (not run)")
            continue
        hits = df[df["factor"] == factor_label][term_col].head(CURATION_TOP_N[label]).tolist() or ["(no hits)"]
        print(f"  {label}:\n" + "\n".join(f"    - {h}" for h in hits))


# %%
def plot_factor_summary(factor_label, embed, adata, scores_df, annot_col="final_annotation", n_genes=4):
    adata.obsm["X_drvi_umap"] = embed[adata.obs.index].obsm["X_umap"]
    drvi.utils.pl.plot_latent_dims_in_umap(
        embed, dim_subset=[factor_label], directional=True, additional_columns=[annot_col]
    )
    top = scores_df[factor_label].sort_values(ascending=False).index.to_list()[:n_genes]
    sc.pl.embedding(adata, "X_drvi_umap", color=top)


# %%
def inspect_marker_gene(gene, embed, adata, scores_df=scores_df, top_n=10):
    adata.obsm["X_drvi_umap"] = embed[adata.obs.index].obsm["X_umap"]
    sc.pl.embedding(adata, "X_drvi_umap", color=gene, cmap="viridis")
    print(f"Top {top_n} factors for gene '{gene}':")
    print(scores_df.loc[gene].sort_values(ascending=False).head(top_n))


# %% [markdown]
# ## Go factor by factor

# %%
factor_label = "DR 43-"
inspect(factor_label, embed)
plot_factor_summary(factor_label, embed, adata, scores_df)

# %% [markdown]
# ### Inspect a marker gene

# %%
marker_gene = "ISG15"
inspect_marker_gene(marker_gene, embed, adata)

# %% [markdown]
# ## Record manual decisions

# %%
MANUAL_LABELS = {
    "DR 43-": "Interferon response",
}

# %% [markdown]
# ## Write manual labels back

# %%
for d, suf in [("+", "positive"), ("-", "negative")]:
    mapping = {k[:-1].strip(): v for k, v in MANUAL_LABELS.items() if k.endswith(d)}
    embed.var[f"{suf}_direction_manual_label"] = embed.var["title"].map(mapping)

ad.settings.allow_write_nullable_strings = True
embed.write_h5ad(embed_path)
print(f"Saved to: {embed_path}")

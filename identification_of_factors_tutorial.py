# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: UV on drvi tutorials
#     language: python
#     name: drvi_t
# ---

# %% [markdown]
# # Identification of factors

# %% [markdown]
# QUESTIONS: Heatmaps: Both plot_latent_dims_in_heatmap calls (Celltypist and available annotations) still use embed, so the heatmaps show one row per factor (no separate +/− rows). 

# %% [markdown]
# In this notebook, we use the already trained DRVI model on the immune dataset to identify biological processes captured by each latent factor. We combine multiple complementary annotation strategies:
#
# 1. **Cell type annotation** — match factors to known cell types using existing labels (SMI) and pre-trained classifiers (CellTypist)
# 2. **Annotation of Biological Processes** using
#     * a. **Gene set enrichment analysis (GSEA)** — identify enriched pathways from ranked gene lists (BlitzGSEA)
#     * b. **Over-representation analysis (ORA)** — test for enriched gene sets using ordered queries (g:Profiler)
#     * c. **Regulator activity inference** — infer transcription factor or pathway activity using a statistical framework (decoupler) integrated with prior knowledge 
#
# Each tool operates on the gene-level interpretability scores produced by DRVI's built-in scoring API (`model.get_interpretability_scores`). DRVI provides two complementary scoring approaches that can be selected via the `INTERPRETABILITY_MODE` config variable:
#
# - **OOD (Out-of-Distribution)**: Uses decoder reconstructions to calculate per-gene effect scores. This is our suggested method to consider for finding cell-types and most specific genes of a program.
# - **IND (Within-Distribution)**: Iterates over all cells to compute weighted mean effects. Captures broader mechanistic effects including shared genes.
#
# All tools in this notebook are **guiding tools**: they summarize large gene-level patterns into interpretable scores, but they do **not** provide definitive labels. Their outputs should always be interpreted in context, compared across methods, and validated against known biology and the original data.
#
# **We always advise examination by a biologist and validation against published literature for any identified processes.**

# %% [markdown]
# ## Intro
#
# This notebook assumes that you have already trained a DRVI model and computed the interpretability scores (via `model.calculate_interpretability_scores` in the general pipeline).
#
# Please refer to the [General training and interpretability pipeline](./general_pipeline.html) tutorial.
#
# While we use the immune dataset as a running example, all code is dataset-agnostic. Configuration variables at the top of each section indicate what to change for your own data.

# %% [markdown]
# ## Contact
#
# For questions and help requests, you can reach out in the [scverse discourse](https://discourse.scverse.org/).
#
# If you found a bug, please use the [issue tracker](https://github.com/theislab/drvi/issues).

# %% [markdown]
# ## Adapting this notebook to your own dataset
#
# To reuse this notebook on a different dataset or DRVI model, check and update the following items:
#
# - **1. File paths and IO**
#   - Update `io_dir` to point to your project directory.
#   - Make sure the following files exist under `io_dir` with your data:
#     - `adata_preprocesses.h5ad` (preprocessed AnnData with HVGs and UMAP)
#     - `drvi_model/` (trained DRVI model directory)
#     - `embed.h5ad` 
#     - Optionally: `immune_all.h5ad` (or your equivalent full-gene data) for defining `all_genes`.
#
# - **2. Interpretability mode**
#   - Set `INTERPRETABILITY_MODE` in the Config cell to `"OOD"` (out-of-distribution, default) or `"IND"` (within-distribution).
#   - Both score sets must have been computed in the general pipeline via `model.calculate_interpretability_scores(embed, "OOD")` and `model.calculate_interpretability_scores(embed, "IND")`.
#
# - **3. Cell-level annotations (optional but recommended)**
#   - If you have cell-type labels, set `annot_col` to the corresponding column in `adata.obs`
#     (e.g. `"final_annotation"`).
#   - If you do **not** have annotations, set `annot_col = None` and skip:
#     - Section 1.1 (SMI with known annotations)
#     - Section 5 (visual validation on UMAP).
#
# - **4. Species and gene-sets**
#   - For BlitzGSEA, set:
#     - `gsea_db` (e.g `"GO_Biological_Process_2023"`)
#   - For g:Profiler, set:
#     - `organism` (e.g. `"hsapiens"`, `"mmusculus"`).
#     - `gp_source` to the GO / pathway collections you care about (e.g. `["GO:BP"]`, `["REAC"]`).
#   - For decoupler, set:
#     - `dc_geneset` (e.g. `"collectri"`, `"dorothea"`, `"progeny"` or another resource name).
#     - `dc_organism` to match your species (e.g. `"human"`, `"mouse"`).
#   - For non-human species, check that your gene-set resources and all_genes use the same gene-name casing; you may need to remove .str.upper() when working with mouse.
#
# - **5. CellTypist (optional)**
#   - Choose a model via `ct_model` that matches your tissue / species
#     (e.g. `"Immune_All_Low.pkl"` for PBMC, `"Developing_Mouse_Brain.pkl"` for mouse brain).
#   - If no suitable model exists, skip the CellTypist section and rely on your own annotations
#     plus the enrichment / decoupler tools using celltype databases.
#
# - **6. Significance thresholds**
#   - `fdr_threshold` controls:
#     - FDR cutoffs for BlitzGSEA and decoupler.
#     - The g:SCS-corrected p-value cutoff in g:Profiler (treated analogously to an FDR threshold).
#
# - **7. Manual curation**
#   - Use the exported `factor_annotation_curation.csv` as your central place to:
#     - Inspect top genes and tool suggestions per factor-direction.
#     - Define `MANUAL_LABELS` and `MANUAL_NOTES` in the helper cell.
#   - Re-import the curated CSV and re-run the final cells to store
#     `embed.var["annotation_final"]` and `embed.var["annotation_source"]` with your labels.

# %% [markdown]
# ## Install
#
# If you try DRVI on colab, the next cell will install dependencies.
#
# Please remove this part if your environment is already set up.

# %%
import sys
import subprocess

branch = "latest"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB and branch == "stable":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "drvi[tutorials]"])
elif IN_COLAB and branch != "stable":
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "git+https://github.com/theislab/drvi.git#egg=drvi[tutorials]"])

if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "celltypist", "blitzgsea", "gprofiler-official", "decoupler"])

# %% [markdown]
# ## Imports

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

import scvi
import drvi
from pathlib import Path
from drvi.model import DRVI
from drvi.utils.misc import hvg_batch

# %%
print("Last run with scvi-tools version:", scvi.__version__)
print("Last run with DRVI version:", drvi.__version__)

# %%
# Plot defaults
sc.settings.set_figure_params(dpi=100, frameon=False)
sc.set_figure_params(figsize=(3, 3))
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (3, 3)

# %% [markdown]
# ## Config

# %%
# Set input output directory
# We use tmp_io/ directory in the same place as this notebook. Update accordingly.
io_dir = Path("/home/icb/clara.sanchez/data/drvi_immune_128")

# ---------------------------------------------------------------------------
# Interpretability method
# ---------------------------------------------------------------------------
# "OOD" — Out-of-distribution: uses decoder reconstructions (faster, default)
# "IND" — Within-distribution: iterates over all cells (captures broader effects)
INTERPRETABILITY_MODE = "OOD"

_SCORE_KEY_MAP = {
    "OOD": "OOD_combined",
    "IND": "IND_linear_weighted_mean",
}
assert INTERPRETABILITY_MODE in _SCORE_KEY_MAP, (
    f"Invalid INTERPRETABILITY_MODE={INTERPRETABILITY_MODE!r}. Choose 'OOD' or 'IND'."
)
score_key = _SCORE_KEY_MAP[INTERPRETABILITY_MODE]
print(f"Interpretability mode: {INTERPRETABILITY_MODE} (score_key={score_key!r})")

# Global significance threshold used across all tools.
# Note: for BlitzGSEA and decoupler this is an FDR cutoff, while for g:Profiler
# it is applied to g:SCS-corrected p-values (not classical FDR).
fdr_threshold = 0.05

# %% [markdown]
# ## Load Data

# %%
# Update this path to point to your project directory
adata = sc.read_h5ad(io_dir / "adata_preprocesses.h5ad")
adata

# %% [markdown]
# ## Load DRVI outputs

# %%
model_path = io_dir / "drvi_model"
embed_path = io_dir / "embed.h5ad"


model = DRVI.load(model_path, adata)
embed = sc.read_h5ad(embed_path)

# %% [markdown]
# ## 0. Prepare shared inputs
#
# All annotation tools operate on the gene-level interpretability scores computed by DRVI. For each latent factor, the positive (+) and negative (−) direction each produce a vector of per-gene scores that quantify how much each gene's predicted expression changes. These scores serve as:
#
# - **Ranked gene lists** for GSEA-style tools (BlitzGSEA)
# - **Ordered gene queries** for ORA-style tools (g:Profiler)
# - **Gene × factor score matrices** for decoupler which uses statistical models (like ULM or MLR) to infer regulator activity by integrating these scores with a prior knowledge network
#
# The scoring approach is controlled by `INTERPRETABILITY_MODE` (set in Config above):
# - **OOD** scores emphasize factor-specific genes via decoder reconstructions.
# - **IND** scores capture broader mechanistic effects by averaging over all cells.
#
# We prepare these shared inputs once and reuse them across all tools.

# %%
# Remove vanished dimensions
embed_nv = embed[:, ~embed.var["vanished"].astype(bool)].copy()
factor_ids = embed_nv.var["title"].astype(str).tolist()
print(f"Active (non-vanished) factors: {embed_nv.n_vars}")

# Get gene-level interpretability scores via the DRVI model API.
# Returns a DataFrame of shape (n_genes, 2*n_factors): one column per factor-direction.
scores_df = model.get_interpretability_scores(embed, adata, key=score_key)
print(f"Scores shape: {scores_df.shape}  (genes x factor-directions)")

# Split into positive (+) and negative (-) direction DataFrames, filtered to non-vanished factors.
# Strip the +/- suffix so column names match factor_ids for downstream tools.
pos_cols = [f"{fid}+" for fid in factor_ids]
neg_cols = [f"{fid}-" for fid in factor_ids]
pos_df = scores_df[pos_cols].rename(columns=lambda c: c[:-1])
neg_df = scores_df[neg_cols].rename(columns=lambda c: c[:-1])

# Background gene universe: all genes measured in the experiment (before HVG filtering).
raw_data_path = io_dir / "immune_all.h5ad"
adata_full = sc.read_h5ad(raw_data_path, backed="r")

# NOTE: We uppercase gene names here, which is appropriate for human (HUGO symbols),
# but will break standard mouse Gene Symbols (e.g., "Cd4" -> "CD4").
# If you work with mouse or another species where case matters, remove `.str.upper()`
# and make sure your gene set resources use the same casing as your data.
all_genes = adata_full.var_names.astype(str).str.strip().str.upper()
adata_full.file.close()
all_genes = pd.Index(all_genes).drop_duplicates().tolist()
print(f"Background genes: {len(all_genes)}")


# %%
def build_inputs(df, all_genes=all_genes):
    """Build enrichment inputs from a genes x factors score matrix.

    Returns
    -------
    std : DataFrame
        Standardized genes x factors scores (uppercased, duplicates merged by max).
    ranked : dict
        {factor: Series sorted by descending score} for GSEA tools.
    """
    std = df.copy()
    std.index = std.index.astype(str).str.strip().str.upper()
    std = std.groupby(std.index).max()

    if all_genes is not None:
        idx = pd.Index(pd.Series(all_genes).astype(str)).drop_duplicates()
        std = std.reindex(idx)

    ranked = {c: std[c].dropna().sort_values(ascending=False) for c in std.columns}
    return std, ranked


pos_std, pos_ranked = build_inputs(pos_df)
neg_std, neg_ranked = build_inputs(neg_df)

# Factor-direction labels used throughout (e.g., "DR 36+")
factor_dir_labels = [f"{f}+" for f in factor_ids] + [f"{f}-" for f in factor_ids]

print(f"Factors: {len(factor_ids)}")
print(f"Factor-directions: {len(factor_dir_labels)}")
print(f"Genes per ranked list: {len(next(iter(pos_ranked.values())))}")

# %% [markdown]
# ## 1. Cell type annotation
#
# Some latent factors capture cell-type identity. We can identify these using:
# - **Known annotations** (if available): measure alignment between factors and annotated cell types via Scaled Mutual Information (SMI)
# - **CellTypist**: classify cells using pre-trained models and correlate class probabilities with factor activities
# - (Not described in this tutorial: using GSEA/ORA methods with Cell Type databases)

# %% [markdown]
# #### Cell Type Annotation Config

# %%
# Column in adata.obs containing cell type labels. Set to None if not available.
annot_col = "final_annotation" 

smi_threshold = 0.5 # Minimum SMI score between factor and cell-type probability profiles to consider a factor as associated with a cell type. Adjust as needed.

# %% [markdown]
# ### 1.1 Known annotations (SMI)
#
# If your dataset has existing cell type annotations, Scaled Mutual Information (SMI) measures how well each latent factor aligns with each annotated category. SMI is normalized to [0, 1], where 1 indicates perfect correspondence between a factor and a cell type.
#
# In this dataset we have annotations stored in `adata.obs["final_annotation"]`.
#
# **Skip this section if your dataset does not have cell type annotations.**

# %% [markdown]
# ####  Imports

# %%
import math
import networkx as nx
from drvi.utils.metrics import DiscreteDisentanglementBenchmark

# %% [markdown]
# #### Scaled Mutual Information

# %%
# Compute SMI between each factor-direction (+/-) and each annotated cell type.
titles = embed_nv.var["title"]
combined_X = np.hstack([embed_nv.X, -embed_nv.X])
combined_titles = [f"{t}+" for t in titles] + [f"{t}-" for t in titles]

benchmark = DiscreteDisentanglementBenchmark(
    combined_X,
    dim_titles=combined_titles,
    discrete_target=embed.obs[annot_col],
    metrics=["SMI-disc"],
    aggregation_methods=["LMS"],
)
benchmark.evaluate()
smi_similarity = benchmark.get_results_details()["SMI-disc"]
smi_similarity.index.name = "title"

# %%
print(f"SMI matrix shape: {smi_similarity.shape} (factor-directions x cell types)")
display(smi_similarity.head())

# %%
# Reshape the SMI matrix from wide to long format, then keep only pairs above the threshold.
smi_top_matches = (
    smi_similarity.reset_index()
    .melt(id_vars="title", value_vars=smi_similarity.columns)
    .query("value >= @smi_threshold")
    .reset_index(drop=True)
)
print(f"Factor–cell type pairs with SMI >= {smi_threshold}: {len(smi_top_matches)}")
display(smi_top_matches.sort_values("value", ascending=False))

# %% [markdown]
# #### Visualize with a Heatmap

# %%
drvi.utils.pl.plot_latent_dims_in_heatmap(
    embed, 
    annot_col, 
    title_col="title", 
    sort_by_categorical=True,
    figsize=(40, 16),
    show=False,
)

fig = plt.gcf()
for ax in fig.axes:
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
    for label in ax.get_yticklabels():
        label.set_fontsize(14)

plt.tight_layout()
plt.savefig("heatmap_large.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# #### Helper function for Plot Packed Network Visualization

# %%
def plot_packed_network(df, title_col="title", var_col="variable", val_col="value"):
    """Visualizes factor–cell type associations as a network with edge weights."""
    G = nx.from_pandas_edgelist(df, title_col, var_col, edge_attr=val_col)

    pos = {}
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    cols = math.ceil(len(components) ** 0.5)
    for i, nodes in enumerate(components):
        sub_pos = nx.spring_layout(G.subgraph(nodes), weight=val_col, k=0.5, seed=42)
        r, c = divmod(i, cols)
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + c * 3, y - r * 3)

    plt.figure(figsize=(14, 10))
    titles = set(df[title_col])
    nx.draw(
        G, pos,
        with_labels=True, font_size=8, font_weight="bold", node_size=600,
        node_color=["#A0CBE2" if n in titles else "#FF9E9E" for n in G.nodes()],
        width=[d[val_col] * 4 for u, v, d in G.edges(data=True)],
        edge_color="grey", alpha=0.6,
    )
    edge_labels = {(u, v): f"{d[val_col]:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis("off")
    plt.show()


# %% [markdown]
# #### Plot Packed Visualization

# %%
plot_packed_network(smi_top_matches)

# %% [markdown]
# ### 1.2 CellTypist
#
# [CellTypist](https://www.celltypist.org/) uses pre-trained logistic regression models trained on large-scale annotated atlases to classify individual cells. We calculate the Similarity Mutual Information (SMI) between the CellTypist probability matrix (cells × cell types) and the DRVI factor activity matrix (cells × factors) to identify which factors correspond to which cell types.
#
# **Skip this section if no CellTypist model matches your tissue.**
#

# %% [markdown]
# #### Imports

# %%
import celltypist
from celltypist import models

# %% [markdown]
# #### CellTypist Model

# %%
# Run celltypist.models.models_description() to see all available models. Choose one matching your tissue. 
ct_model = "Immune_All_Low.pkl"  # e.g., "Developing_Mouse_Brain.pkl" for mouse brain
models.download_models(force_update=False, model=ct_model)

ct_model = models.Model.load(model=ct_model)
# Run print(ct_model.cell_types) to see available cell types


# %% [markdown]
# #### CellTypist Annotation

# %% [markdown]
# Each cell receives a predicted label via logistic regression based on its transcriptomic profile. Setting majority_voting=True refines these labels by assigning the most frequent label within a cell's local neighborhood (kNN), reducing technical noise. The resulting per-cell labels are stored in adata.obs.

# %%
predictions = celltypist.annotate(adata, model=ct_model, majority_voting=True)
adata.obs["celltypist_labels"] = predictions.predicted_labels["predicted_labels"]
adata.obs["celltypist_majority"] = predictions.predicted_labels["majority_voting"]

# %% [markdown]
# #### Extract Probability Matrix

# %%
# Probability matrix: sigmoid-transformed decision scores (cells x cell types)
prob_matrix = predictions.probability_matrix
prob_matrix.index = adata.obs_names
print(f"Probability matrix: {prob_matrix.shape[0]} cells x {prob_matrix.shape[1]} cell types")

# %% [markdown]
# #### CellTypist Mutual Information

# %%
# Compute SMI between each factor-direction and each CellTypist-predicted cell type.
ct_titles = embed_nv.var["title"]
ct_combined_X = np.hstack([embed_nv.X, -embed_nv.X])
ct_combined_titles = [f"{t}+" for t in ct_titles] + [f"{t}-" for t in ct_titles]

ct_benchmark = DiscreteDisentanglementBenchmark(
    ct_combined_X,
    dim_titles=ct_combined_titles,
    discrete_target=adata.obs.loc[embed_nv.obs_names, "celltypist_majority"],
    metrics=["SMI-disc"],
    aggregation_methods=["LMS"],
)
ct_benchmark.evaluate()
ct_smi_matrix = ct_benchmark.get_results_details()["SMI-disc"]
ct_smi_matrix.index.name = "factor"
print(f"CellTypist SMI matrix: {ct_smi_matrix.shape} (factor-directions x CellTypist labels)")

# %%
common_cells = adata.obs_names.intersection(embed.obs_names)
embed = embed[common_cells].copy()
adata = adata[common_cells].copy()

embed.obs["celltypist_majority"] = (
    adata.obs["celltypist_majority"]
    .reindex(embed.obs_names)
)

drvi.utils.pl.plot_latent_dims_in_heatmap(
    embed, 
    "celltypist_majority", 
    title_col="title", 
    sort_by_categorical=True,
    figsize=(40, 16),
    show=False,  
)

fig = plt.gcf()
for ax in fig.axes:
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
    for label in ax.get_yticklabels():
        label.set_fontsize(14)

plt.tight_layout()
plt.savefig("heatmap_large.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# #### Summary Table

# %%
# Top CellTypist match per factor-direction
idx_name = ct_smi_matrix.index.name or "index"
ct_long = (
    ct_smi_matrix.reset_index()
    .melt(id_vars=idx_name, var_name="cell_type", value_name="smi")
    .rename(columns={idx_name: "factor"})
    .sort_values("smi", ascending=False)
    .drop_duplicates(subset="factor", keep="first")
)

ct_significant = ct_long.query("smi >= @smi_threshold").copy()

print(
    f"CellTypist matches with SMI >= {smi_threshold}: "
    f"{len(ct_significant)} / {len(ct_long)} factors"
)

# %% [markdown]
# ## 2. Biological process identification
#
# Factors that do not map to a single cell type often capture biological processes (e.g., interferon response, cell cycle, stress). We use three complementary enrichment approaches, each with different strengths:
#
# | Tool | Method | Input | Strengths |
# |------|--------|-------|-----------|
# | **BlitzGSEA** | Pre-ranked GSEA | Full ranked gene list | Fast; uses entire ranking; uses an analytical null distribution |
# | **g:Profiler** | Over-representation (ORA) | Ordered gene query | Robust multiple-testing (g:SCS); well-suited for biological pathways and GO terms |
# | **decoupler** | Activity Inference (ULM/MLM) | Gene score matrix + Prior Knowledge | Regression-based; identifies specific regulatory drivers (e.g., TFs) using curated networks |

# %% [markdown]
# ### 2.1 BlitzGSEA
#
# [BlitzGSEA](https://github.com/MaayanLab/blitzgsea) performs pre-ranked Gene Set Enrichment Analysis using an analytical approximation of the null distribution rather than permutations, enabling high-performance enrichment testing across many factors.
#
# - **Input**: Full ranked gene list (genes sorted by their DRVI effect scores, capturing the magnitude and direction of expression change)
# - **Output**: Normalized Enrichment Score (NES) and FDR-adjusted p-values per gene set
# - **Database**: Compatible with any standard .gmt file or Enrichr library (e.g., MSigDB, Reactome)

# %%
# Enrichr library to use. See Appendix for available databases.
# Common choices: "MSigDB_Hallmark_2020", "GO_Biological_Process_2023",
#                 "Reactome_2022", "KEGG_2021_Human"

gsea_db = "GO_Biological_Process_2023"

# %%
import blitzgsea as blitz

signature_lib = blitz.enrichr.get_library(gsea_db)
print(f"Loaded {gsea_db}: {len(signature_lib)} gene sets")

# %%
blitzgsea_rows = []

for fac in factor_ids:
    for direction, ranked_dict in [("pos", pos_ranked), ("neg", neg_ranked)]:
        factor_label = f"{fac}+" if direction == "pos" else f"{fac}-"
        series = ranked_dict[fac]

        # BlitzGSEA expects a DataFrame with columns "i" (gene) and "v" (score)
        signature = series.rename("v").reset_index().rename(columns={"index": "i"})
        signature["v"] = pd.to_numeric(signature["v"], errors="coerce")
        signature = signature.replace([np.inf, -np.inf], np.nan).dropna(subset=["v"])

        try:
            res = blitz.gsea(signature, signature_lib, processes=4)
            sig = res[res["fdr"] < fdr_threshold].sort_values("fdr")
            if len(sig):
                # Keep up to the top 3 most significant terms per factor-direction
                top_sig = sig.head(3)
                for term, row in top_sig.iterrows():
                    blitzgsea_rows.append({
                        "factor": factor_label,
                        "term": term,
                        "NES": round(float(row["nes"]), 3),
                        "FDR": float(row["fdr"]),
                    })
        except Exception as e:
            print(f"BlitzGSEA failed for {factor_label}: {e}")

blitzgsea_results = pd.DataFrame(blitzgsea_rows)
print(
    f"BlitzGSEA significant directions: {blitzgsea_results['factor'].nunique()} / {len(factor_dir_labels)} "
    f"(with up to 3 terms per direction)"
)
display(blitzgsea_results.sort_values(["factor", "FDR"]))

# %% [markdown]
# ### 2.2 g:Profiler
#
# [g:Profiler](https://biit.cs.ut.ee/gprofiler/) performs Over-Representation Analysis (ORA) using a hypergeometric test. It employs a custom multiple-testing correction (g:SCS), which is specifically optimized to handle the hierarchical and overlapping structure of Gene Ontology terms.
#
# In **ordered query** mode, g:Profiler processes genes sorted by their DRVI effect scores and iteratively tests enrichment at increasing increments. This approach automatically identifies the optimal gene set size for enrichment, making it more sensitive than using a fixed "top-N" cutoff for continuous latent factor scores.
#
# How it works:
# - **Input**: Ordered gene list (genes sorted by absolute or directional traverse effect scores)
# - **Output**: Enriched terms with p-values corrected via g:SCS
# - **Database**: Comprehensive support for GO (BP, MF, CC), Reactome, KEGG, WikiPathways, and regulatory motifs
#
# Many functional annotation collections (for example Gene Ontology, pathway databases, or phenotype ontologies) are hierarchical and redundant. Broad "umbrella" terms tend to be enriched across multiple latent factors, while more specific child terms capture finer-grained biology. Because of this structure, there is rarely a single automatically chosen term that is clearly "the" correct label for a factor.
#
# In this tutorial, we therefore treat g:Profiler as a tool to obtain **shortlists of enriched terms per factor-direction**, not as an automatic source of single-factor labels.
#
# **Key configuration** (set in the next cell): `organism` (e.g. `"hsapiens"`, `"mmusculus"`) and `gp_source` (e.g. `["GO:BP"]`, `["REAC"]`).

# %%
# Organism string. Common values: "hsapiens", "mmusculus", "drerio"
organism = "hsapiens"

# Source database(s).
# Common choices: ["GO:BP"], ["GO:MF"], ["GO:CC"], ["REAC"], ["KEGG"], ["HP"]
gp_source = ["GO:BP"]

# %%
from gprofiler import GProfiler

gp = GProfiler(return_dataframe=True)


def run_gprofiler_for_factor(genes, factor_label):
    """Run g:Profiler ordered-query ORA for a single factor-direction."""
    genes = pd.Series(genes).dropna().astype(str).drop_duplicates().tolist()
    if not genes:
        return pd.DataFrame()

    res = gp.profile(
        organism=organism,
        query=genes,
        sources=gp_source,
        ordered=True,
        user_threshold=fdr_threshold,
        background=all_genes,
    )
    if res is None or res.empty:
        return pd.DataFrame()

    res = res.copy()
    res["factor"] = factor_label
    return res


# %%
gprofiler_parts = []

for fac in factor_ids:
    for direction, ranked_dict in [("pos", pos_ranked), ("neg", neg_ranked)]:
        factor_label = f"{fac}+" if direction == "pos" else f"{fac}-"
        genes = ranked_dict[fac].index.tolist()
        gprofiler_parts.append(run_gprofiler_for_factor(genes, factor_label))

gprofiler_all = pd.concat(
    [x for x in gprofiler_parts if not x.empty], ignore_index=True
) if any(not x.empty for x in gprofiler_parts) else pd.DataFrame()

sig = gprofiler_all[gprofiler_all["p_value"] < fdr_threshold].copy()
print(
    f"g:Profiler results: {len(sig)} significant rows across "
    f"{sig['factor'].nunique()} factor-directions "
    f"(g:SCS-corrected p < {fdr_threshold})."
)

# %% [markdown]
# #### Explore top pathway terms per factor-direction

# %%
# Example: "DR 2+", "DR 36-", etc.
factor_to_inspect = "DR 2+"
top_n_gp_terms = 10  # or 5, as you prefer

if not gprofiler_all.empty:
    available = sorted(gprofiler_all["factor"].unique())
    print(f"Available factor-directions ({len(available)}): {available[:10]}{' ...' if len(available) > 10 else ''}\n")

    df_fac = gprofiler_all[
        (gprofiler_all["factor"] == factor_to_inspect)
        & (gprofiler_all["p_value"] < fdr_threshold)
    ].copy()

    if df_fac.empty:
        print(f"No significant g:Profiler terms for factor {factor_to_inspect} at this threshold.")
    else:
        df_fac = (
            df_fac.sort_values("p_value")
            [["factor", "name", "p_value", "intersection_size", "term_size"]]
            .rename(columns={"name": "term"})
            .head(top_n_gp_terms)
        )
        print(f"Top {len(df_fac)} g:Profiler terms for {factor_to_inspect}:")
        display(df_fac)
else:
    print("gprofiler_all is empty — run the g:Profiler enrichment cell above first.")

# %% [markdown]
# ### 2.3 decoupler
#
# [decoupler](https://decoupler-py.readthedocs.io/) uses regression-based methods (Univariate/Multivariate Linear Models, z-score) and weighted sums to infer the activity of regulators from gene-level scores. Unlike enrichment-based methods, it models the relationship between observed gene scores and a Prior Knowledge Network (PKN), quantifying the specific influence of a regulator.
#
# decoupler provides access to curated regulatory resources from [OmniPath](https://omnipathdb.org/):
#
# - **CollecTRI**: Comprehensive transcription factor (TF) → target gene interactions, well-suited for discovering TF-level drivers of latent factors.
# - **DoRothEA**: TF regulons categorized by confidence levels (A–D) based on supporting evidence; also TF-centric, with tunable stringency.
# - **PROGENy**: Pathway footprints that infer upstream pathway activity (e.g., Hypoxia, EGFR, TGFb) from downstream responsive genes.
#
# For DRVI latent factor annotation, **CollecTRI and DoRothEA are usually the most informative options**, because latent factors can capture TF-driven gene programs or cell identity signatures. **PROGENy can still be used as an exploratory option**, but in practice it may yield few or no strongly significant hits when factors are not dominated by a small set of canonical signaling pathways (as in this immune example).
#
# Multiple decoupler methods are run sequentially and combined via a consensus step to produce robust p-values.
#
# **Runtime-relevant configuration knobs:**
#
# - **`dc_methods`**: methods run sequentially; dropping `"mlm"` gives ~33% speedup with minimal consensus impact.
# - **`dc_min` / `tmin`**: Minimum number of genes from the gene set that must be present in the data for a valid enrichment test
# - **`dc_geneset`**: network size scales runtime linearly. CollecTRI (~1,185 regulators) is slowest; DoRothEA A-B (~500) is ~2x faster at some coverage cost.

# %%
# Gene set / network to use.
# Recommended options for factor annotation: "collectri", "dorothea".
# PROGENy ("progeny") is more pathway-focused and may give few strong hits
# if latent factors are not dominated by canonical signaling pathways.
dc_geneset = "collectri"  # or "dorothea"

# Organism. Must match ORGANISM above: "human" for hsapiens, "mouse" for mmusculus
dc_organism = "human"

dc_methods = ["ulm", "zscore"]
dc_min = 10 

# %%
import decoupler as dc
from statsmodels.stats.multitest import multipletests

# %%
# Load the prior knowledge network for the selected gene set resource.
net_dispatch = {
    "collectri": lambda: dc.op.collectri(organism=dc_organism),
    "dorothea": lambda: dc.op.dorothea(organism=dc_organism, levels=["A", "B", "C"]),
    "progeny": lambda: dc.op.progeny(organism=dc_organism),
}
net = net_dispatch.get(
    dc_geneset.strip().lower(),
    lambda: dc.op.resource(name=dc_geneset, organism=dc_organism),
)()

cols = ["source", "target"] + (["weight"] if "weight" in net.columns else [])
net = net[cols].dropna().drop_duplicates().reset_index(drop=True)
print(f"Network: {len(net)} interactions, {net['source'].nunique()} regulators")


# %%
def run_decouple(df_factors_by_genes):
    """Run decoupler consensus on a factors x genes score matrix.

    Expects a (factor-directions x genes) DataFrame with rows like "DR 1+", "DR 1-".
    """
    mat = df_factors_by_genes.copy()
     # Standardize gene names to uppercase to match the PKN, and keep only genes present in the network.
    mat.columns = mat.columns.astype(str).str.strip().str.upper()
    targets = net["target"].astype(str).str.strip().str.upper().unique()
    keep_cols = [g for g in mat.columns if g in targets]
    mat = mat[keep_cols]
    mat = mat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    net_use = net.copy()
    net_use["target"] = net_use["target"].astype(str).str.strip().str.upper()

    res = dc.mt.decouple(
        data=mat,
        net=net_use,
        methods=dc_methods,
        cons=False,
        tmin=dc_min,
        verbose=True,
    )
    _, pvals = dc.mt.consensus(res)

    out = pvals.stack().rename("p_value").reset_index()
    out.columns = ["factor", "term", "p_value"]

    _, p_adj, _, _ = multipletests(out["p_value"].values, method="fdr_bh")
    out["p_adj"] = p_adj
    return out[["factor", "term", "p_value", "p_adj"]]


# Build a factor-directions x genes matrix for both positive and negative directions.
pos_T = pos_std.T.copy()
pos_T.index = [f"{f}+" for f in pos_T.index]
neg_T = neg_std.T.copy()
neg_T.index = [f"{f}-" for f in neg_T.index]
combined_std = pd.concat([pos_T, neg_T], axis=0)

decoupler_all = run_decouple(combined_std)

# Keep the most significant regulator per factor-direction for a summary view.
decoupler_results = (
    decoupler_all[decoupler_all["p_adj"] < fdr_threshold]
    .sort_values("p_adj")
    .groupby("factor", as_index=False)
    .first()
    [["factor", "term", "p_adj"]]
)

print(
    f"decoupler significant regulators for "
    f"{decoupler_results['factor'].nunique()} / {len(factor_dir_labels)} factor-directions "
    f"(top 1 regulator per direction with FDR < {fdr_threshold})"
)
display(decoupler_results.sort_values("p_adj"))

# %% [markdown]
# ## 3. Curation table and export
#
# We now merge evidence from all tools into a single curation table. Each row represents one factor-direction, with the top hits from each tool (if significant) shown in a compact format. The table also includes the top 10 genes driving each factor-direction to facilitate manual validation via literature or LLMs.
#
# **Recommended workflow (code-based labeling):**
# 1. Run the curation cells below to build the table and export a CSV template.
# 2. Edit `MANUAL_LABELS` in the helper cell to define your final annotations per factor-direction.
# 3. Re-run the curation and export cells to refresh the CSV with your labels.
# 4. Run the re-import cell to store final annotations in the embedding object.
#

# %%
# Build the base table: all factor-directions with top 10 genes
curation_rows = []
for fac in factor_ids:
    for direction, ranked_dict in [("pos", pos_ranked), ("neg", neg_ranked)]:
        factor_label = f"{fac}+" if direction == "pos" else f"{fac}-"
        top_genes = ", ".join(ranked_dict[fac].head(10).index.tolist())
        curation_rows.append({"factor": factor_label, "top_genes": top_genes})

curation = pd.DataFrame(curation_rows)

# Known-annotation SMI (section 1.1): top cell-type match per factor-direction
known_map = {}
if len(smi_top_matches):
    for fac in curation["factor"].unique():
        sub = smi_top_matches[smi_top_matches["title"] == fac]
        if len(sub):
            row = sub.loc[sub["value"].idxmax()]
            known_map[fac] = f"{row['variable']} || SMI={row['value']:.2f}"
curation["known_annotation"] = curation["factor"].map(known_map).fillna("")

# CellTypist column (per factor-direction SMI; directions already in ct_significant)
ct_map = {}
if len(ct_significant):
    for _, row in ct_significant.iterrows():
        label = f"{row['cell_type']} || SMI={row['smi']:.2f}"
        ct_map[row["factor"]] = label
curation["celltypist"] = curation["factor"].map(ct_map).fillna("")

# BlitzGSEA column (top 5 terms per factor-direction)
bg_map = {}
if len(blitzgsea_results):
    for fac, grp in blitzgsea_results.sort_values("FDR").groupby("factor"):
        top = grp.head(5)
        bg_map[fac] = " | ".join(
            f"{r['term']} || NES={r['NES']:.2f} || FDR={r['FDR']:.2e}"
            for _, r in top.iterrows()
        )
curation["blitzgsea"] = curation["factor"].map(bg_map).fillna("")

# g:Profiler column (top 10 terms per factor-direction)
gp_map = {}
if not gprofiler_all.empty:
    sig = gprofiler_all[gprofiler_all["p_value"] < fdr_threshold].copy()
    for fac, grp in sig.sort_values("p_value").groupby("factor"):
        top = grp.head(10)
        gp_map[fac] = " | ".join(
            f"{r['name']} (p={r['p_value']:.2e})" for _, r in top.iterrows()
        )
curation["gprofiler"] = curation["factor"].map(gp_map).fillna("")

# decoupler column (top 5 regulators per factor-direction)
# Filter decoupler results to significant hits before grouping.
dc_map = {}
dc_sig = decoupler_all[decoupler_all["p_adj"] < fdr_threshold] if len(decoupler_all) else decoupler_all
if len(dc_sig):
    for fac, grp in dc_sig.sort_values("p_adj").groupby("factor"):
        top = grp.head(5)
        dc_map[fac] = " | ".join(
            f"{r['term']} || p_adj={r['p_adj']:.2e}" for _, r in top.iterrows()
        )
curation["decoupler"] = curation["factor"].map(dc_map).fillna("")

# Empty columns for manual curation (filled later via MANUAL_LABELS at import time)
curation["manual_label"] = ""
curation["manual_notes"] = ""

display(curation)

# %%
curation_path = io_dir / "factor_annotation_curation.csv"
curation.to_csv(curation_path, index=False)
print(f"Curation table exported to: {curation_path}")
print("\nEdit MANUAL_LABELS in the helper cell above, then re-run the curation and export cells.")
print("Then run the cells below to finalize and store annotations in embed.var.")

# %% [markdown]
# ### Manual labeling helper
#
# Edit the dictionaries below to define your final annotations. Use factor-direction labels (e.g. `"DR 2+"`, `"DR 36-"`) as keys. After editing, re-run the curation table cell above and the export cell to update the CSV with your labels.

# %%
# Define your final annotations here. Keys are factor-direction labels (e.g. "DR 2+", "DR 36-").
# Example: MANUAL_LABELS = {"DR 2+": "T cell activation", "DR 36-": "Monocyte differentiation"}
MANUAL_LABELS = {"DR 36+": "Monocyte progenitor"}

# Optional: free-text notes per factor-direction
MANUAL_NOTES = {}


def set_label(factor, label, notes=None):
    """Helper to add or update a label for a factor-direction."""
    MANUAL_LABELS[factor] = label
    if notes is not None:
        MANUAL_NOTES[factor] = notes
    print(f"Set {factor} -> {label}")


# %% [markdown]
# ### 3.1 Re-import and finalize
#
# Re-import the curation CSV and store final annotations in the embedding object. The `final_label` is taken **only** from `manual_label` (as populated from `MANUAL_LABELS` in the helper cell). No automatic fallback to tool-based columns.

# %%
curation_edited = pd.read_csv(curation_path)

# Apply manual labels from MANUAL_LABELS at import time (overrides any CSV edits)
curation_edited["manual_label"] = curation_edited["factor"].map(lambda f: MANUAL_LABELS.get(f, ""))


def pick_final_label(row):
    """Pick final annotation: use only the manual label if provided."""
    if pd.notna(row.get("manual_label")) and str(row["manual_label"]).strip():
        return str(row["manual_label"]).strip(), "manual"
    return "", "none"

labels_and_sources = curation_edited.apply(pick_final_label, axis=1, result_type="expand")
curation_edited["final_label"] = labels_and_sources[0]
curation_edited["label_source"] = labels_and_sources[1]

display(curation_edited[["factor", "final_label", "label_source"]].head(20))

# %%
# Store annotations in embed.var for persistence
# Map factor-direction labels back to factor base names for embed.var
# We take the "+" direction label as the primary annotation per factor
annot_by_factor = {}
for _, row in curation_edited.iterrows():
    fac_base = row["factor"][:-1]  # strip +/- suffix
    direction = row["factor"][-1]
    if direction == "+" and row["final_label"]:
        annot_by_factor[fac_base] = (row["final_label"], row["label_source"])
    elif fac_base not in annot_by_factor and row["final_label"]:
        annot_by_factor[fac_base] = (row["final_label"], row["label_source"])

embed.var["annotation_final"] = embed.var["title"].map(
    lambda t: annot_by_factor.get(t, ("", ""))[0]
)
embed.var["annotation_source"] = embed.var["title"].map(
    lambda t: annot_by_factor.get(t, ("", ""))[1]
)

print("Annotations stored in embed.var:")
display(embed.var[["title", "vanished", "annotation_final", "annotation_source"]].head(20))

# %%
embed.write_h5ad(embed_path)
print(f"Updated embedding saved to: {embed_path}")

# %% [markdown]
# ## 4. Visual validation
#
# Sanity-check a few annotated factors by visualizing their activity on the UMAP alongside the preannotated cell type.
#
# > These plots require a per-cell annotation column in `adata.obs` (configured as `annot_col`, e.g. `final_annotation`). If your dataset does not have such annotations, you can skip this section.

# %%
example_dims = ["DR 36"]  # Replace with factor-directions you want to visualize

if annot_col is not None and example_dims:
    drvi_factors_df = pd.DataFrame(embed_nv.X, index=embed_nv.obs_names, columns=factor_ids)
    for dim in example_dims:
        if dim in drvi_factors_df.columns:
            adata.obs["_factor_check"] = drvi_factors_df[dim].reindex(adata.obs_names).values
            annot = embed.var.set_index("title").loc[dim, "annotation_final"]
            print(f"\n{dim} — annotated as: {annot}")

            sc.pl.umap(
                adata,
                color=["_factor_check", annot_col],
                ncols=2,
                frameon=False,
                title=[f"Factor: {dim}", annot_col],
            )


# %% [markdown]
# ## Appendix: Database reference
#
# The table below lists curated databases available for factor annotation, organized by domain. You can swap any of the tool-specific config variables above (e.g., `gsea_db`, `gp_source`, `dc_geneset`) to use different databases.
#
#
# ### Biological process databases
#
# | Database | Description | BlitzGSEA| g:Profiler | 
# |----------|-------------|-------------------|------------|
# | MSigDB Hallmark | 50 well curated non redundant biological states | `MSigDB_Hallmark_2020` | — |
# | GO Biological Process | Comprehensive hierarchical processes | `GO_Biological_Process_2025` | `GO:BP` |
# | GO Cellular Component | Subcellular localization | `GO_Cellular_Component_2025` | `GO:CC` |
# | GO Molecular Function | Molecular activities | `GO_Molecular_Function_2025` | `GO:MF` |
# | Reactome | Detailed, reaction-based pathways | `Reactome_Pathways_2024` | `REAC` |
# | KEGG | Classic metabolic/signaling maps | `KEGG_2026` | `KEGG` | 
# | WikiPathways | Community-curated biological maps | `WikiPathways_2024_Human` | `WP` |
#
# ### Regulatory networks (decoupler only)
#
# | Network | Description | decoupler name | Notes|
# |---------|-------------|---------------|---------|
# | CollecTRI | TF → target gene regulons| `collectri` | Recommended for identifying TF drivers of a factor |
# | PROGENy | Pathway-responsive genes signatures | `progeny` | Best for signaling (TGFb, MAPK, etc.) activity |
# | DoRothEA |  TF → target gene interactions | `dorothea` |  Curated resource for Transcription Factor (TF) activity; uses confidence levels (A-D) |
#
# ### Clinical and Disease Phenotypes
#
# | Database | Description | BlitzGSEA| g:Profiler |
# |----------|-------------|-------------------|------------|
# | Human Phenotype Ont. | Genes linked to clinical signs | Human_Phenotype_Ontology | HP |
# | OMIM | Human genes and genetic disorders | OMIM_Disease | OMIM
#

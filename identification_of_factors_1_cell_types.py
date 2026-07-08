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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %% [markdown]
# # Identifying cell types of DRVI factors
#
# Some DRVI latent factors capture **cell-type identity**. When cell-type annotations are
# available, we can use them to identify such factors — by measuring how well each factor's
# direction aligns with a categorical annotation, using **Scaled Mutual Information (SMI)**
# (normalized to [0, 1]; 1 = perfect one-to-one correspondence). The annotations can come from:
#
# 1. **User annotations** — labels already present in `adata.obs` (Section 1).
# 2. **Predictions from a pre-trained model** — if you have no annotations (or want finer
#    subtypes), borrow labels from a model already trained on annotated atlases (Section 2).
#
# Importantly, this supervised information is never given to DRVI during training, so the factors
# themselves are **not biased** by it — the annotations only help us label the factors after
# training. Any assignment made here can later be independently verified using each factor's
# top-scoring genes.
#
# Both sections share the same SMI machinery, defined once as helper functions.
#
# > This is one of three companion notebooks that identify DRVI factors. The others cover
# > [biological processes via enrichment](./identification_of_factors_2_biological_processes.html)
# > and [LLM-based tools](./identification_of_factors_3_llm_tools.html). They all read and write
# > the same `embed.h5ad`, so results accumulate.
#
# **We always advise examination by a biologist and validation against published literature.**

# %% [markdown]
# ## Prerequisites
#
# This notebook assumes you have already trained a DRVI model and computed interpretability
# scores (`model.calculate_interpretability_scores`). See the
# [general training and interpretability pipeline](./general_pipeline.html).
#
# **Adapting to your own model** — only Section 0 and a couple of config values change:
#
# - Point `io_dir` at your project directory (holding `adata_preprocesses.h5ad`,
#   `drvi_model/`, `embed.h5ad`).
# - **Section 1**: set `annot_col` to the `adata.obs` column with your cell-type labels.
# - **Section 2**: to use CellTypist as the pre-trained model, set `ct_model` to a CellTypist
#   model matching your tissue/species (e.g. `"Immune_All_Low.pkl"` for PBMC,
#   `"Developing_Mouse_Brain.pkl"` for mouse brain).

# %% [markdown]
# ## Contact
#
# For questions and help requests, reach out on the
# [scverse discourse](https://discourse.scverse.org/). If you found a bug, please use the
# [issue tracker](https://github.com/theislab/drvi/issues).

# %% [markdown]
# ## Install
#
# On Colab, the next cell installs the dependencies. Remove it if your environment is ready.

# %%
import sys
import subprocess

branch = "latest"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB and branch == "stable":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "drvi-py[tutorials]"])
elif IN_COLAB and branch != "stable":
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "git+https://github.com/theislab/drvi.git#egg=drvi-py[tutorials]"])

if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "celltypist"])

# %% [markdown]
# ## Imports

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

import scvi
import drvi
from drvi.model import DRVI
from drvi.utils.metrics import DiscreteDisentanglementBenchmark

# %%
print("Last run with scvi-tools version:", scvi.__version__)
print("Last run with DRVI version:", drvi.__version__)

# %%
# Plot defaults
sc.set_figure_params(dpi=100, frameon=False, figsize=(3, 3))
plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## 0. Setup

# %% [markdown]
# ### Config

# %%
# Input/output directory holding the trained model and embeddings. Update accordingly.
# (Same layout as the general_pipeline tutorial.)
io_dir = Path("./tmp_io/drvi_immune_128/").resolve()
print(f"Using directory: {io_dir}")

# %% [markdown]
# ### Load model and embeddings

# %%
# AnnData DRVI was trained on (HVG-selected).
adata = sc.read_h5ad(io_dir / "adata_preprocesses.h5ad")

model = DRVI.load(io_dir / "drvi_model", adata)

# Latent AnnData with interpretability scores (from the general pipeline).
embed_path = io_dir / "embed.h5ad"
embed = sc.read_h5ad(embed_path)
embed

# %% [markdown]
# ## Shared SMI helpers
#
# Sections 1 and 2 both compare factor activities against a categorical label with SMI. The
# logic is identical, so we define it once here.
#
# `DiscreteDisentanglementBenchmark` (the class we use to evaluate models) computes pairwise
# similarity between latent factors and a supervised target. We split each factor into its
# positive and negative directions and drop vanished ones.

# %%
def build_directional_df(embed):
    """Cells x (non-vanished factor-directions). Column names carry a '+'/'-' suffix."""
    embed_pos = embed[:, ~embed.var["vanished_positive_direction"]].copy()
    embed_neg = embed[:, ~embed.var["vanished_negative_direction"]].copy()
    embed_pos.var.index = embed_pos.var["title"] + "+"
    embed_neg.var.index = embed_neg.var["title"] + "-"
    embed_pos.X = embed_pos.X.clip(min=0)
    embed_neg.X = -embed_neg.X.clip(max=0)
    return pd.concat([embed_pos.to_df(), embed_neg.to_df()], axis=1).loc[embed.obs.index]

# %%
def smi_matches(embed_directional_df, target, threshold):
    """SMI between every factor-direction and every category of `target`.

    Returns (full SMI matrix, long table of pairs with SMI >= threshold sorted descending).
    """
    benchmark = DiscreteDisentanglementBenchmark(
        embed_directional_df.values,
        dim_titles=embed_directional_df.columns,
        discrete_target=target,
        metrics=["SMI"],
        aggregation_methods=[],
    )
    benchmark.evaluate()
    smi = benchmark.get_results_details()["SMI"]
    smi.index.name = "title"

    top = (
        smi.reset_index()
        .melt(id_vars="title", value_vars=smi.columns)
        .query("value >= @threshold")
        .reset_index(drop=True)
        .sort_values("value", ascending=False)
    )
    return smi, top

# %%
def plot_packed_network(df, title_col="title", var_col="variable", val_col="value", figsize=(14, 10)):
    """Visualize factor-cell type associations as a network with edge weights."""
    G = nx.from_pandas_edgelist(df, title_col, var_col, edge_attr=val_col)

    pos = {}
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    cols = 3
    for i, nodes in enumerate(components):
        sub_pos = nx.spring_layout(G.subgraph(nodes), weight=val_col, k=0.5, seed=42)
        r, c = divmod(i, cols)
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + c * 3, y - r * 3)

    plt.figure(figsize=figsize)
    titles = set(df[title_col])
    weights = [d[val_col] for u, v, d in G.edges(data=True)]
    nx.draw(
        G, pos,
        with_labels=True, font_size=8, font_weight="bold", node_size=600,
        node_color=["#A0CBE2" if n in titles else "#FF9E9E" for n in G.nodes()],
        width=[w * 4 for w in weights],
        edge_color=weights, edge_cmap=plt.cm.Oranges, alpha=0.6,
    )
    edge_labels = {(u, v): f"{d[val_col]:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis("off")
    plt.show()

# %%
def store_matches(embed, top, suffix):
    """Store the best match per factor-direction in `embed.var` and the full table in `embed.uns`.

    Writes columns `positive_direction_match_with_<suffix>` and
    `negative_direction_match_with_<suffix>`.
    """
    first = top.drop_duplicates(subset=["title"]).copy()
    first["direction"] = first["title"].str[-1:]
    first["title"] = first["title"].str[:-1]

    embed.var.set_index("title", drop=False, inplace=True)
    for d, sign in [("positive", "+"), ("negative", "-")]:
        col = f"{d}_direction_match_with_{suffix}"
        embed.var[col] = None
        sub = first.query("direction == @sign")
        embed.var.loc[sub["title"], col] = sub["variable"].values
    embed.var.index = embed.var["original_dim_id"].astype(int).astype(str)
    embed.var.index.name = None

    embed.uns[f"best_smi_matching_{suffix}_results"] = top

# %% [markdown]
# ## 1. Cell types from user annotations
#
# If your dataset already has cell-type labels, SMI shows which latent factors encode them —
# the highest-confidence anchor available. Note that DRVI factors can be **finer** than manual
# labels (subtypes, developmental stages, shared processes), so most factors will not match a
# label, and that is expected: the other notebooks characterize the rest.
#
# **Skip this section if your dataset has no cell-type annotations** and instead follow Section 2.

# %% [markdown]
# ### Config

# %%
# Column in adata.obs / embed.obs with cell-type labels.
annot_col = "final_annotation"

# Minimum SMI to call a factor associated with a cell type. Adjust as needed.
smi_threshold = 0.7

# %% [markdown]
# ### Visualize the factor x cell-type relationship

# %%
drvi.utils.pl.plot_latent_dims_in_heatmap(embed, annot_col, title_col="title", sort_by_categorical=True)

# %% [markdown]
# Some factors show a clear one-to-one relationship with a cell type. Let's identify them.

# %% [markdown]
# ### Compute SMI and keep the strong matches

# %%
embed_directional_df = build_directional_df(embed)
smi_similarity, smi_top_matches = smi_matches(embed_directional_df, embed.obs[annot_col], smi_threshold)

print(f"SMI matrix shape: {smi_similarity.shape} (factor-directions x cell types)")
print(f"Factor-cell type pairs with SMI >= {smi_threshold}: {len(smi_top_matches)}")
smi_top_matches

# %% [markdown]
# ### Visualize the matches as a network

# %%
plot_packed_network(smi_top_matches, figsize=(20, 20))

# %% [markdown]
# ### Store results

# %%
store_matches(embed, smi_top_matches, suffix="user_annotations")
(
    embed.var["positive_direction_match_with_user_annotations"].dropna().unique(),
    embed.var["negative_direction_match_with_user_annotations"].dropna().unique(),
)

# %% [markdown]
# **How to read this.** Each row is a factor-direction mapped to an annotated cell type (e.g. in
# the immune atlas, a plasmacytoid-DC factor, a CD16+ monocyte factor, a plasma-cell factor).
# These matches let us label factors with the cell types we already recognize in the data.
# Unmatched factors are the ones the remaining notebooks aim to characterize.

# %% [markdown]
# ## 2. Cell types from a pre-trained model
#
# When you have no manual labels (or want finer subtypes), you can borrow labels from any model
# already trained on annotated atlases. **In this example we use
# [CellTypist](https://www.celltypist.org/)**, a logistic-regression classifier — but any
# annotation source plugs in the same way. We run it on the data, then compute SMI between its
# labels and the DRVI factors, reusing exactly the helpers from Section 1.
#
# Useful when you have **no manual labels**, or want **finer subtypes** than your labels provide.
# **Skip it if your annotations are already sufficient.**

# %% [markdown]
# ### Config

# %%
import celltypist

smi_threshold = 0.7

# CellTypist model. Run celltypist.models.models_description() to list options, and pick one
# matching your tissue/species. print(ct_model.cell_types) shows a model's cell types.
ct_model = "Immune_All_Low.pkl"  # e.g. "Developing_Mouse_Brain.pkl"

# %% [markdown]
# ### Annotate cells with CellTypist
#
# CellTypist expects normalized, log1p data over the **full** gene set, so we load the full-gene
# AnnData. `majority_voting=True` refines per-cell labels within kNN neighborhoods (reduces noise).

# %%
raw_data_path = io_dir.parent / "immune_all.h5ad"
adata_full = sc.read_h5ad(raw_data_path)

adata_full.X = adata_full.layers["counts"].copy()
sc.pp.normalize_total(adata_full, target_sum=1e4)
sc.pp.log1p(adata_full)

celltypist.models.download_models(force_update=False, model=ct_model)
ct_model = celltypist.models.Model.load(model=ct_model)

predictions = celltypist.annotate(adata_full, model=ct_model, majority_voting=True)

# %%
# Attach CellTypist outputs to the latent AnnData.
embed.obs["celltypist_labels"] = predictions.predicted_labels["predicted_labels"].loc[embed.obs.index]
embed.obs["celltypist_majority"] = predictions.predicted_labels["majority_voting"].loc[embed.obs.index]

annot_col = "celltypist_majority"

# %% [markdown]
# ### Visualize the factor x cell-type relationship

# %%
drvi.utils.pl.plot_latent_dims_in_heatmap(embed, annot_col, title_col="title", sort_by_categorical=True)

# %% [markdown]
# ### Compute SMI and keep the strong matches

# %%
embed_directional_df = build_directional_df(embed)
smi_similarity, smi_top_matches = smi_matches(embed_directional_df, embed.obs[annot_col], smi_threshold)

print(f"Factor-cell type pairs with SMI >= {smi_threshold}: {len(smi_top_matches)}")
smi_top_matches

# %% [markdown]
# ### Visualize the matches as a network

# %%
plot_packed_network(smi_top_matches, figsize=(20, 20))

# %% [markdown]
# ### Store results

# %%
store_matches(embed, smi_top_matches, suffix="celltypist")
(
    embed.var["positive_direction_match_with_celltypist"].dropna().unique(),
    embed.var["negative_direction_match_with_celltypist"].dropna().unique(),
)

# %% [markdown]
# **How to read this.** A classifier can extend or replace manual labels: it may resolve finer
# subtypes (e.g. naive vs. memory B cells, classical vs. non-classical monocytes) and can annotate
# factors that had no prior label. We suggest verifying these assignments against each factor's
# top-scoring genes. Note that, because DRVI is trained independently, errors from these
# classifiers do not leak into the DRVI factors.

# %% [markdown]
# ## 3. Save
#
# Write the annotated embedding back to disk so the other notebooks (and the curation step in the
# biological-processes notebook) can pick these results up.

# %%
import anndata as ad

ad.settings.allow_write_nullable_strings = True
embed.write_h5ad(embed_path)
print(f"Updated embedding saved to: {embed_path}")

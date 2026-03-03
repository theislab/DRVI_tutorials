# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: python_apptainer
#     language: python
#     name: python_apptainer
# ---

# # General training and interpretability pipeline

# In this notebook, we analyze the immune dataset of 9 batches from four human peripheral blood and bone marrow studies, with 16 annotated cell types. We apply DRVI with 128 latent dimensions to showcase the following:
#
# - How to train DRVI
# - Observe vanished dimensions
# - Observe the latent space in UMAP and heatmap
# - How to run the interpretability pipeline
# - How to identify and check individual dimensions

# ## Contact

# For questions and help requests, you can reach out in the [scverse discourse](https://discourse.scverse.org/).
#
# If you found a bug, please use the [issue tracker](https://github.com/theislab/drvi/issues).

# ## Install

# If you try DRVI on colab, next cell will install dependencies.
#
# Please remove this part if your environment is already setup.

# +
import sys

# if branch is stable, will install via pypi, else will install from source
branch = "latest"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB and branch == "stable":
    # !pip install drvi-py[tutorials]
elif IN_COLAB and branch != "stable":
    # !pip install git+https://github.com/theislab/drvi.git#egg=drvi-py[tutorials]
# -

# ## Imports

import warnings
warnings.filterwarnings("ignore")

# +
import anndata as ad
import scanpy as sc

import scvi
import drvi
from pathlib import Path
from drvi.model import DRVI
from drvi.utils.misc import hvg_batch
# -

print("Last run with scvi-tools version:", scvi.__version__)
print("Last run with DRVI version:", drvi.__version__)

# +
# Making plots prettier
sc.settings.set_figure_params(dpi=100, frameon=False)
sc.set_figure_params(dpi=100)
sc.set_figure_params(figsize=(3, 3))

from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (3, 3)
# -

# ## Config

# +
# Set this to false if you already trained your model and do not want to retrain.
overwrite = False
SEED = 1  # Set to None if you don't want to set seed

# Set input output directory to load data from and store model and embeddings there
# We use tmp_io/ directory in the same place as this notebook. Update accordingly.
io_dir = Path("./tmp_io/drvi_immune_128/")
io_dir.mkdir(parents=True, exist_ok=True)
io_dir
# -

# ## Download data

input_anndata_path = io_dir.parent / "immune_all.h5ad"
input_anndata_path

# +
# Run this cell only if you need to download the data
import requests

url = f"https://api.figshare.com/v2/file/download/25717328"

if input_anndata_path.exists():
    print("File already exists.")
else:
    print("Downloading ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(input_anndata_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
    print(f"Successfully downloaded: {input_anndata_path}")
# -

# ## Load Data

adata = sc.read_h5ad(input_anndata_path)
# Remove dataset with non-count values
adata = adata[adata.obs["batch"] != "Villani"].copy()
# We shuffle the data for better visualization. Otherwise order of points in UMAP will not be random.
sc.pp.subsample(adata, fraction=1.)
adata

# ## Pre-processing

adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata

# Batch aware HVG selection (method is obtained from scIB metrics)
hvg_genes = hvg_batch(adata, batch_key="batch", target_genes=2000, adataOut=False)
adata = adata[:, hvg_genes].copy()
adata


sc.pl.umap(adata, color=["batch", "final_annotation"], ncols=1, frameon=False)


# Save pre-processed data for next notebooks
if overwrite or not (io_dir / "adata_preprocesses.h5ad").exists():
    adata.write_h5ad(io_dir / "adata_preprocesses.h5ad")

# ## Train DRVI

# +
# You can also skip this cell if model is already trained

# Setup data
DRVI.setup_anndata(
    adata,
    # DRVI accepts count data by default.
    # Do not forget to change gene_likelihood if you provide a non-count data.
    layer="counts",
    batch_key="batch",
    # In addition to batch_key, you can also provide additional `categorical_covariate_keys`.
    # DRVI accepts count data by default.
    # Set to false if you provide log-normalized data and use normal distribution (mse loss).
    is_count_data=True,
)

# Setting seed (set to None if you don't want to fix seed)
scvi.settings.seed = SEED

# construct the model
model = DRVI(
    adata,
    n_latent=128,
    # For encoder and decoder dims, provide a list of integers.
    encoder_dims=[128, 128],
    decoder_dims=[128, 128],
    # depending on the variability of gene dispersions use 'gene' (default) or 'gene-batch'
    # dispersion='gene',
    # dispersion='gene-batch',
)
model

# +
# For cpu training you should add the following line to the model.train parameters:
# accelerator="cpu", devices=1,
#
# For mps acceleration on macbooks, add the following line to the model.train parameters:
# accelerator="mps", devices=1,
#
# For gpu training don't provide any additional parameter.
# More details here: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html

n_epochs = 400
model_path = io_dir / "drvi_model"

# train the model and save (if not already trained)
if overwrite or not model_path.exists():
    model.train(
        max_epochs=n_epochs,
        early_stopping=False,
        early_stopping_patience=20,
        # mps
        # accelerator="mps", devices=1,
        # cpu
        # accelerator="cpu", devices=1,
        # gpu: no additional parameter
        #
        # No need to provide `plan_kwargs` if n_epochs >= 400.
        plan_kwargs={
            "n_epochs_kl_warmup": n_epochs,
        },
    )
    
    # Save the model
    model.save(model_path, overwrite=True)

# Runtime:
# The runtime for CPU laptop (M1) is 208 minutes
# The runtime for Macbook gpu (M1) is 64 minutes
# The runtime for GPU (H100) is 10 minutes
# -

# ## Latent space

# Load the model
model = DRVI.load(model_path, adata)
model

# +
embed_path = io_dir / "embed.h5ad"

# Create latent space data in anndata format
if overwrite or not embed_path.exists():
    embed = ad.AnnData(model.get_latent_representation(), obs=adata.obs)

    # We set latent dimension stats here (see docs for more info)
    print("Setting latent dimension stats ...")
    model.set_latent_dimension_stats(embed, vanished_threshold=0.5)
    
    # We immediately calculate the interpretability gene scores with different approaches
    print("Calculating gene scores per factor ...")
    # out-of-distribution (OOD) approach uses decoder reconstructions to calculate gene scores (faster)
    model.calculate_interpretability_scores(embed, "OOD")
    # within-distribution (IND) approach iterates over all cells and calculates gene scores
    model.calculate_interpretability_scores(embed, "IND")

    print("Dimension reduction ...")
    sc.pp.neighbors(embed, n_neighbors=10, use_rep="X", n_pcs=embed.X.shape[1])
    sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
    sc.pp.pca(embed)

    print("Writing ...")
    embed.write_h5ad(embed_path)
# -

embed = sc.read_h5ad(embed_path)

sc.pl.umap(embed, color=["batch", "final_annotation"], ncols=1, frameon=False)


# ### Check latent dimension stats

# Show information for latent factors
embed.var.sort_values("reconstruction_effect", ascending=False)[:5]

drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=2)


# You can check the same plot after removing vanished dimensions

drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=2, remove_vanished=True)


# ### Plot latent dimensions

# By default, vanished dimensions are not plotted. Change arguments if you would like to.

# #### UMAP

drvi.utils.pl.plot_latent_dims_in_umap(embed)

# #### Heatmap

# Heatmaps can be useful to visualize general relationship between latent dims and known categories of data

drvi.utils.pl.plot_latent_dims_in_heatmap(embed, "final_annotation", title_col="title")

# It is possible to sort dimensions based on the top relevance with respect to a categorical variable

drvi.utils.pl.plot_latent_dims_in_heatmap(embed, "final_annotation", title_col="title", sort_by_categorical=True)


# ## Interpretability

#  The scores are already calculated and stored in embed.varm.

embed.varm

# ### Out-Of-Distribution (OOD) scores
#
# This approach iterates over latent dimensions and calculates decoder effects.

# We first visualize gene scores based on default algorithm (optionally you can pass `key="OOD_combined"`)
#
# These scores show a combination of max effect and specificity. So, this is our suggested method to consider for finding cell-types and most specific genes of a program.
#
# If human readable gene symbols are present in a different column of adata other than adata.var.index, please pass that column as `gene_symbols=...` to the function.

model.plot_interpretability_scores(embed, adata)

# You can get all scores as a dataframe:

# Note: Genes (rows of the dataframe) appear as in adata and are not sorted.
scores_df = model.get_interpretability_scores(embed, adata)
scores_df.iloc[:10, :10]

# A user can take a deeper look into individual dimensions. By visualizing the min_possible, and max_possible log-fold-changes of each dimension in OOD settings. Please refer to paper appendix for details on these scores that together form OOD_combined.
#
# ```
# scores_df = model.get_interpretability_scores(embed, adata, key="OOD_max_possible")
# scores_df = model.get_interpretability_scores(embed, adata, key="OOD_min_possible")
# ```
#
# or for visualization:
# ```
# model.plot_interpretability_scores(embed, adata, key="OOD_max_possible")
# model.plot_interpretability_scores(embed, adata, key="OOD_max_possible")
# ```

# ---
# Users can plot top relevant genes of a factor on UMAP using scanpy plotting functions:

# +
# DR 11- shows CD8+ cells and DR 27+ shows T-reg (May vary depending on the system and initialization)

# We first copy UMAP embeddings to original anndata
adata.obsm['X_drvi_umap'] = embed[adata.obs.index].obsm['X_umap']

# Show top 4 genes related to these two dimensions
for dim_title in ['DR 11-', 'DR 27+']:
    print(dim_title)
    top_genes = scores_df[dim_title].sort_values(ascending=False).index.to_list()[:4]
    drvi.utils.pl.plot_latent_dims_in_umap(embed, dim_subset=[dim_title], directional=True)
    sc.pl.embedding(adata, "X_drvi_umap", color=top_genes)
# -

# ### Within-Distribution (IND) scores
#
# This approach iterates over all cells in anndata and averages the effect of each latent factor on each gene. The scores are already stored in embed.

# These scores reflect the broad mechanistic effect of each latent dimension. Because genes are not filtered for uniqueness, shared genes retain high scores, providing a complete view of how each factor influences the genetic landscape.

model.plot_interpretability_scores(embed, adata, key="IND_linear_weighted_mean")

# You can get all scores as a dataframe:

# Note: Genes (rows of the dataframe) appear as in adata and are not sorted.
scores_df = model.get_interpretability_scores(embed, adata, key="IND_linear_weighted_mean")
scores_df.iloc[:10, :10]

# ## Identification of programs

# Once we identify the top relevant genes, we can determine some programs through supervised external information, such as:
# - existing annotations
# - examination by biologists
# - gene-set enrichment analysis (GSEA)
# - scientific literature
# - automated tools based on language models
#
# <!-- **Please refer to this tutorial for some tools that we found useful for identification of programs** -->
#
# It is worth mentioning that since such supervised information is not given to the model, the quality of the derived signatures is neither affected nor biased by it. Unidentified processes with high gene scores are promising candidates for further literature search, additional analysis, and even experimental design.





# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
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
    # !pip install multigrate[tutorials]
elif IN_COLAB and branch != "stable":
    # !pip install git+https://github.com/theislab/drvi.git#egg=drvi[tutorials]
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
# Set this to false if you already trained your model and do not like to retrain.
overwrite = False
SEED = 1  # Set to None if you don't want to set seed

# Set input output directory to load data from and store model and embeddings there
# We use tmp_io/ directory in the same place as this notebook. Update accordingly.
io_dir = Path("./tmp_io/drvi_immune_128/")
io_dir.mkdir(parents=True, exist_ok=True)
io_dir
# -

# ## Load Data

# + magic_args="-s \"$io_dir\"" language="bash"
# export io_dir=$1
#
# # Download Example Immune dataset if it does not exist
# if [ ! -f $io_dir/immune_all.h5ad ]; then
#   curl -L https://figshare.com/ndownloader/files/25717328 -o $io_dir/immune_all.h5ad
#   echo "File downloaded successfully."
# else
#   echo "File already exists."
# fi
# -

adata = sc.read_h5ad(io_dir / "immune_all.h5ad")
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


# Save pre-processes data for next notebooks
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
    # Always provide a list. DRVI can accept multiple covariates.
    categorical_covariate_keys=["batch"],
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

    # We set latent dimension stats here (see drvi.utils.pl.plot_latent_dimension_stats in next cells)
    drvi.utils.tl.set_latent_dimension_stats(model, embed, vanished_threshold=0.1)
    
    sc.pp.neighbors(embed, n_neighbors=10, use_rep="X", n_pcs=embed.X.shape[1])
    sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
    sc.pp.pca(embed)
    
    embed.write_h5ad(embed_path)
# -

embed = sc.read_h5ad(embed_path)

sc.pl.umap(embed, color=["batch", "final_annotation"], ncols=1, frameon=False)


# ### Chack latent dimension stats

# Show information for latnet factors
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

# It is possible to sort dimensions based on the top relevance with respect to a categoricals variable

drvi.utils.pl.plot_latent_dims_in_heatmap(embed, "final_annotation", title_col="title", sort_by_categorical=True)


# ## Interpretability

# ### Traversing the latent space

# Here we use DRVI's utils to traverse latent space and find the effect of each latent dimension

# +
traverse_adata_path = io_dir / "traverse_adata.h5ad"

if overwrite or not traverse_adata_path.exists():
    traverse_adata = drvi.utils.tl.traverse_latent(model, embed, n_samples=20, max_noise_std=0.0)
    drvi.utils.tl.calculate_differential_vars(traverse_adata)
    traverse_adata.write_h5ad(traverse_adata_path)
# -

traverse_adata = sc.read_h5ad(traverse_adata_path)
traverse_adata

# ### Getting the results

# We can then visualize the top relevant genes for each dimension

drvi.utils.pl.show_top_differential_vars(traverse_adata, key="combined_score", score_threshold=0.0)




# +
dimensions_interpretability = drvi.utils.tools.iterate_on_top_differential_vars(
    traverse_adata, key="combined_score", score_threshold=0.0
)

# For making it brief we just iterate over 5 dimensions
for dim_title, gene_scores in dimensions_interpretability[:5]:
    print(dim_title)
    # We just print top 20
    print(gene_scores[:20])
# -

# ### Looking into individual dimensions

# A user can take a deeper look into individual dimensions. we can see the min_possible, and max_possible log-fold-changes of each dimension. In addition, one can check the activity of top relevant genes for dimensions of interest.

# We visualize 3 dimensions:
# 1. DR 13- highlighting CD8
# 2. DR 35+ highlighting Technical stress response
# 3. DR 28- highlighting T-reg cells
drvi.utils.pl.show_differential_vars_scatter_plot(
    traverse_adata,
    key_x="max_possible",
    key_y="min_possible",
    key_combined="combined_score",
    dim_subset=["DR 11-", "DR 30+", "DR 40+"],
    score_threshold=0.0,
)


for fig in drvi.utils.pl.plot_relevant_genes_on_umap(
    adata,
    embed,
    traverse_adata,
    traverse_adata_key="combined_score",
    dim_subset=["DR 11-", "DR 30+", "DR 40+"],
    score_threshold=0.0,
):
    plt.show()



# ## Identification of programs

# Once we identify the top relevant genes, we can determine some programs through supervised external information, such as:
# - existing annotations
# - examination by biologists
# - gene-set enrichment analysis (GSEA)
# - scientific literature
# - automated tools based on language models
#
# **Please refer to this tutorial for some tools that we found useful for identification of programs**
#
# It is worth mentioning that since such supervised information is not given to the model, the quality of the derived signatures is neither affected nor biased by it. Unidentified processes with high gene scores are promising candidates for further literature search, additional analysis, and even experimental design.





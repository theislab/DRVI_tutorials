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

# # Transfer learning to new dataset (query to reference mapping)

# In this notebook, we train a DRVI model on reference data and transfer processes and embeddings to query data. DRVI uses the scArches approach internally. This covers:
#
# - Training DRVI
# - Query to reference mapping
# - Observe the integrated latent space in UMAP
# - Observe transferred factors
#
# **IMPORTANT:** Set `encode_covariates=True` when initializing the model (not default). This ensures the model uses batch information in the encoder, which is essential for query to reference mapping.

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
io_dir = Path("./tmp_io/drvi_immune_128_q2r/")
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

adata.obs['study'].unique()

# Holding out an unseen dataset as query
adata_ref = adata[adata.obs['study'] != 'Sun'].copy()
adata_query = adata[adata.obs['study'] == 'Sun'].copy()

# Batch aware HVG selection (method is obtained from scIB metrics)
hvg_genes = hvg_batch(adata_ref, batch_key="batch", target_genes=2000, adataOut=False)
adata_ref = adata_ref[:, hvg_genes].copy()
adata_query = adata_query[:, hvg_genes].copy()
adata_ref, adata_query


# Save pre-processed data for next notebooks
if overwrite or not (io_dir / "adata_preprocesses_ref.h5ad").exists():
    adata_ref.write_h5ad(io_dir / "adata_preprocesses_ref.h5ad")
if overwrite or not (io_dir / "adata_preprocesses_query.h5ad").exists():
    adata_query.write_h5ad(io_dir / "adata_preprocesses_query.h5ad")

del adata

# ## Train DRVI

# +
# You can also skip this cell if model is already trained
# For more details on training params please refer to the general pipeline notebook

# Setup data
DRVI.setup_anndata(
    adata_ref,
    layer="counts",
    batch_key="batch",
    # In addition to batch_key, you can also provide additional `categorical_covariate_keys`.
    # DRVI supports query to reference mapping with new categorical covariates.
    is_count_data=True,
)

# Setting seed (set to None if you don't want to fix seed)
scvi.settings.seed = SEED

# construct the model
model = DRVI(
    adata_ref,
    n_latent=128,
    # IMPORTANT: you need to allow encoder to get covariates as input to be able to do query to reference mapping
    encode_covariates=True,
    # For encoder and decoder dims, provide a list of integers.
)
model

# +
n_epochs = 400
model_path = io_dir / "drvi_model"

# train the model and save (if not already trained)
if overwrite or not model_path.exists():
    model.train(
        max_epochs=n_epochs,
        early_stopping=False,
        early_stopping_patience=20,
        # No need to provide `plan_kwargs` if n_epochs >= 400.
        plan_kwargs={
            "n_epochs_kl_warmup": n_epochs,
        },
    )
    
    # Save the model
    model.save(model_path, overwrite=True)
# -



# ## Get reference embeddings

# Load the model
model = DRVI.load(model_path, adata_ref)
model

embed_ref = ad.AnnData(model.get_latent_representation(adata_ref), obs=adata_ref.obs)
model.set_latent_dimension_stats(embed_ref, vanished_threshold=0.5)
embed_ref.write_h5ad(io_dir / "embed_ref.h5ad")

# If you are interested to observe your latent space and latent factors of the reference model, please have a look at the main tutorial.



# ## Transfer learning

# +
model_path = io_dir / "drvi_model_transfer"
if overwrite or not model_path.exists():
    drvi.model.DRVI.prepare_query_anndata(adata_query, model)
    model_transfer = drvi.model.DRVI.load_query_data(adata_query, model)
    
    # It is important to pass plan_kwargs={"weight_decay": 0.0} to make sure reference embeddings are untouched
    model_transfer.train(
        max_epochs=100, 
        plan_kwargs={
            "weight_decay": 0.0,
        }
    )
    model_transfer.save(model_path, overwrite=True)

model_transfer = DRVI.load(model_path, adata_query)
# -



# ## Latent space

# +
embed_path = io_dir / "embed.h5ad"

# Create latent space data in anndata format
if overwrite or not embed_path.exists():
    embed_ref = sc.read_h5ad(io_dir / "embed_ref.h5ad")  # Reference
    embed_query = ad.AnnData(model_transfer.get_latent_representation(), obs=adata_query.obs)  # Query

    # Combining the two
    embed_ref.obs['split'] = 'reference'
    embed_query.obs['split'] = 'query'
    embed = ad.concat([embed_ref, embed_query], join='outer')
    embed.var = embed_ref.var.copy()

    sc.pp.neighbors(embed, n_neighbors=10, use_rep="X", n_pcs=embed.X.shape[1])
    sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
    sc.pp.pca(embed)
    
    embed.write_h5ad(embed_path)
embed
# -

embed = sc.read_h5ad(embed_path)

sc.pl.umap(embed, color=["batch", "split", "final_annotation"], ncols=1, frameon=False)


sc.pl.umap(embed, mask_obs=embed.obs['split']=='query', color=["final_annotation"], ncols=1, frameon=False)

# ### Plot latent dimensions

# By default, vanished dimensions are not plotted. Change arguments if you would like to.

# #### UMAP of factors for all cells

drvi.utils.pl.plot_latent_dims_in_umap(embed)

# #### UMAP of factors for query cells

drvi.utils.pl.plot_latent_dims_in_umap(
    embed,
    mask_obs=embed.obs['split']=='query',
)

# #### Heatmaps

drvi.utils.pl.plot_latent_dims_in_heatmap(embed_ref, "final_annotation", title_col="title")

embed_query.var = embed_ref.var.copy()
drvi.utils.pl.plot_latent_dims_in_heatmap(embed_query, "final_annotation", title_col="title")



# ## Notes
# For more details such as interpretability of latent factors please refer to other tutorials. 



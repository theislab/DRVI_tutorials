# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Porting a DRVI model (drvi-py < 0.3) to scvi.external.DRVI (scvi-tools)
#
# DRVI's PyTorch model has moved into scvi-tools as `scvi.external.DRVI`. A model that was
# **trained and saved with drvi-py (< 0.3)** can be migrated to the scvi-tools layout with
# `drvi.utils.port_to_scvi_tools` and then loaded via `scvi.external.DRVI.load`.
#
# The port is pure checkpoint surgery (no need to load the actual model). This minimal notebook trains
# a tiny DRVI model with drvi-py, ports it, loads it into scvi-tools, and shows that both produce
# **exactly the same latent representation**.
#
# Requires: `drvi-py >= 0.2.7` (which provides `port_to_scvi_tools`) and a scvi-tools that ships
# `scvi.external.DRVI` (>= 1.5.0).

# %%
import os
import tempfile

import numpy as np
import pandas as pd
import anndata as ad

import drvi
import scvi

print("drvi", drvi.__version__)
print("scvi", scvi.__version__)

save_dir = tempfile.TemporaryDirectory()

# %% [markdown]
# ## Synthetic count data

# %%
rng = np.random.default_rng(0)
n_cells, n_genes = 300, 60
X = rng.poisson(1.0, size=(n_cells, n_genes)).astype(np.float32)

adata = ad.AnnData(
    X=X,
    obs=pd.DataFrame(
        {"batch": [f"b{i % 2}" for i in range(n_cells)]},
        index=[f"cell_{i}" for i in range(n_cells)],
    ),
    var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
)
adata.layers["counts"] = X.copy()
adata

# %% [markdown]
# ## Train and save a DRVI model (drvi-py)

# %%
drvi.model.DRVI.setup_anndata(adata, layer="counts", batch_key="batch")
model = drvi.model.DRVI(adata, n_latent=16)
model.train(max_epochs=3)

# Latent representation from the original drvi-py model
z_drvi = model.get_latent_representation()

model_path = os.path.join(save_dir.name, "drvi_model")
model.save(model_path, overwrite=True)
z_drvi.shape

# %% [markdown]
# ## Port the checkpoint to scvi-tools
#
# `port_to_scvi_tools` reads the saved `model.pt`, rewrites it to the scvi-tools DRVI layout, and
# writes a new model directory (default: `"<source>_scvi_tools"`, i.e. next to the source inside our
# temporary directory).

# %%
ported_dir = os.path.join(save_dir.name, "drvi_model_scvi_tools")
drvi.utils.port_to_scvi_tools(model_path, ported_dir, overwrite=True)
ported_dir

# %% [markdown]
# ## Load with `scvi.external.DRVI`
#
# The port does not touch the data, so we load the ported model against the *same* `adata`
# (matching `var_names` and the covariate columns used at setup time).

# %%
scvi_model = scvi.external.DRVI.load(ported_dir, adata)
z_scvi = scvi_model.get_latent_representation()
z_scvi.shape

# %% [markdown]
# ## The latents are identical

# %%
max_abs_diff = np.abs(z_drvi - z_scvi).max()
print("max absolute difference:", max_abs_diff)
print("exactly equal:", np.array_equal(z_drvi, z_scvi))
assert np.allclose(z_drvi, z_scvi, atol=1e-6)
print("OK - drvi-py and scvi.external.DRVI produce the same latent representation.")

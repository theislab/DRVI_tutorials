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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Identifying biological processes of DRVI factors
#
# Factors that do not map cleanly to a single cell type often capture **biological processes**
# (interferon response, cell cycle, stress, ...). This notebook annotates them with three
# complementary, non-LLM enrichment tools that all consume DRVI's per-gene interpretability
# scores:
#
# | Tool | Method | Input | Strengths |
# |------|--------|-------|-----------|
# | **Enrichr** (via GSEApy) | Over-representation (ORA)     | Top-N gene list      | Fast; huge library collection |
# | **g:Profiler**           | Over-representation (ORA)     | Ordered gene query   | g:SCS correction for hierarchical GO terms |
# | **decoupler**            | Activity inference (ULM/MLM)  | Gene-score matrix + prior network | Reports *regulators* (TFs), not just terms |
#
# It ends with an **appendix** of databases you can swap in. A separate
# [factor-curation notebook](./identification_of_factors_4_curation.html) then pulls every tool's
# stored output (including the cell-type and LLM notebooks, if you ran them) into one per-factor
# view for final labelling.
#
# Note: Enrichment usually returns **broad, high-level** processes (e.g. "defense
# response to virus", "cell cycle"). For a **finer** reading — the specific pathway branch or the
# program driving a factor — align the factor's top genes with the primary literature, or use the
# [LLM-based tools](./identification_of_factors_3_llm_tools.html) to *suggest* candidate
# processes and then verify those suggestions against the literature. LLM output is a hypothesis,
# not evidence.
#
# > This is one of four companion notebooks. See
# > [cell types from annotations](./identification_of_factors_1_cell_types.html),
# > [LLM-based tools](./identification_of_factors_3_llm_tools.html), and
# > [factor curation](./identification_of_factors_4_curation.html). They share `embed.h5ad`, so
# > run the cell-type notebook first so its labels feed the curation step.
#
# > **A note on pre-ranked GSEA.** GSEA-style methods expect signed, two-sided rankings (top =
# > up, bottom = down). DRVI interpretability scores are non-negative by construction (each
# > factor-direction has its own positive ranking), so there is no meaningful "bottom" for GSEA
# > to exploit. In practice pre-ranked GSEA returned very few, no-more-informative hits here, so
# > we restrict this notebook to ORA (Enrichr, g:Profiler) plus TF-activity inference (decoupler).
#
# **All results are guiding — interpret in context and validate against known biology.**

# %% [markdown]
# ## Prerequisites
#
# Assumes a trained DRVI model with interpretability scores (see the
# [general pipeline](./general_pipeline.html)).
#
# **Adapting to your own model** — change `io_dir` (Section 0) and, per tool, the database/organism
# config: `gseapy_db` (Section 1), `organism` + `gp_source` (Section 2), and `dc_geneset` /
# `dc_organism` (Section 3). See the [appendix](#6.-Appendix:-database-reference) for options.

# %% [markdown]
# ## Contact
#
# Questions: [scverse discourse](https://discourse.scverse.org/). Bugs:
# [issue tracker](https://github.com/theislab/drvi/issues).

# %% [markdown]
# ## Install
#
# This notebook uses the **`tutorials-biological-processes`** extra of `drvi-py`, which adds
# GSEApy, g:Profiler, decoupler, and statsmodels on top of DRVI. Install it once in your
# environment with:
#
# ```bash
# pip install "drvi-py[tutorials-biological-processes]"
# ```
#
# On Colab, the next cell does this for you. Remove it if your environment is already set up.

# %%
import sys
import subprocess

branch = "latest"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB and branch == "stable":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "drvi-py[tutorials-biological-processes]"])
elif IN_COLAB and branch != "stable":
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "git+https://github.com/theislab/drvi.git#egg=drvi-py[tutorials-biological-processes]"])

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
from pathlib import Path

import scvi
import drvi
from drvi.model import DRVI

# %%
print("Last run with scvi-tools version:", scvi.__version__)
print("Last run with DRVI version:", drvi.__version__)

# %%
sc.set_figure_params(dpi=100, frameon=False, figsize=(3, 3))
plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## 0. Setup

# %% [markdown]
# ### Config

# %%
# Input/output directory holding the trained model and embeddings. Update accordingly.
io_dir = Path("./tmp_io/drvi_immune_128/").resolve()
print(f"Using directory: {io_dir}")

# DRVI provides two per-gene scoring approaches (both precomputed by the general pipeline):
#   "OOD_combined"            — decoder-based, favors specific genes (best for identity/top genes)
#   "IND_linear_weighted_mean"— per-cell mean effect, keeps broadly shared genes
# This is the shared scoring choice for every tool below.
score_key = "OOD_combined"

# %% [markdown]
# ### Load model and embeddings

# %%
adata = sc.read_h5ad(io_dir / "adata_preprocesses.h5ad")
model = DRVI.load(io_dir / "drvi_model", adata)

embed_path = io_dir / "embed.h5ad"
embed = sc.read_h5ad(embed_path)

# Gene x factor-direction score matrix used by all three tools.
scores_df = model.get_interpretability_scores(embed, adata, key=score_key)
gene_background = adata.var_names.tolist()

scores_df.iloc[:3, :3]


# %% [markdown]
# ### Shared helper
#
# Every tool starts from the same operation: for a factor-direction, take the top genes above a
# score cutoff.

# %%
def top_genes(scores_df, col, cutoff, top_n):
    """Genes for a factor-direction with score >= cutoff, at most `top_n`, ranked descending."""
    s = scores_df[col]
    return s[s >= cutoff].nlargest(top_n).index.astype(str).tolist()


def factor_first(df):
    """Return `df` with the `factor` column moved to the front (no-op if empty/absent)."""
    if df.empty or "factor" not in df.columns:
        return df
    return df[["factor", *df.columns.drop("factor")]]


# %% [markdown]
# ## 1. Enrichr (via GSEApy)
#
# [GSEApy](https://github.com/zqfang/GSEApy)'s Enrichr ORA tests whether a factor's top-N gene
# list is enriched for terms in a chosen library, against a gene background.

# %%
import gseapy

# %% [markdown]
# ### Config

# %%
gseapy_db = "GO_Biological_Process_2023"   # any Enrichr library (see appendix)
gseapy_cutoff = 0.1                         # ~0.1 for OOD, ~0.5 for IND
gseapy_top_n = 100
padj_threshold = 0.05                       # adjusted p-value


# %% [markdown]
# ### Run enrichment

# %%
def run_gseapy_enrichr(scores_df, gene_sets, cutoff, top_n, padj_cutoff, background):
    rows = []
    for col in scores_df.columns:
        genes = top_genes(scores_df, col, cutoff, top_n)
        if not genes:
            continue
        try:
            enr = gseapy.enrich(gene_list=genes, gene_sets=gene_sets, background=background,
                                no_plot=True, outdir=None)
        except Exception as e:
            print(f"ORA failed for {col}: {e}")
            continue
        hits = enr.results[enr.results["Adjusted P-value"] < padj_cutoff].sort_values("Adjusted P-value")
        rows.append(hits.assign(factor=col))
    return factor_first(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()


gseapy_enrichr_results = run_gseapy_enrichr(
    scores_df, gseapy_db, gseapy_cutoff, gseapy_top_n, padj_threshold, gene_background
)
display(gseapy_enrichr_results.head())

# %%
# Let's look at the top enriched term for each factor-direction.
display(gseapy_enrichr_results.drop_duplicates(subset=["factor"], keep="first"))

# %% [markdown]
# ### Store results

# %%
embed.uns["gseapy_enrichr_results"] = gseapy_enrichr_results.convert_dtypes(
    convert_integer=False, convert_floating=False
)
n_sig = gseapy_enrichr_results["factor"].nunique() if not gseapy_enrichr_results.empty else 0
print(f"Enrichr significant factor-directions: {n_sig} / {scores_df.shape[1]}")

# %% [markdown]
# **How to read this.** ORA works well when the factor's biology is well represented in the chosen
# library: hits for a lineage-aligned factor are usually specific and internally consistent (e.g.
# a B-cell factor returning Ig-mediated immune response, BCR signaling). Quality is bounded by the
# database, so factors whose biology is under-represented return loosely related terms. Interpret
# alongside g:Profiler below — convergent terms across the two ORA tools are better supported.

# %% [markdown]
# ## 2. g:Profiler
#
# [g:Profiler](https://biit.cs.ut.ee/gprofiler/) runs ORA with g:SCS multiple-testing correction,
# designed for hierarchical GO terms. In **ordered-query** mode it walks the ranked gene list to
# find the best-enriched prefix, which suits continuous factor scores better than a fixed top-N.

# %%
from gprofiler import GProfiler

# %% [markdown]
# ### Config

# %%
organism = "hsapiens"        # e.g. "mmusculus", "drerio"
gp_source = ["GO:BP"]        # e.g. ["GO:MF"], ["REAC"], ["KEGG"], ["HP"]
gp_cutoff = 0.1              # ~0.1 for OOD, ~0.5 for IND
gp_top_n = 100               # max top genes per factor-direction
pval_threshold = 0.05


# %% [markdown]
# ### Run enrichment

# %%
def run_gprofiler(scores_df, background, organism, sources, pval_threshold, cutoff, top_n):
    gp = GProfiler(return_dataframe=True)
    rows = []
    for col in scores_df.columns:
        genes = top_genes(scores_df, col, cutoff, top_n)
        if not genes:
            continue
        res = gp.profile(organism=organism, query=genes, sources=sources, ordered=True,
                         user_threshold=pval_threshold, background=background)
        if res is None or res.empty:
            continue
        rows.append(res.assign(factor=col))
    return factor_first(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()


gprofiler_results = run_gprofiler(
    scores_df, gene_background, organism, gp_source, pval_threshold, gp_cutoff, gp_top_n
)
if not gprofiler_results.empty:
    gprofiler_results["parents"] = gprofiler_results["parents"].astype(str)

# %%
n_sig = gprofiler_results["factor"].nunique() if not gprofiler_results.empty else 0
print(f"g:Profiler significant factor-directions: {n_sig} / {scores_df.shape[1]}")
gprofiler_results.sort_values(["factor", "p_value"]).head() if not gprofiler_results.empty else gprofiler_results

# %%
# Let's look at the top enriched term for each factor-direction.
if not gprofiler_results.empty:
    display(gprofiler_results.drop_duplicates(subset=["factor"], keep="first"))

# %% [markdown]
# ### Store results

# %%
embed.uns["gprofiler_results"] = gprofiler_results.convert_dtypes(
    convert_integer=False, convert_floating=False
)

# %% [markdown]
# **How to read this.** Top hits for a factor often organize general-to-specific, and g:Profiler
# can surface more mechanistic terms than top-N ORA (e.g. UPR/ERAD components on a plasma-cell
# factor where Enrichr returns generic B-cell terms). Coverage is typically lower than Enrichr;
# read the two together and trust convergent calls more than single-tool ones.

# %% [markdown]
# ## 3. decoupler
#
# [decoupler](https://decoupler-py.readthedocs.io/) infers the **activity of regulators** from
# gene-level scores against a prior-knowledge network, rather than testing gene-set overlap.
# A significant hit is a transcription factor whose targets are co-enriched in the factor's top
# genes — a mechanistic layer the ORA/LLM tools cannot provide. Curated networks from
# [OmniPath](https://omnipathdb.org/):
#
# - **CollecTRI** — comprehensive TF → target regulons (recommended for TF drivers).
# - **DoRothEA** — TF regulons with confidence tiers A–D (tunable stringency).
# - **PROGENy** — pathway footprints (Hypoxia, EGFR, TGFb, ...); exploratory for signaling.

# %%
import decoupler as dc
from statsmodels.stats.multitest import multipletests

# %% [markdown]
# ### Config

# %%
dc_geneset = "collectri"        # or "dorothea", "progeny"
dc_organism = "human"           # match the data species
dc_cutoff = 0.01                # ~0.01 for OOD, ~0.05 for IND; scores below this are zeroed
fdr_threshold = 0.05

dc_methods = ["ulm", "mlm"]
dc_min = 10                     # min genes of a regulon present in the data for a valid test
dorothea_levels = ["A", "B", "C"]
fdr_method = "fdr_bh"

# %% [markdown]
# ### Load the regulatory network

# %%
net_dispatch = {
    "collectri": lambda: dc.op.collectri(organism=dc_organism),
    "dorothea":  lambda: dc.op.dorothea(organism=dc_organism, levels=dorothea_levels),
    "progeny":   lambda: dc.op.progeny(organism=dc_organism),
}
net = net_dispatch.get(
    dc_geneset.strip().lower(),
    lambda: dc.op.resource(name=dc_geneset, organism=dc_organism),
)()

cols = ["source", "target"] + (["weight"] if "weight" in net.columns else [])
net = net[cols].dropna().drop_duplicates().reset_index(drop=True)
print(f"Network: {len(net)} interactions, {net['source'].nunique()} regulators")


# %% [markdown]
# ### Run

# %%
def run_decouple(factors_by_genes, net, methods, tmin, fdr_method):
    mat = factors_by_genes.copy()
    mat.columns = mat.columns.astype(str).str.upper()

    net_u = net.copy()
    net_u["target"] = net_u["target"].astype(str).str.upper()

    keep = mat.columns.intersection(net_u["target"].unique())
    mat = mat[keep].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    res = dc.mt.decouple(data=mat, net=net_u, methods=methods, cons=False, tmin=tmin, verbose=True)
    _, pvals = dc.mt.consensus(res)

    out = pvals.stack().reset_index(name="p_value").rename(columns={"level_0": "factor", "level_1": "term"})
    out["p_adj"] = multipletests(out["p_value"].values, method=fdr_method)[1]
    return out


input_df = scores_df.copy()
input_df[input_df < dc_cutoff] = 0
decoupler_all = run_decouple(input_df.T, net, dc_methods, dc_min, fdr_method)

# %%
# Keep the most significant regulator per factor-direction for a summary view.
decoupler_results = (
    decoupler_all[decoupler_all["p_adj"] < fdr_threshold]
    .sort_values("p_adj")
    .groupby("factor", as_index=False)
    .first()
)
print(f"Significant regulators for {decoupler_results['factor'].nunique()} / {scores_df.shape[1]} directions.")
display(decoupler_results.sort_values("p_adj"))

# %% [markdown]
# ### Store results

# %%
embed.uns["decoupler_results"] = decoupler_results.convert_dtypes(
    convert_integer=False, convert_floating=False
)

# %% [markdown]
# **How to read this.** Where decoupler reports a significant TF, it often lines up with the other
# tools (e.g. EBF1 on a B-cell-precursor factor, PRDM1/Blimp-1 on a plasma-cell factor, RFXAP
# alongside "MHC class II assembly"). Coverage is limited by design — many factors are not
# dominated by a single TF — and some hits are not obviously related to the factor's identity, so
# treat regulators as candidate drivers to cross-check, not conclusions.

# %% [markdown]
# ## 4. Manual Exploration
#
# Not every factor is a cell type. Some capture a process shared across many different cell types.
# DR 43− is a good example: an **antiviral / interferon response**. Its top genes are canonical
# interferon-stimulated genes (IFIT1/2/3, ISG15, MX1, CXCL10); all three tools converge on
# virus/interferon terms and point to the interferon regulator IRF9.

# %%
process_factor = "DR 43-"   # a program shared across cell types (edit to explore another factor)


def show_factor_evidence(factor_label, cutoff=0.1, n_genes=12):
    genes = top_genes(scores_df, factor_label, cutoff, n_genes)
    print(f"{factor_label} — top genes:\n  {', '.join(genes)}\n")
    for name, df, col in [
        ("Enrichr", gseapy_enrichr_results, "Term"),
        ("g:Profiler", gprofiler_results, "name"),
        ("decoupler TF", decoupler_results, "term"),
    ]:
        terms = df.loc[df["factor"] == factor_label, col].head(5).tolist() if not df.empty else []
        print(f"{name}: {terms or '(no hit)'}")


show_factor_evidence(process_factor)

# %% [markdown]
# ## 5. Save
#
# Persist the enrichment results so the curation view (and the other notebooks) can read them.

# %%
ad.settings.allow_write_nullable_strings = True
embed.write_h5ad(embed_path)
print(f"Updated embedding saved to: {embed_path}")

# %% [markdown]
# ## 6. Appendix: database reference
#
# Swap the tool-specific config variables above (`gseapy_db`, `gp_source`, `dc_geneset`) to use
# different databases. The small tables below list common choices.
#
# For a much larger, curated catalog, this repo bundles
# [`resources/gene_set_libraries_master_v0_8_FINAL.xlsx`](./resources/gene_set_libraries_master_v0_8_FINAL.xlsx)
# — a registry of gene-set libraries with, per library, its authoritative source, license,
# curation-quality rating, and flags for which tool can consume it
# (`usable_in_gseapy` / `usable_in_gprofiler` / `usable_in_decoupler`). It also includes
# recommended panels for this exact workflow (e.g. `Core_Biology_DRVI`, `Cell_Type_Annotation`,
# `Regulatory_Activity`). Load it with `pd.read_excel(..., sheet_name="libraries_master")` and
# filter to the tool you are using.
#
# ### Biological process databases
#
# | Database | Description | Enrichr library | g:Profiler |
# |----------|-------------|-----------------|------------|
# | MSigDB Hallmark | 50 curated, non-redundant states | `MSigDB_Hallmark_2020` | — |
# | GO Biological Process | Hierarchical processes | `GO_Biological_Process_2025` | `GO:BP` |
# | GO Cellular Component | Subcellular localization | `GO_Cellular_Component_2025` | `GO:CC` |
# | GO Molecular Function | Molecular activities | `GO_Molecular_Function_2025` | `GO:MF` |
# | Reactome | Reaction-based pathways | `Reactome_Pathways_2024` | `REAC` |
# | KEGG | Metabolic/signaling maps | `KEGG_2026` | `KEGG` |
# | WikiPathways | Community-curated maps | `WikiPathways_2024_Human` | `WP` |
#
# ### Cell-type marker databases (Enrichr)
#
# | Database | Description | Enrichr library |
# |----------|-------------|-----------------|
# | CellMarker 2.0 | Curated markers from literature | `CellMarker_2024` |
# | PanglaoDB | Curated scRNA-seq markers | `PanglaoDB_Augmented_2021` |
# | Tabula Sapiens | Human single-cell atlas markers | `Tabula_Sapiens` |
# | Human Gene Atlas | Tissue/cell-type expression | `Human_Gene_Atlas` |
#
# ### Regulatory networks (decoupler)
#
# | Network | Description | decoupler name | Notes |
# |---------|-------------|----------------|-------|
# | CollecTRI | TF → target regulons | `collectri` | Recommended for TF drivers |
# | DoRothEA | TF → target, confidence A–D | `dorothea` | Tunable stringency |
# | PROGENy | Pathway-responsive signatures | `progeny` | Best for signaling activity |
#
# ### Clinical / disease phenotypes
#
# | Database | Description | Enrichr | g:Profiler |
# |----------|-------------|---------|------------|
# | Human Phenotype Ontology | Genes linked to clinical signs | `Human_Phenotype_Ontology` | `HP` |
# | OMIM | Human genes and genetic disorders | `OMIM_Disease` | `OMIM` |

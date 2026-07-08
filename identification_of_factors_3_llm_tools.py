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
# # Identifying DRVI factors with LLM tools
#
# LLM-based annotators take a factor's top marker genes and return a cell-type or
# biological-process label in natural language without requiring a reference atlas.
# In practice, they can label factors the annotation- and enrichment-based tools
# leave empty — which makes them useful when neither of the previous steps identifies a factor.
# This notebook covers:
#
# 1. **Direct LLM annotation** — a single, well-structured prompt you control, runnable against any
#    backend (**Ollama**, **Claude API**, **Claude Code** (no API key), **OpenAI**, or **Gemini**).
#    You own the prompt and can inspect exactly what the model was asked.
# 2. **CASSIA** — a multi-agent annotator (chain-of-thought → validation loop → structured output)
#    with a quality score.
# 3. **gs2txt** — free-text process summaries that run pathway enrichment first, then one LLM call.
#
# > This is one of four companion notebooks. See
# > [cell types from annotations](./identification_of_factors_1_cell_types.html),
# > [biological processes via enrichment](./identification_of_factors_2_biological_processes.html),
# > and [factor curation](./identification_of_factors_4_curation.html).
# > All share `embed.h5ad`; the curation notebook picks up the results stored here.
#
# > **Note:** LLM output is fluent but produced *without an uncertainty signal* and can
# > be confidently wrong, so always cross-check it against the SMI and enrichment tools and against
# > the literature.
#
# Install the packages for your chosen backend via the install cell below.

# %% [markdown]
# ## Prerequisites
#
# Assumes a trained DRVI model with interpretability scores (see the
# [general pipeline](./general_pipeline.html)).
#
# **Adapting to your own model** — change `io_dir` (Section 0), pick `LLM_BACKEND`, set that
# backend's model and credentials, and set `llm_tissue_context` / `llm_species`.

# %% [markdown]
# ## Contact
#
# Questions: [scverse discourse](https://discourse.scverse.org/). Bugs:
# [issue tracker](https://github.com/theislab/drvi/issues).

# %% [markdown]
# ## Install
#
# Install only what your chosen backend needs (none of these are in `requirements.txt`):
#
# - **Ollama** or **OpenAI**: `pip install openai`
# - **Claude API**: `pip install anthropic`
# - **Claude Code** (no API key — uses your Claude Code login): `pip install claude-agent-sdk`;
#   also needs the `claude` CLI installed and logged in (`claude login`), and in Jupyter
#   `pip install nest-asyncio`.
# - **Gemini**: `pip install google-genai`
# - **CASSIA** (Section 2): `pip install CASSIA`
# - **gs2txt** (Section 3): `pip install "gs2txt[enrichment]"`

# %%
import sys
import subprocess

# Uncomment the line(s) for the backend/tools you want to use:
# subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])            # Ollama / OpenAI
# subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])         # Claude API
# subprocess.check_call([sys.executable, "-m", "pip", "install", "claude-agent-sdk", "nest-asyncio"])  # Claude Code
# subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])      # Gemini
# subprocess.check_call([sys.executable, "-m", "pip", "install", "CASSIA"])            # Section 2
# subprocess.check_call([sys.executable, "-m", "pip", "install", "gs2txt[enrichment]"])  # Section 3

# %% [markdown]
# ## Imports

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import json
import re
import asyncio
import pandas as pd
from pathlib import Path

from drvi.model import DRVI
import scanpy as sc

# %% [markdown]
# ## 0. Setup

# %% [markdown]
# ### Config

# %%
# Input/output directory holding the trained model and embeddings. Update accordingly.
io_dir = Path("./tmp_io/drvi_immune_128/").resolve()

# DRVI provides two complementary per-gene score matrices (both precomputed by the general pipeline):
#   OOD ("OOD_combined")             — SPECIFIC genes: highlights genes that uniquely mark a program;
#                                      genes shared across many programs are penalized.
#   IND ("IND_linear_weighted_mean") — DIRECT effect: the latent factor's effect on each gene, similar
#                                      to a log fold-change, so it also keeps differential genes that
#                                      are SHARED between programs.
score_key = "OOD_combined"                   # OOD (specific) — also used by CASSIA and gs2txt below
score_key_ind = "IND_linear_weighted_mean"   # IND (direct effect, logFC-like)

# Top genes sent to the LLM per score type, plus a cutoff for each score.
llm_top_n_genes = 100
drvi_score_cutoff = 0.5     # OOD cutoff (specific genes)
ind_score_cutoff = 0.5      # IND cutoff (direct-effect genes)

# Biological context passed to every tool.
llm_tissue_context = "human immune cells (PBMC / bone marrow)"
llm_species = "human"  # or "mouse"

# How many informative factor-directions to annotate. Set to an int for a quick/cheap smoke test
# (annotates the first N); None = annotate all. Applies to every section below.
# NOTE: the example outputs saved in this notebook were produced with the 8-direction sample below.
max_directions = None

# %% [markdown]
# ### Load model and embeddings

# %%
adata = sc.read_h5ad(io_dir / "adata_preprocesses.h5ad")
model = DRVI.load(io_dir / "drvi_model", adata)

embed_path = io_dir / "embed.h5ad"
embed = sc.read_h5ad(embed_path)

# Per-gene score matrices. scores_df (OOD, specific) is used by every tool; the direct-LLM section
# below also uses ind_scores (IND, direct effect) so the model sees both specific and shared genes.
scores_df = model.get_interpretability_scores(embed, adata, key=score_key)
ind_scores = model.get_interpretability_scores(embed, adata, key=score_key_ind)

# %% [markdown]
# ## 1. Direct LLM annotation
#
# Instead of relying on a wrapper package's hidden prompt, we send our **own** structured prompt
# and choose the backend. For each factor-direction the model receives **both** DRVI score views —
# the OOD *specific* genes and the IND *direct-effect* genes — with an explanation of what each
# means, the tissue context, is asked to reason, and returns a small JSON object (`cell_type`,
# `biological_process`, `key_genes`, `confidence`, `reasoning`) that we parse and store. Because you
# control the prompt, you can adapt it to your tissue and see exactly what was asked.

# %% [markdown]
# ### Choose a backend
#
# Set `LLM_BACKEND` and fill in the model + credentials for that backend only:
#
# - **`"ollama"`** — free, local/cluster, OpenAI-compatible. See the Ollama setup guide below.
# - **`"claude"`** — Anthropic API via the `anthropic` SDK. Set `ANTHROPIC_API_KEY`.
# - **`"claude_code"`** — Claude Agent SDK, which uses your existing **Claude Code login** — no API
#   key needed. Requires the `claude` CLI installed and authenticated (`claude login`); the SDK
#   talks to that local CLI process.
# - **`"openai"`** — OpenAI API. Set `OPENAI_API_KEY`.
# - **`"gemini"`** — Google Gemini API. Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

# %%
LLM_BACKEND = "claude_code"  # one of: "ollama", "claude", "claude_code", "openai", "gemini"

# Ollama (OpenAI-compatible; no API key needed)
OLLAMA_URL   = "http://supergpu22.scidom.de:8979"  # replace with your node and port
OLLAMA_MODEL = "qwen3.6:35b"

# Claude via the Anthropic API SDK — reads ANTHROPIC_API_KEY from the environment
CLAUDE_MODEL = "claude-opus-4-8"   # "claude-haiku-4-5" is cheaper/faster

# Claude via the Claude Agent SDK (uses your Claude Code login; no API key)
CLAUDE_CODE_MODEL = "opus"         # short alias ("opus"/"sonnet"/"haiku") or a full model ID

# OpenAI — reads OPENAI_API_KEY from the environment
OPENAI_MODEL = "gpt-4o"

# Gemini (google-genai) — reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment
GEMINI_MODEL = "gemini-2.5-flash"

# %% [markdown]
# ### The backend dispatcher
#
# One `call_llm(system, user)` function, five backends. Only the selected backend's package needs
# to be installed.

# %%
def _run_async(coro):
    """Run an async coroutine from sync code, tolerating an already-running loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import nest_asyncio  # only needed inside a live loop (Jupyter)
    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


def call_llm(system, user):
    if LLM_BACKEND == "ollama":
        from openai import OpenAI
        client = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        return resp.choices[0].message.content

    if LLM_BACKEND == "openai":
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        return resp.choices[0].message.content

    if LLM_BACKEND == "claude":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        # Note: newer Claude models reject temperature/top_p — steer via the prompt instead.
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    if LLM_BACKEND == "claude_code":
        # Claude Agent SDK — talks to your local, logged-in `claude` CLI (no API key).
        from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

        async def _ask():
            text = ""
            options = ClaudeAgentOptions(
                system_prompt=system, model=CLAUDE_CODE_MODEL, max_turns=1, allowed_tools=[],
            )
            async for message in query(prompt=user, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text += block.text
            return text

        return _run_async(_ask())

    if LLM_BACKEND == "gemini":
        from google import genai
        client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=f"{system}\n\n{user}")
        return resp.text

    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r}")


# %% [markdown]
# ### The predefined prompt
#
# A fixed expert system prompt plus a per-factor user prompt that injects **two** ranked gene lists
# (OOD = specific genes, IND = direct-effect / logFC-like genes), explains what each means, asks the
# model to reason, and constrains the output to a small JSON object. Adapt the wording to your own
# tissue/organism if needed.

# %%
IDENTIFY_SYSTEM = (
    "You are an expert computational biologist specializing in single-cell transcriptomics and "
    "immunology. You interpret latent gene programs learned by DRVI, a disentangled variational "
    "model. Each program is summarized by two complementary ranked marker-gene lists — a "
    "specificity score and a direct-effect score — whose meanings are explained in the prompt. "
    "Given these lists and the tissue context, identify what the program most likely represents, "
    "reasoning from established marker-gene biology. Be precise and do not overstate confidence "
    "when the genes are ambiguous."
)


def build_identify_prompt(factor_label, ood_genes, ind_genes, tissue):
    return (
        f"Tissue context: {tissue}\n"
        f"DRVI program: {factor_label}\n\n"
        "You are given two complementary ranked marker-gene lists for this program "
        "(both ranked most-influential first):\n\n"
        "1. SPECIFIC genes (OOD score): genes that most *specifically* mark this program. This "
        "score penalizes genes that are shared across many programs, so these are the program's "
        "most distinctive identity markers.\n"
        f"{', '.join(ood_genes)}\n\n"
        "2. DIRECT-EFFECT genes (IND score): the latent factor's direct effect on each gene, "
        "analogous to a log fold-change. It does NOT penalize sharing, so it also includes "
        "differential genes that are shared between programs — useful for reading the broader "
        "biological process and shared machinery.\n"
        f"{', '.join(ind_genes)}\n\n"
        "Use the SPECIFIC list mainly to pin down cell-type identity, and the DIRECT-EFFECT list to "
        "read the broader process (including shared genes). Reason from both, then give your answer "
        "as a JSON object with exactly these keys:\n"
        '  "cell_type": most likely cell type or cell state (or "unclear")\n'
        '  "biological_process": dominant biological process or pathway (or "unclear")\n'
        '  "key_genes": up to 5 genes that most support the call (list of strings)\n'
        '  "confidence": one of "high", "medium", "low"\n'
        '  "reasoning": one or two sentences justifying the call\n'
        "Respond with ONLY the JSON object — no surrounding text and no code fences."
    )


# %% [markdown]
# ### Parse and run
#
# LLMs sometimes wrap JSON in prose or code fences, so we extract the first JSON object defensively.

# %%
def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def top_genes(scores, col, cutoff, top_n):
    s = scores[col]
    return s[s >= cutoff].nlargest(top_n).index.astype(str).tolist()


def identify_factors(ood_scores, ind_scores, tissue, ood_cutoff, ind_cutoff, top_n, max_dirs=None):
    rows = []
    for col in ood_scores.columns:
        ood_genes = top_genes(ood_scores, col, ood_cutoff, top_n)
        if not ood_genes:  # uninformative direction — skip
            continue
        ind_genes = top_genes(ind_scores, col, ind_cutoff, top_n)
        parsed = parse_json(
            call_llm(IDENTIFY_SYSTEM, build_identify_prompt(col, ood_genes, ind_genes, tissue))
        )
        key_genes = parsed.get("key_genes")
        rows.append({
            "factor": col[:-1].strip(),
            "direction": col[-1],
            "cell_type": parsed.get("cell_type"),
            "biological_process": parsed.get("biological_process"),
            "key_genes": ", ".join(key_genes) if isinstance(key_genes, list) else key_genes,
            "confidence": parsed.get("confidence"),
            "reasoning": parsed.get("reasoning"),
        })
        print(f"{col}: {parsed.get('cell_type')} / {parsed.get('biological_process')}")
        if max_dirs is not None and len(rows) >= max_dirs:
            break
    return pd.DataFrame(rows)


llm_direct_results = identify_factors(
    scores_df, ind_scores, llm_tissue_context, drvi_score_cutoff, ind_score_cutoff,
    llm_top_n_genes, max_directions,
)
with pd.option_context("display.max_colwidth", None):
    display(llm_direct_results)

# %% [markdown]
# ### Store results

# %%
embed.uns["llm_direct_results"] = llm_direct_results.convert_dtypes(
    convert_integer=False, convert_floating=False
)
embed.var.set_index("title", drop=False, inplace=True)
for d, suf in [("+", "positive"), ("-", "negative")]:
    sub = llm_direct_results.query("direction == @d").set_index("factor")
    embed.var[f"{suf}_direction_llm_celltype"] = sub["cell_type"]
    embed.var[f"{suf}_direction_llm_process"] = sub["biological_process"]
embed.var.index = embed.var["original_dim_id"].astype(int).astype(str)
embed.var.index.name = None

# %% [markdown]
# **How to read this.** This runs on every factor.
# When the gene list points clearly at one lineage, `cell_type` and `biological_process`
# tend to agree and `confidence` is `"high"`. Because the output is fluent and self-reported, treat
# `"low"`/`"medium"` confidence calls with caution and always cross-check against the SMI,
# enrichment tools, and the literature.
# The `key_genes` and `reasoning` fields let you trace each call back to the factor's marker list.

# %% [markdown]
# ## 2. CASSIA
#
# [CASSIA](https://github.com/ElliotXie/CASSIA)
# ([Nature Comms 2025](https://www.nature.com/articles/s41467-025-67084-x)) is a multi-agent system:
# a **chain-of-thought annotation agent**, a **validation agent** that loops (up to 3×) checking
# marker consistency, and a **formatting agent** that emits a general + detailed cell type. Backends:
# OpenAI, Anthropic, OpenRouter, or any OpenAI-compatible URL (Ollama). It writes CSV/JSON/HTML
# reports to the working directory on each run (cleaned up below).

# %% [markdown]
# ### Setup

# %%
import CASSIA

# CASSIA reaches an OpenAI-compatible endpoint; here we point it at Ollama.
cassia_output_name = "cassia_drvi"
cassia_provider = f"{OLLAMA_URL}/v1"
cassia_model = OLLAMA_MODEL

CASSIA.set_api_key("ollama", provider=cassia_provider)


# %% [markdown]
# ### Run

# %%
def run_cassia_annotation(scores_df, tissue, cutoff, top_n, output_name, provider, model, species,
                          max_dirs=None):
    rows = []
    for col in scores_df.columns:
        genes = scores_df[col][scores_df[col] >= cutoff].nlargest(top_n).index.tolist()
        if genes:
            cluster_id = f"{col[:-1].strip().replace(' ', '_')}{col[-1]}"
            rows.append({"cluster": cluster_id, "gene": ", ".join(genes)})
            if max_dirs is not None and len(rows) >= max_dirs:
                break

    cassia_input = pd.DataFrame(rows)
    print(f"CASSIA input: {len(cassia_input)} factor-directions")

    CASSIA.runCASSIA_batch(
        marker=cassia_input, output_name=output_name, provider=provider, model=model,
        tissue=tissue, species=species, max_workers=4, validate_api_key_before_start=False,
    )

    results = pd.read_csv(f"{output_name}_summary.csv")
    results["factor"] = results["Cluster ID"].str[:-1].str.replace("_", " ")
    results["direction"] = results["Cluster ID"].str[-1]

    for p in Path(".").glob(f"{output_name}*"):
        p.unlink()
    return results


cassia_results = run_cassia_annotation(
    scores_df=scores_df, tissue=llm_tissue_context, cutoff=drvi_score_cutoff, top_n=llm_top_n_genes,
    output_name=cassia_output_name, provider=cassia_provider, model=cassia_model, species=llm_species,
    max_dirs=max_directions,
)
cassia_results.head()

# %% [markdown]
# ### Store results

# %%
embed.uns["cassia_results"] = cassia_results.convert_dtypes(
    convert_integer=False, convert_floating=False
)
embed.var.set_index("title", drop=False, inplace=True)
for d, suf in [("+", "positive"), ("-", "negative")]:
    sub = cassia_results.query("direction == @d").set_index("factor")
    embed.var[f"{suf}_direction_cassia_general"] = sub["Predicted General Cell Type"]
    embed.var[f"{suf}_direction_cassia_detailed"] = sub["Predicted Detailed Cell Type"]
embed.var.index = embed.var["original_dim_id"].astype(int).astype(str)
embed.var.index.name = None

# %% [markdown]
# **How to read this.** This runs on every factor.
# We expect cell types to be captured well by this approach.
# Because the output is fluent and self-reported, treat results with caution and always cross-check against the SMI,
# marker databases, and the literature.

# %% [markdown]
# ## 3. gs2txt
#
# gs2txt runs pathway enrichment on the gene set first, then combines the enriched terms into a
# structured prompt so the LLM produces a free-text process description. Providers: OpenAI,
# Anthropic, or any OpenAI-compatible endpoint via `base_url` (Ollama). Install with the
# `enrichment` extra so gseapy is available.

# %% [markdown]
# ### Setup

# %%
from gs2txt import GeneSetAnnotator
from gs2txt.llm import OpenAIProvider

gs2txt_temperature = 0.1
gs2txt_enrichment_method = "pathway"

gs2txt_annotator = GeneSetAnnotator(
    llm_provider=OpenAIProvider(
        api_key="ollama", model_id=OLLAMA_MODEL,
        temperature=gs2txt_temperature, base_url=f"{OLLAMA_URL}/v1",
    ),
    enrichment_method=gs2txt_enrichment_method,
    organism=llm_species,
)


# %% [markdown]
# ### Run

# %%
def run_gs2txt(scores_df, annotator, cutoff, top_n, context, max_dirs=None):
    rows = []
    for col in scores_df.columns:
        top = scores_df[col][scores_df[col] >= cutoff].nlargest(top_n)
        if top.empty:
            continue
        rows.append({
            "factor": col[:-1].strip(),
            "direction": col[-1],
            "description": annotator.annotate(
                pd.DataFrame({"gene": top.index, "logFC": top.values}),
                max_gene_num=top_n,
                additional_context=f"DRVI factor {col} - {context}",
            ),
        })
        if max_dirs is not None and len(rows) >= max_dirs:
            break
    return pd.DataFrame(rows)


gs2txt_results = run_gs2txt(
    scores_df, gs2txt_annotator, drvi_score_cutoff, llm_top_n_genes, llm_tissue_context, max_directions
)
with pd.option_context("display.max_colwidth", None):
    display(gs2txt_results)

# %% [markdown]
# ### Store results

# %%
embed.uns["gs2txt_results"] = gs2txt_results.convert_dtypes(
    convert_integer=False, convert_floating=False
)
embed.var.set_index("title", drop=False, inplace=True)
for d, suf in [("+", "positive"), ("-", "negative")]:
    sub = gs2txt_results.query("direction == @d").set_index("factor")
    embed.var[f"{suf}_direction_gs2txt_label"] = sub["description"]
embed.var.index = embed.var["original_dim_id"].astype(int).astype(str)
embed.var.index.name = None

# %% [markdown]
# **How to read this.** Because gs2txt names genes and pathways, its summaries can be traced back to
# the factor's top-ranked list — a useful gene-level complement to the ORA and TF tools. As with the
# others, the output is fluent and unscored, so interpret it with care and alongside the rest rather than on its own.

# %% [markdown]
# ## 4. Save

# %%
import anndata as ad

ad.settings.allow_write_nullable_strings = True
embed.write_h5ad(embed_path)
print(f"Updated embedding saved to: {embed_path}")

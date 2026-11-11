"""
Runs the walkthrough on a Modal GPU worker. Two modes:

    # Load the model + pre-fitted lens, print the top-5 readout, write slice.html locally.
    uv run --with modal modal run scripts/modal_jlens.py --mode apply

    # Fit a lens from scratch (walkthrough section 6); output is saved to the "jlens" volume.
    uv run --with modal modal run scripts/modal_jlens.py --mode fit --n-prompts 100

The "jlens" Modal volume holds the HF cache (model weights + downloaded lens, so only the first run
downloads) and fitted lenses at lenses/<model-slug>/jacobian_lens_n<N>.pt. To use one locally:

    uv run --with modal modal volume get jlens lenses/<model-slug>/jacobian_lens_n100.pt lenses/

Optionally create a secret named "huggingface" containing HF_TOKEN for faster, rate-limit-free downloads:

    uv run --with modal modal secret create huggingface HF_TOKEN=...
"""

import os
import sys
from pathlib import Path

import modal

# In the container the local jlens/ is copied to /root/jlens; make it importable.
sys.path.insert(0, "/root")

# Set MODAL_GPU to change the card. 4B fits on an L40S; Qwen3.6-27B needs an 80GB card (e.g. A100-80GB).
GPU = os.environ.get("MODAL_GPU", "L40S") or None
VOL = "/vol"

# Same lens table as walkthrough.ipynb.
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = {
    "Qwen/Qwen3.5-4B": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    "Qwen/Qwen3.6-27B": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
}

app = modal.App("jlens")
volume = modal.Volume.from_name("jlens", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pinned versions come from uv.lock. The dev extra is only here for `datasets` (fit's wikitext prompts);
    # the uv venv has no pip, so a separate .pip_install would fail.
    .uv_sync(extra_options="--extra dev")
    .env({"HF_HOME": f"{VOL}/hf"})
    # add_local_dir so the gloss file in assets/ and the vis HTML template in jlens/data/ come along.
    .add_local_dir("jlens", remote_path="/root/jlens")
    .add_local_dir("assets", remote_path="/root/assets")
)
function_kwargs = dict(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface")] if os.environ.get("MODAL_HF_SECRET") else [],
)


def _load_model(model_name: str):
    import torch
    import transformers

    import jlens

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return jlens.from_hf(hf_model, tokenizer), tokenizer


@app.function(**function_kwargs)
def apply(model_name: str, example_slug: str) -> tuple[str, str]:
    """Walkthrough sections 1-4: readout at the second-to-last position + an embedded slice page, both of the example's prompt."""
    import gzip
    import json

    import jlens
    from jlens.examples import EXAMPLES, resolve_prompt
    from jlens.vis import build_page, compute_slice

    model, tokenizer = _load_model(model_name)
    example = next(e for e in EXAMPLES if e.slug == example_slug)
    prompt = resolve_prompt(example, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE[model_name], revision=LENS_REVISION)

    layers = [model.n_layers // 4, model.n_layers // 2, model.n_layers // 4 * 3, model.n_layers - 2]
    jlens_logits, model_logits, _ = lens.apply(model, prompt, layers=layers, positions=[-2])
    logit_lens, _, _ = lens.apply(model, prompt, layers=layers, positions=[-2], use_jacobian=False)

    def top5(logits):
        return [tokenizer.decode([t]) for t in logits.topk(5).indices]

    lines = []
    for layer in layers:
        lines.append(f"L{layer:>3} logit-lens: {top5(logit_lens[layer][0])}")
        lines.append(f"L{layer:>3} J-lens:     {top5(jlens_logits[layer][0])}")
    lines.append(f"model:           {top5(model_logits[0])}")

    gloss = {int(k): v for k, v in json.load(gzip.open("/root/assets/qwen_gloss.json.gz")).items()}
    slice_data = compute_slice(model, lens, prompt, layer_stride=2, mask_display=True)
    page, _, _ = build_page(
        slice_data,
        prompt,
        title=example.section,
        description=example.description,
        alt_token=gloss,
    )
    volume.commit()  # persist the HF cache for the next run
    return "\n".join(lines), page


@app.function(**{**function_kwargs, "timeout": 60 * 60 * 6})
def fit(model_name: str, n_prompts: int, dim_batch: int, checkpoint_every: int) -> str:
    """Walkthrough section 6. Resumable: rerunning with the same args picks up from the checkpoint."""
    import jlens
    from jlens.examples import load_wikitext_prompts

    model, _ = _load_model(model_name)
    out_dir = Path(VOL) / "lenses" / model_name.split("/")[-1].lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"jacobian_lens_n{n_prompts}.pt"

    prompts = load_wikitext_prompts(n_prompts=n_prompts)
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=dim_batch,
        max_seq_len=128,
        checkpoint_path=str(out_dir / f"ckpt_n{n_prompts}.pt"),
        checkpoint_every=checkpoint_every,
    )
    lens.save(str(out_path))
    volume.commit()  # make the lens durable before the function returns
    return f"{lens}\nsaved to volume jlens at {out_path.relative_to(VOL)}"


@app.local_entrypoint()
def main(
    mode: str = "apply",
    model: str = "Qwen/Qwen3.5-4B",
    example: str = "multihop",
    n_prompts: int = 100,
    dim_batch: int = 32,
    checkpoint_every: int = 10,
) -> None:
    if mode == "apply":
        text, page = apply.remote(model, example)
        print(text)
        Path("slice.html").write_text(page)
        print("wrote slice.html")
    elif mode == "fit":
        print(fit.remote(model, n_prompts, dim_batch, checkpoint_every))
    else:
        raise SystemExit(f"unknown mode {mode!r}; use 'apply' or 'fit'")

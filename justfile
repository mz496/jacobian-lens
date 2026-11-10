# modal isn't a dependency of this repo, so it's added per-invocation with --with.

# Applies a pre-fitted lens on a Modal GPU worker; prints the readout and writes slice.html locally
# e.g. just modal-apply --model Qwen/Qwen3.6-27B --example ascii-face   (set MODAL_GPU=A100-80GB for 27B)
modal-apply *args:
    uv run --with modal modal run scripts/modal_jlens.py --mode apply {{args}}

# Fits a lens on a Modal GPU worker; resumable, output goes to the jlens volume under lenses/
# e.g. just modal-fit --n-prompts 100 --dim-batch 32
modal-fit *args:
    uv run --with modal modal run scripts/modal_jlens.py --mode fit {{args}}

# Downloads a fitted lens from the jlens volume into lenses/
# e.g. just modal-get-lens qwen3.5-4b/jacobian_lens_n100.pt
modal-get-lens path:
    mkdir -p lenses
    uv run --with modal modal volume get jlens lenses/{{path}} lenses/

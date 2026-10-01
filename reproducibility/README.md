# Data and execution guide

The reconstruction tools obtain conversation text from upstream repositories. `input_manifest.json` identifies the sources, staging configuration and expected reconstructed input hashes. `source_hashes.json` contains the upstream file hashes used by the reconstruction checks. These files are input configuration and integrity metadata.

## Data sources

| Corpus | Upstream source |
| :--- | :--- |
| OASST1 | https://huggingface.co/datasets/OpenAssistant/oasst1 |
| WildChat | https://huggingface.co/datasets/allenai/WildChat-1M |
| ShareGPT | https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered |
| OpenOrca | https://huggingface.co/datasets/Open-Orca/OpenOrca |

Consult the upstream source for dataset access conditions and license terms. Dataset revisions for the original inputs were not recorded; exact reconstruction depends on obtaining bytes matching the recorded hashes. Reconstruction stops on a hash mismatch.

## Reconstruct and verify inputs

Run from the repository root, with the virtual environment activated:

```sh
python reproducibility/rebuild_data.py --storage work/data --download
python reproducibility/verify_inputs.py work/data/derived
```

Downloads require several GB of disk space. To supply local files, place them at `work/data/raw/<corpus>/<filename>` and omit `--download`. Use `--corpora oasst` to reconstruct one corpus. The verification command expects all four reconstructed inputs.

The staging seed and retention rates are recorded in `input_manifest.json`. Staging keeps complete conversations or rows. The simulator then applies canonical exact deduplication and deterministic sampling. OpenOrca supplies system and question text; OASST1 retains branching reply trees.

## Experiment commands

From the repository root, set `DATA` to your verified input directory and keep caches inside the checkout:

```sh
export DATA="$PWD/work/data/derived"
export TIKTOKEN_CACHE_DIR="$PWD/work/cache/tiktoken"
export HF_HOME="$PWD/work/cache/huggingface"
export PYTHONPATH=revision
python -m sim.run_tinv --data-dir "$DATA" --out work/runs/counting
python -m sim.run_phase2 --data-dir "$DATA" --out work/runs/schedule --exp A
python revision/engine_ref/run_compare.py --data-dir "$DATA" --out work/runs/component
```

Additional entry points cover full analysis frames (`sim.run_fullframe`), deduplication and sampling (`sim.run_strata`), and tokenizer template comparisons (`sim.run_template`). Run each entry point with `--help` for its options. Choose a separate `--out` directory for each run. The runners generate new result files locally.

For native templates, pass `--tok-dir work/data/tokenizers` and place tokenizer files under `Qwen/Qwen2.5-0.5B` and `TinyLlama/TinyLlama-1.1B-Chat-v1.0` within that directory. The tokenizer revisions are `060db6499f32faf8b98477b0a26969ef7d8b9987` and `fe8a4ea1ffedaf415f4da2f062534de366a451e6`, respectively. Model weights are unnecessary.

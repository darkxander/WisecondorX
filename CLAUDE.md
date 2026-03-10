# WisecondorX

## Project Overview

WisecondorX is a shallow WGS copy number variant detection tool.

## Development

- Package manager: **uv** — always use `uv run` for Python, `uv add` for deps
- Python: 3.12 (requires `>=3.9,<3.14` — scikit-learn<=1.4.2 fails on 3.14)
- Build: hatchling (`pyproject.toml`)
- Install with ANN backends: `uv sync --extra fast --extra ann`
- CPU-only: faiss-cpu, no CUDA. Dev on M4 Mac, production on Linux AMD.

## Key Source Files

- `src/wisecondorx/newref_tools.py` — masking, PCA, KNN backends, null ratios
- `src/wisecondorx/newref_control.py` — newref orchestration (prep, main, merge)
- `src/wisecondorx/main.py` — CLI entry point
- `src/wisecondorx/predict_tools.py` — prediction (uses reference indexes/distances)
- `src/wisecondorx/overall_tools.py` — sample loading, bin scaling

## Architecture: newref pipeline

1. Load samples → gender model → masking
2. `tool_newref_prep`: normalize → early masking → PCA correction → PCA distance filter → dim reduction
3. `tool_newref_main`: faiss/hnswlib/sklearn KNN → joblib-parallel null ratios → save
4. Repeat for A(utosomal), F(emale gonosomal), M(ale gonosomal)
5. `tool_newref_merge`: combine into final .npz

## Critical Constraints

- `indexes` in .npz are indices into chr_data (all bins excluding current chromosome)
  — predict_tools.py:_normalize_once() depends on this exact format
- Distances are squared L2 (native faiss/hnswlib format)
- Output .npz structure must match: same keys, dtypes for predict compatibility
- KNN backend chain: faiss-cpu → hnswlib → sklearn (auto-detected at import)

## Gotchas

- After depth normalization, per-bin values are ~1/n_bins (e.g. 3e-5 for 30k bins)
  — never use absolute thresholds; use relative (fraction of median)
- With few samples (e.g. 10), PCA n_components is capped at n_samples-1
- Early masking can remove too many bins if thresholds are absolute

## Testing

```bash
# Build reference with example data (10 samples, 5k binsize)
uv run wisecondorx newref examples/npz_files/*.npz /tmp/ref_test.npz --cpus 4
# At native 5k binsize (~550k bins):
uv run wisecondorx newref examples/npz_files/*.npz /tmp/ref_5k.npz --cpus 4 --binsize 5000
```

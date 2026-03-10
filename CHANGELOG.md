# Changelog

## 2.0.0

Major performance and architecture overhaul for reference building (`newref`).

### Added
- **Approximate Nearest Neighbor search** using faiss-cpu (`IndexFlatL2` / `IndexIVFFlat`),
  with tiered fallback to hnswlib then sklearn `NearestNeighbors`
- **PCA dimensionality reduction** before KNN — reduces bin representations from n_samples
  to configurable number of components (default: 30) for faster distance computation
- **Early bin masking** — pre-filters bins with near-zero mean (<1% of median) or excessive
  coefficient of variation (>3x median CV) before PCA and KNN
- **Chunked processing** — processes bins in configurable chunks (default: 50,000) to bound
  peak RAM regardless of total bin count
- **Joblib parallelism** for null ratio computation over bin chunks
- **Vectorized null ratio computation** — replaces Python loops with numpy operations
- New CLI arguments: `--chunk-size`, `--n-components`, `--pcacomp`
- Reference QC (`qc_reference`) now runs automatically after `newref`
- Gonosomal references (F/M) now respect `--cpus` (previously hardcoded to 1)

### Changed
- **Project management migrated to uv** with `pyproject.toml` (hatchling build system)
- Removed `setup.cfg` and `setup.py`
- Python requirement updated to `>=3.9,<3.14`
- `faiss-cpu` and `hnswlib` are optional dependencies (`--extra fast`, `--extra ann`)
- `joblib` added as explicit dependency (was transitive via scikit-learn)
- KNN distances are now squared L2 in reduced PCA space (prediction uses adaptive cutoff,
  so this is transparent)

### Removed
- Exact KNN implementation (`get_ref_for_bins`, `get_reference`) — replaced by ANN backends
- Thread-based parallelism via `ThreadPoolExecutor` — replaced by joblib
- Intermediate part-file mechanism (`_tool_newref_part`, `tool_newref_post`) — no longer needed
  since ANN search is fast enough for direct processing

### Performance
- Reference building at 100kb binsize (~27k bins): ~15s vs ~60s (4x faster)
- Reference building at 5kb binsize (~550k bins): ~4.5min vs hours
- Enables reference building at 1kb binsize (~3M bins) which was previously intractable

## 1.2.10

- Added reference QC module
- PCA distance filtering for anomalous bin removal
- Docker and Conda configuration files
- Custom regions bed file output

## 1.2.9 and earlier

See git history for previous changes.

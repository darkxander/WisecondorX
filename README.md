# WisecondorX

An evolved WISECONDOR for copy number detection from shallow whole-genome sequencing data.

WisecondorX normalizes sequencing data using within-sample reference bins identified via
nearest-neighbor search, enabling detection of copy number alterations in NIPT, gDNA, PGT,
FFPE, LQB and other sample types.

For background on the algorithm, see
[Straver et al (2014)](https://www.ncbi.nlm.nih.gov/pubmed/24170809),
[Huijsdens-van Amsterdam et al (2018)](https://www.nature.com/articles/gim201832.epdf),
and [our comparison paper](https://www.ncbi.nlm.nih.gov/pubmed/30566647).

## What's new in v2.0.0

- **Approximate Nearest Neighbor search** via faiss-cpu (with hnswlib and sklearn fallbacks)
- **PCA dimensionality reduction** before KNN for faster distance computation
- **Early bin masking** removes low-mean and high-CV bins before reference building
- **Chunked processing** with configurable chunk size to bound peak RAM
- **Joblib parallelism** over bin chunks for null ratio computation
- **uv-managed project** with `pyproject.toml` (replaces setup.cfg/setup.py)
- Reference building at 5kb binsize (~550k bins) now completes in minutes instead of hours

## Installation

### Using uv (recommended)

```bash
# Clone and install
git clone https://github.com/CenterForMedicalGeneticsGhent/WisecondorX.git
cd WisecondorX
uv sync

# With fast ANN backend (recommended)
uv sync --extra fast

# With alternative ANN backend
uv sync --extra ann

# With both
uv sync --extra fast --extra ann
```

### Using pip

```bash
pip install -U git+https://github.com/CenterForMedicalGeneticsGhent/WisecondorX

# Optional: install faiss for best performance
pip install faiss-cpu
# Or hnswlib as alternative
pip install hnswlib
```

### Using Conda

```bash
conda install -f -c conda-forge -c bioconda wisecondorx
```

## Quick Start

```bash
# 1. Convert BAM/CRAM to .npz
wisecondorx convert input.bam output.npz

# 2. Create reference from healthy controls
wisecondorx newref reference_dir/*.npz reference.npz --cpus 8

# 3. Predict copy number alterations
wisecondorx predict test.npz reference.npz output_id --bed --plot
```

## Usage

### Stage 1: Convert aligned reads

```bash
wisecondorx convert input.bam/cram output.npz [--optional arguments]
```

| Argument | Description |
|:---|:---|
| `--reference x` | Fasta reference for CRAM inputs |
| `--binsize x` | Bin size in bp; reference binsize should be a multiple of this (default: 5000) |
| `--normdup` | Skip duplicate removal |

**Note:** Do **not** apply read quality filtering before WisecondorX — low-quality reads
help distinguish informative from non-informative bins. We recommend
[bowtie2](https://github.com/BenLangmead/bowtie2) for mapping.

### Stage 2: Create reference

```bash
wisecondorx newref reference_dir/*.npz reference.npz [--optional arguments]
```

| Argument | Description |
|:---|:---|
| `--nipt` | **Required for NIPT references** |
| `--binsize x` | Output resolution in bp (default: 100000) |
| `--refsize x` | Reference locations per target bin (default: 300) |
| `--cpus x` | Number of parallel workers (default: 1) |
| `--chunk-size x` | Bins per processing chunk; controls peak RAM (default: 50000) |
| `--n-components x` | PCA components for KNN dimensionality reduction (default: 30) |
| `--pcacomp x` | PCA components for between-sample normalization (default: 5) |
| `--yfrac x` | Manual Y-fraction cutoff for gender classification |
| `--plotyfrac x` | Plot Y-fraction histogram to file; exits after plotting |

**Important notes:**
- The reference set must consist of **negative control samples** from the same sequencer,
  mapper, reference genome and sample type as the test samples.
- Include at least 50 samples per reference. More is better up to ~500.
- Gender prediction uses a Gaussian mixture model. With <20 samples or imbalanced
  gender ratios, consider setting `--yfrac` manually.

### Stage 3: Predict copy number alterations

```bash
wisecondorx predict test.npz reference.npz output_id [--optional arguments]
```

| Argument | Description |
|:---|:---|
| `--bed` | Output tab-delimited .bed files **(*)** |
| `--plot` | Output .png plots **(*)** |
| `--minrefbins x` | Min reference bins per target (default: 150) |
| `--maskrepeats x` | Masking stringency cycles (default: 5) |
| `--zscore x` | Z-score cutoff for aberration calling (default: 5) |
| `--alpha x` | CBS breakpoint p-value cutoff (default: 1e-4) |
| `--beta x` | Ratio cutoff for aberrations; overrides `--zscore`. Optimally close to purity (0–1) |
| `--blacklist x` | Headerless .bed file for additional masking |
| `--gender x` | Force analysis as male (M) or female (F) |
| `--regions x` | Mark custom regions on plots (headerless .bed) |
| `--ylim [a,b]` | Y-axis limits for plotting |
| `--cairo` | Use cairo bitmap type for plots |
| `--seed x` | Random seed for CBS |

<sup>**(*)** At least one output format must be selected</sup>

### Gender prediction

```bash
wisecondorx gender test.npz reference.npz
```

## KNN Backend Selection

WisecondorX automatically selects the best available KNN backend:

| Priority | Backend | Package | Performance |
|:---|:---|:---|:---|
| 1 | faiss | `faiss-cpu` | Best — optimized BLAS, IVF indexing for >500k bins |
| 2 | hnswlib | `hnswlib` | Good — HNSW graph search, native batch queries |
| 3 | sklearn | (built-in) | Baseline — exact search, always available |

Install faiss-cpu for best performance:
```bash
uv add faiss-cpu  # or: pip install faiss-cpu
```

## Output Files

### Plots

Each dot represents a bin, positioned by chromosome (X-axis) and log2 ratio (Y-axis).
Copy-neutral bins cluster around 0. Segments (white lines) indicate regions of predicted
equal copy number. Dot size reflects reference-set variability (larger = more certain).
Grey bars indicate blacklisted regions. Colored dotted lines show expected constitutional
1n/3n states at 100% purity.

### Tables

| File | Contents |
|:---|:---|
| `ID_bins.bed` | Per-bin information. NaN = blacklisted. Z-scores use within-sample reference. |
| `ID_segments.bed` | Per-segment information with between-sample Z-scores. |
| `ID_aberrations.bed` | Aberrant segments per `--beta` or `--zscore` thresholds. |
| `ID_statistics.bed` | Per-chromosome and overall statistics. Useful for NIPT. |

## Dependencies

### Required
- Python >= 3.9, < 3.14
- scipy, scikit-learn (<=1.4.2), pysam, numpy, pandas, joblib

### Optional (recommended)
- faiss-cpu — fast approximate nearest neighbor search
- hnswlib — alternative ANN backend

### For plotting (CBS segmentation)
- R with packages: jsonlite, DNAcopy (Bioconductor)

## License

Licensed under [CC BY-NC-SA](LICENSE.md).

## Authors

Matthias De Smet, Lennart Raman — Center for Medical Genetics Ghent

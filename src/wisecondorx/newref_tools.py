# WisecondorX

import logging
import random

import numpy as np
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# KNN backend detection (faiss > hnswlib > sklearn)
# ---------------------------------------------------------------------------
_KNN_BACKEND = None

try:
    import faiss

    _KNN_BACKEND = "faiss"
except ImportError:
    try:
        import hnswlib

        _KNN_BACKEND = "hnswlib"
    except ImportError:
        _KNN_BACKEND = "sklearn"


def _log_knn_backend():
    if _KNN_BACKEND == "faiss":
        logging.info("KNN backend: faiss-cpu")
    elif _KNN_BACKEND == "hnswlib":
        logging.info(
            "KNN backend: hnswlib (faiss-cpu not installed)"
        )
    else:
        logging.warning(
            "KNN backend: sklearn (slowest). "
            "Install faiss-cpu or hnswlib for better performance."
        )


# ---------------------------------------------------------------------------
# Gender model
# ---------------------------------------------------------------------------


def train_gender_model(args, samples):
    genders = np.empty(len(samples), dtype="object")
    y_fractions = []
    for sample in samples:
        y_fractions.append(
            float(np.sum(sample["24"]))
            / float(
                np.sum([np.sum(sample[x]) for x in sample.keys()])
            )
        )
    y_fractions = np.array(y_fractions)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        reg_covar=1e-99,
        max_iter=10000,
        tol=1e-99,
    )
    gmm.fit(X=y_fractions.reshape(-1, 1))
    gmm_x = np.linspace(0, 0.02, 5000)
    gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

    if args.plotyfrac is not None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.hist(y_fractions, bins=100, density=True)
        ax.plot(gmm_x, gmm_y, "r-", label="Gaussian mixture fit")
        ax.set_xlim([0, 0.02])
        ax.legend(loc="best")
        plt.savefig(args.plotyfrac)
        logging.info(
            "Image written to {}, now quitting ...".format(
                args.plotyfrac
            )
        )
        exit()

    if args.yfrac is not None:
        cut_off = args.yfrac
    else:
        sort_idd = np.argsort(gmm_x)
        sorted_gmm_y = gmm_y[sort_idd]

        local_min_i = argrelextrema(sorted_gmm_y, np.less)

        cut_off = gmm_x[local_min_i][0]
        logging.info(
            "Determined --yfrac cutoff: {}".format(
                str(round(cut_off, 4))
            )
        )

    genders[y_fractions > cut_off] = "M"
    genders[y_fractions < cut_off] = "F"

    return genders.tolist(), cut_off


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def get_mask(samples):
    by_chr = []
    bins_per_chr = []
    sample_count = len(samples)

    for chr in range(1, 25):
        max_len = max(
            [sample[str(chr)].shape[0] for sample in samples]
        )
        this_chr = np.zeros((max_len, sample_count), dtype=float)
        bins_per_chr.append(max_len)
        i = 0
        for sample in samples:
            this_chr[:, i] = sample[str(chr)]
            i += 1
        by_chr.append(this_chr)
    all_data = np.concatenate(by_chr, axis=0)

    sum_per_sample = np.sum(all_data, 0)
    all_data = all_data / sum_per_sample

    sum_per_bin = np.sum(all_data, 1)
    median_cov = np.median(sum_per_bin[sum_per_bin > 0])
    mask = sum_per_bin > (0.05 * median_cov)

    return mask, bins_per_chr


def apply_early_masking(masked_data, mask, samples, chrs):
    """Pre-filter bins with near-zero mean or excessive variance.

    Returns updated masked_data after modifying mask in-place.
    """
    mean_per_bin = np.mean(masked_data, axis=1)

    # Bins with near-zero mean (relative threshold: < 1% of median)
    # After depth normalization, absolute values are tiny fractions,
    # so we use a relative threshold instead of absolute 1e-3.
    median_mean = np.median(mean_per_bin[mean_per_bin > 0])
    low_thresh = median_mean * 0.01
    bad_low_mean = mean_per_bin < low_thresh
    n_low = int(np.sum(bad_low_mean))

    # Coefficient of variation filter
    std_per_bin = np.std(masked_data, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_per_bin = np.where(
            mean_per_bin > 0,
            std_per_bin / mean_per_bin,
            np.inf,
        )
    finite_cv = cv_per_bin[np.isfinite(cv_per_bin)]
    if len(finite_cv) > 0:
        median_cv = np.median(finite_cv)
        bad_high_cv = cv_per_bin > 3 * median_cv
    else:
        bad_high_cv = np.zeros(len(cv_per_bin), dtype=bool)
    n_cv = int(np.sum(bad_high_cv & ~bad_low_mean))

    bad_bins = bad_low_mean | bad_high_cv
    n_total = int(np.sum(bad_bins))

    if n_total > 0:
        logging.info(
            "Early masking: removing {} bins "
            "({} low-mean, {} high-CV)".format(n_total, n_low, n_cv)
        )
        masked_indices = np.where(mask)[0]
        mask[masked_indices[bad_bins]] = False
        masked_data = normalize_and_mask(samples, chrs, mask)

    return masked_data


# ---------------------------------------------------------------------------
# Normalization & PCA
# ---------------------------------------------------------------------------


def normalize_and_mask(samples, chrs, mask):
    by_chr = []
    sample_count = len(samples)

    for chr in chrs:
        max_len = max(
            [sample[str(chr)].shape[0] for sample in samples]
        )
        this_chr = np.zeros((max_len, sample_count), dtype=float)
        i = 0
        for sample in samples:
            this_chr[:, i] = sample[str(chr)]
            i += 1
        by_chr.append(this_chr)
    all_data = np.concatenate(by_chr, axis=0)

    sum_per_sample = np.sum(all_data, 0)
    all_data = all_data / sum_per_sample

    masked_data = all_data[mask, :]

    return masked_data


def train_pca(ref_data, pcacomp=5):
    t_data = ref_data.T
    pca = PCA(n_components=pcacomp)
    pca.fit(t_data)
    PCA(copy=True, whiten=False)
    transformed = pca.transform(t_data)
    inversed = pca.inverse_transform(transformed)
    corrected = t_data / inversed

    return corrected.T, pca


def reduce_dimensions(pca_corrected_data, n_components=30):
    """PCA dimensionality reduction for KNN distance computation.

    Reduces each bin's representation from n_samples to n_components
    dimensions. Returns float32 array for ANN library compatibility.
    """
    n_bins, n_samples = pca_corrected_data.shape
    actual = min(n_components, n_samples - 1, n_bins - 1)
    if actual < n_components:
        logging.info(
            "Reducing n_components from {} to {} "
            "(limited by data dimensions)".format(n_components, actual)
        )
    pca_reduce = PCA(n_components=actual)
    reduced = pca_reduce.fit_transform(pca_corrected_data)
    logging.info(
        "Dimensionality reduction: {} -> {} components "
        "(explained variance: {:.1f}%)".format(
            n_samples,
            actual,
            100.0 * np.sum(pca_reduce.explained_variance_ratio_),
        )
    )
    return reduced.astype(np.float32)


# ---------------------------------------------------------------------------
# KNN search backends
# ---------------------------------------------------------------------------


def _search_faiss(candidate_data, query_data, k):
    """KNN search using faiss. Returns (distances, indices)."""
    n_candidates, dim = candidate_data.shape
    candidate_f32 = np.ascontiguousarray(
        candidate_data, dtype=np.float32
    )
    query_f32 = np.ascontiguousarray(
        query_data, dtype=np.float32
    )

    if n_candidates > 500_000:
        nlist = int(np.sqrt(n_candidates))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(candidate_f32)
        index.add(candidate_f32)
        index.nprobe = min(50, max(1, nlist // 4))
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(candidate_f32)

    distances, indices = index.search(query_f32, k)
    return distances, indices.astype(np.int32)


def _search_hnswlib(candidate_data, query_data, k):
    """KNN search using hnswlib. Returns (distances, indices)."""
    n_candidates, dim = candidate_data.shape
    candidate_f32 = np.ascontiguousarray(
        candidate_data, dtype=np.float32
    )
    query_f32 = np.ascontiguousarray(
        query_data, dtype=np.float32
    )

    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(
        max_elements=n_candidates, ef_construction=200, M=32
    )
    index.add_items(candidate_f32)
    index.set_ef(max(k * 2, 100))

    indices, distances = index.knn_query(query_f32, k=k)
    return distances.astype(np.float64), indices.astype(np.int32)


def _search_sklearn(candidate_data, query_data, k):
    """KNN search using sklearn. Returns (sq distances, indices)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        n_neighbors=k, algorithm="auto", metric="euclidean"
    )
    nn.fit(candidate_data)
    distances, indices = nn.kneighbors(query_data)
    # Square distances for consistency with faiss/hnswlib (L2 squared)
    return (distances ** 2).astype(np.float64), indices.astype(
        np.int32
    )


def build_and_search_knn(candidate_data, query_data, k):
    """Dispatch KNN search to best available backend.

    Returns (distances, indices) where distances are squared L2.
    """
    if _KNN_BACKEND == "faiss":
        return _search_faiss(candidate_data, query_data, k)
    elif _KNN_BACKEND == "hnswlib":
        return _search_hnswlib(candidate_data, query_data, k)
    else:
        return _search_sklearn(candidate_data, query_data, k)


# ---------------------------------------------------------------------------
# Chromosome-level KNN orchestration
# ---------------------------------------------------------------------------


def knn_search_all_chromosomes(
    pca_reduced_data,
    masked_bins_per_chr,
    masked_bins_per_chr_cum,
    ref_size,
    chunk_size=50000,
):
    """Find KNN references for all bins, chromosome by chromosome.

    For each chromosome, candidate bins are all bins NOT on that
    chromosome (same logic as the original get_reference).

    Returns (indexes, distances) with identical semantics to the
    original code: indexes are into chr_data (the all-but-this-chr
    concatenation).
    """
    _log_knn_backend()

    total_bins = int(masked_bins_per_chr_cum[-1])
    all_indexes = np.zeros(
        (total_bins, ref_size), dtype=np.int32
    )
    all_distances = np.ones(
        (total_bins, ref_size), dtype=np.float64
    )

    n_chrs = len(masked_bins_per_chr)
    is_gonosomal = n_chrs > 22

    for chr_idx in range(n_chrs):
        chr_start = int(
            masked_bins_per_chr_cum[chr_idx]
            - masked_bins_per_chr[chr_idx]
        )
        chr_end = int(masked_bins_per_chr_cum[chr_idx])
        n_chr_bins = chr_end - chr_start

        if n_chr_bins == 0:
            continue

        # For gonosomal references, skip non-gonosomal chromosomes
        if is_gonosomal and chr_idx != 22 and chr_idx != 23:
            continue

        # Build candidate data: all bins NOT on this chromosome
        # Same concatenation as original code
        candidate_data = np.concatenate(
            (
                pca_reduced_data[:chr_start],
                pca_reduced_data[chr_end:],
            )
        )

        query_data = pca_reduced_data[chr_start:chr_end]

        # Process in chunks for memory efficiency
        for chunk_start in range(0, n_chr_bins, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_chr_bins)
            chunk_query = query_data[chunk_start:chunk_end]

            actual_k = min(ref_size, len(candidate_data))
            distances, indices = build_and_search_knn(
                candidate_data, chunk_query, actual_k
            )

            global_start = chr_start + chunk_start
            global_end = chr_start + chunk_end

            if actual_k < ref_size:
                all_indexes[global_start:global_end, :actual_k] = (
                    indices
                )
                all_distances[global_start:global_end, :actual_k] = (
                    distances
                )
            else:
                all_indexes[global_start:global_end] = indices
                all_distances[global_start:global_end] = distances

        logging.info(
            "KNN complete for chr {} ({} bins)".format(
                chr_idx + 1, n_chr_bins
            )
        )

    return all_indexes, all_distances


# ---------------------------------------------------------------------------
# Null ratio computation (vectorized)
# ---------------------------------------------------------------------------


def compute_null_ratios_chunk(
    pca_corrected_data, indexes, start, end
):
    """Compute null ratios for bins [start, end). Vectorized."""
    n_samples = pca_corrected_data.shape[1]
    n_null = min(n_samples, 100)
    chunk_indexes = indexes[start:end]
    null_ratios = np.zeros((end - start, n_null))

    samples_t = pca_corrected_data.T  # (n_samples, n_bins)
    sample_indices = random.sample(range(n_samples), n_null)

    for null_i, case_i in enumerate(sample_indices):
        sample = samples_t[case_i]
        bin_values = sample[start:end]
        ref_values = sample[chunk_indexes]
        medians = np.median(ref_values, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            null_ratios[:, null_i] = np.log2(
                bin_values / medians
            )

    return null_ratios


def compute_null_ratios_parallel(
    pca_corrected_data, indexes, cpus=1, chunk_size=50000
):
    """Compute null ratios in parallel chunks using joblib."""
    from joblib import Parallel, delayed

    total_bins = len(indexes)
    chunks = []
    for start in range(0, total_bins, chunk_size):
        end = min(start + chunk_size, total_bins)
        chunks.append((start, end))

    logging.info(
        "Computing null ratios: {} chunks, {} cpus".format(
            len(chunks), cpus
        )
    )

    if cpus == 1:
        results = [
            compute_null_ratios_chunk(
                pca_corrected_data, indexes, start, end
            )
            for start, end in chunks
        ]
    else:
        results = Parallel(n_jobs=cpus)(
            delayed(compute_null_ratios_chunk)(
                pca_corrected_data, indexes, start, end
            )
            for start, end in chunks
        )

    return np.concatenate(results, axis=0)

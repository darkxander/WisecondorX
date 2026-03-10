# WisecondorX

import logging
import os

import numpy as np

from wisecondorx.newref_tools import (
    apply_early_masking,
    compute_null_ratios_parallel,
    knn_search_all_chromosomes,
    normalize_and_mask,
    reduce_dimensions,
    train_pca,
)


def tool_newref_prep(args, samples, gender, mask, bins_per_chr):
    """Prepare reference data: normalize, mask, PCA, dim reduction.

    Creates:
      - prepfile: metadata .npz
      - prepdatafile: full PCA-corrected data .npy (for null ratios)
      - prepreducedfile: reduced data .npy (for KNN distances)
    """
    if gender == "A":
        last_chr = 22
    elif gender == "F":
        last_chr = 23
    else:
        last_chr = 24

    bins_per_chr = bins_per_chr[:last_chr]
    mask = mask[: np.sum(bins_per_chr)]
    chrs = range(1, last_chr + 1)

    masked_data = normalize_and_mask(samples, chrs, mask)

    # Early bin masking: remove low-mean and high-CV bins
    masked_data = apply_early_masking(
        masked_data, mask, samples, chrs
    )

    pca_corrected_data, pca = train_pca(
        masked_data, pcacomp=args.pcacomp
    )

    # PCA Distance filtering: remove bins far from median profile
    med_prof = np.median(pca_corrected_data, axis=0)
    dist_to_med = np.sum(
        (pca_corrected_data - med_prof) ** 2, axis=1
    )
    mad = np.median(
        np.abs(dist_to_med - np.median(dist_to_med))
    )
    cutoff = max(np.median(dist_to_med) + 10 * mad, 5.0)
    bad_bins_mask = dist_to_med > cutoff

    if np.any(bad_bins_mask):
        n_removed = np.sum(bad_bins_mask)
        logging.info(
            "Removing {} anomalous bins based on PCA distance "
            "(cutoff={:.4f})".format(n_removed, cutoff)
        )
        masked_indices = np.where(mask)[0]
        global_bad_indices = masked_indices[bad_bins_mask]
        mask[global_bad_indices] = False

        masked_data = normalize_and_mask(samples, chrs, mask)
        pca_corrected_data, pca = train_pca(
            masked_data, pcacomp=args.pcacomp
        )

    # Dimensionality reduction for KNN
    pca_reduced_data = reduce_dimensions(
        pca_corrected_data, n_components=args.n_components
    )

    masked_bins_per_chr = [
        sum(
            mask[
                sum(bins_per_chr[:i]) : sum(bins_per_chr[:i]) + x
            ]
        )
        for i, x in enumerate(bins_per_chr)
    ]
    masked_bins_per_chr_cum = [
        sum(masked_bins_per_chr[: x + 1])
        for x in range(len(masked_bins_per_chr))
    ]

    np.save(args.prepdatafile, pca_corrected_data)
    np.save(args.prepreducedfile, pca_reduced_data)

    np.savez_compressed(
        args.prepfile,
        binsize=args.binsize,
        gender=gender,
        mask=mask,
        bins_per_chr=bins_per_chr,
        masked_bins_per_chr=masked_bins_per_chr,
        masked_bins_per_chr_cum=masked_bins_per_chr_cum,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
    )


def tool_newref_main(args, cpus):
    """Run KNN search and null ratio computation."""
    pca_corrected_data = np.load(
        args.prepdatafile, mmap_mode="r"
    )
    pca_reduced_data = np.load(
        args.prepreducedfile, mmap_mode="r"
    )
    npzdata = np.load(
        args.prepfile, encoding="latin1", allow_pickle=True
    )
    masked_bins_per_chr = npzdata["masked_bins_per_chr"]
    masked_bins_per_chr_cum = npzdata["masked_bins_per_chr_cum"]

    # KNN search using ANN backend
    indexes, distances = knn_search_all_chromosomes(
        pca_reduced_data,
        masked_bins_per_chr,
        masked_bins_per_chr_cum,
        ref_size=args.refsize,
        chunk_size=args.chunk_size,
    )

    # Null ratios — vectorized, parallelized over chunks
    null_ratios = compute_null_ratios_parallel(
        pca_corrected_data,
        indexes,
        cpus=cpus,
        chunk_size=args.chunk_size,
    )

    # Save results directly (no intermediate part files)
    np.savez_compressed(
        args.tmpoutfile,
        binsize=npzdata["binsize"].item(),
        gender=npzdata["gender"].item(),
        mask=npzdata["mask"],
        bins_per_chr=npzdata["bins_per_chr"],
        masked_bins_per_chr=npzdata["masked_bins_per_chr"],
        masked_bins_per_chr_cum=npzdata["masked_bins_per_chr_cum"],
        pca_components=npzdata["pca_components"],
        pca_mean=npzdata["pca_mean"],
        indexes=indexes,
        distances=distances,
        null_ratios=null_ratios,
    )

    # Cleanup prep files
    os.remove(args.prepfile)
    os.remove(args.prepdatafile)
    os.remove(args.prepreducedfile)


def tool_newref_merge(args, outfiles, trained_cutoff):
    """Merge gender-specific references into final output."""
    final_ref = {"has_female": False, "has_male": False}
    for file_id in outfiles:
        npz_file = np.load(
            file_id, encoding="latin1", allow_pickle=True
        )
        gender = str(npz_file["gender"])
        for component in [
            x for x in npz_file.keys() if x != "gender"
        ]:
            if gender == "F":
                final_ref["has_female"] = True
                final_ref["{}.F".format(str(component))] = (
                    npz_file[component]
                )
            elif gender == "M":
                final_ref["has_male"] = True
                final_ref["{}.M".format(str(component))] = (
                    npz_file[component]
                )
            else:
                final_ref[str(component)] = npz_file[component]
        os.remove(file_id)
    final_ref["is_nipt"] = args.nipt
    final_ref["trained_cutoff"] = trained_cutoff
    np.savez_compressed(args.outfile, **final_ref)

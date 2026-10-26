"""Weighted statistics of particle quantities.
   (Not named 'statistics.py' to avoid shadowing the Python standard-library module.)"""

import numpy as np


def weighted_percentiles(values, weights, percentiles=(16, 50, 84)):
    """Percentiles of values in which each element counts proportionally to its weight."""

    ok = np.isfinite(values) & (weights > 0)
    values, weights = values[ok], weights[ok]
    order = np.argsort(values)                    # indices that sort values
    cum_w = np.cumsum(weights[order])             # cumulative weight along the sorted values
    cum_w /= cum_w[-1]                            # normalised to [0, 1]
    return np.interp(np.asarray(percentiles) / 100, cum_w, values[order])


def weighted_mean_std(values, weights):
    """Weighted mean and standard deviation of the finite values:
       mean = sum w q / sum w,   std = sqrt( sum w (q - mean)^2 / sum w ).
       For a log quantity (e.g. [O/H]) they are the mean of the log and the scatter in dex."""

    ok = np.isfinite(values) & (weights > 0)
    v, w = values[ok], weights[ok]
    mean = np.sum(w * v) / np.sum(w)
    std  = np.sqrt(np.sum(w * (v - mean)**2) / np.sum(w))
    return mean, std


def print_weighted_statistics(quantities, weights):
    """quantities, weights: {label: float64 numpy array}.
       For each weight and quantity print the 16th, 50th (median), 84th weighted percentiles,
       the weighted mean and standard deviation, and f_fin = weight fraction with a finite value."""

    w = max(len(q) for q in quantities)
    print(f"\n[+] Weighted statistics of particle quantities:")
    for w_label, w_values in weights.items():
        print(f"\n     weight = {w_label}")
        print(f"     {'quantity':<{w}}  {'16th':>10} {'50th':>10} {'84th':>10} {'mean':>10} {'std':>10} {'f_fin':>7}")
        print("     " + "-"*(w + 65))
        for q_label, q_values in quantities.items():
            ok    = np.isfinite(q_values) & (w_values > 0)
            f_fin = w_values[ok].sum() / w_values.sum()      # weight fraction with a finite value (e.g. enriched gas)
            if not ok.any():
                print(f"     {q_label:<{w}}  {'no finite values':>32}")
                continue
            p16, p50, p84 = weighted_percentiles(q_values, w_values)
            mean, std     = weighted_mean_std(q_values, w_values)
            print(f"     {q_label:<{w}}  {p16:10.3e} {p50:10.3e} {p84:10.3e} {mean:10.3e} {std:10.3e} {f_fin:7.3f}")


def binned_percentiles(x, y, weights, x_bins, percentiles=(16, 50, 84)):
    """Weighted percentiles of y in bins of x. Returns bin centres and an array (n_bins, n_percentiles);
       empty bins are nan."""
    centres = 0.5 * (x_bins[1:] + x_bins[:-1])
    out = np.full((len(centres), len(percentiles)), np.nan)
    idx = np.digitize(x, x_bins) - 1
    for i in range(len(centres)):
        sel = idx == i
        if np.any(sel & np.isfinite(y)):
            out[i] = weighted_percentiles(y[sel], weights[sel], percentiles)
    return centres, out

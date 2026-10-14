"""Generic plots: phase diagrams, distributions, 2D histograms, abundance maps."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from hydrogen import N_HI_LLS, N_HI_DLA, ABSORBER_CLASSES


def plot_phase_diagram(ax, n_H, T, weights, cbar_label, bins=200, ranges=((-8, 4), (1, 9))):
    """log n_H - log T 2D histogram: each cell is the fraction of the total weight it contains."""

    hist, xedges, yedges = np.histogram2d(np.log10(n_H), np.log10(T), bins=bins,
                                          range=ranges, weights=weights)
    hist /= weights.sum()   # normalization of histo to 1

    im = ax.pcolormesh(xedges, yedges, hist.T, norm=LogNorm(), cmap='viridis')   # .T: hist is [ix, iy] as for the maps
    ax.set_xlabel(r'$\log_{10} \ n_{\rm H} \ [\rm cm^{-3}]$')
    ax.set_ylabel(r'$\log_{10} \ T \ [\rm K]$')
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)

    return im


def plot_column_density_pixels(ax, N_HI_map, bins=np.arange(12, 23.01, 0.1)):
    """Distribution of log N_HI over the map pixels (NOT a CDDF), with LLS and DLA thresholds.
       Returns the fraction of pixels above the LLS and DLA thresholds."""

    N_HI = N_HI_map.to("cm**-2").value.ravel()
    with np.errstate(divide="ignore"):   # for log10(0)
        logN = np.log10(N_HI)

    ax.hist(logN[np.isfinite(logN)], bins=bins, histtype='step', color='k', density=True)
    ax.axvline(np.log10(N_HI_LLS), color='tab:blue', ls='--', label='LLS')
    ax.axvline(np.log10(N_HI_DLA), color='tab:red',  ls='--', label='DLA')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10} \ N_{\rm HI} \ [\rm cm^{-2}]$')
    ax.set_ylabel('Pixel PDF')
    ax.legend()

    return np.mean(N_HI >= N_HI_LLS), np.mean(N_HI >= N_HI_DLA)  # covering LLS and DLA fraction


def plot_abundance_map(image, ax, Lbox, cbar_label, vmin=-4, vmax=0, cmap="RdYlBu_r"):
    """Draw a 2D abundance map [X/Y] (numpy array indexed [ix, iy]) in linear scale.
       Pristine pixels (non-finite values) are shown in grey."""

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("lightgrey")
    image = np.where(np.isfinite(image), image, np.nan)   # -inf (pristine) -> nan -> 'bad' colour

    im = ax.imshow(image.T, origin='lower', extent=[0, Lbox[0], 0, Lbox[1]], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_box_aspect(1)
    ax.set_xlabel(r'$\rm X \ [ cMpc ]$')
    ax.set_ylabel(r'$\rm Y \ [ cMpc ]$')
    cbar = ax.figure.colorbar(mappable=im, ax=ax, orientation='vertical', pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)

    return im


def plot_class_histograms(ax, logN_HI, image, xlabel, bins=np.arange(-6, 1.01, 0.1)):
    """Normalised pixel distribution of a map (e.g. [O/H]_HI) for each optically thick N_HI class.
       Only enriched (finite) pixels enter the histogram. Values outside the bin range are piled up
       in the first/last bin (display only); the fraction below the range is given in the legend."""

    for name in ["LLS", "subDLA", "DLA"]:
        lo, hi = ABSORBER_CLASSES[name]
        sel = (logN_HI >= lo) & (logN_HI < hi)
        v = image[sel]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        f_below = np.mean(v < bins[0])
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype='step', density=True, lw=1.5,
                label=f"{name}: {sel.sum()} pix, below {bins[0]:.0f}: {f_below:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Pixel PDF')
    ax.legend(fontsize=9)


def plot_weighted_pdf(ax, values, weights, bins, xlabel):
    """Distribution of a particle quantity for several weights (dict {label: array}).
       Each curve is normalised over the particles with a finite value; values outside the bins are
       piled up in the first/last bin (display only) and the fraction below the range is in the legend."""

    for w_label, w_values in weights.items():
        ok = np.isfinite(values) & (w_values > 0)
        v, w = values[ok], w_values[ok]
        f_below = w[v < bins[0]].sum() / w.sum()
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, weights=w, histtype='step', density=True, lw=1.5,
                label=f"{w_label} (below {bins[0]:.0f}: {f_below:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel('PDF')
    ax.legend(fontsize=9)


def plot_hist2d(ax, x, y, x_bins, y_bins, xlabel, ylabel, weights=None, conditional=False):
    """Weighted 2D histogram of (x, y).
       conditional=False: each cell = fraction of the total weight (joint distribution, like the phase diagram).
       conditional=True : each column normalised to 1, i.e. P(y | x); white line = median of y per column.
       y = -inf (pristine) is piled up in the lowest y bin (display only)."""

    w  = np.ones_like(x) if weights is None else weights
    ok = np.isfinite(x) & ~np.isnan(y) & (w > 0)
    x, y, w = x[ok], np.clip(y[ok], y_bins[0], y_bins[-1]), w[ok]

    hist, xe, ye = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=w)
    with np.errstate(divide="ignore", invalid="ignore"):
        if conditional:
            hist = hist / hist.sum(axis=1, keepdims=True)   # normalise each x column
        else:
            hist = hist / hist.sum()
    im = ax.pcolormesh(xe, ye, hist.T, norm=LogNorm(vmin=1e-4 if conditional else None), cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label(r'$P(y\,|\,x)$ per column' if conditional else 'Weight fraction', fontsize=11)

    if conditional:                                         # median of y in each populated x column
        centres = 0.5 * (xe[1:] + xe[:-1])
        idx     = np.digitize(x, xe) - 1                    # x column of each point
        medians = [np.median(y[idx == i]) if np.any(idx == i) else np.nan for i in range(len(centres))]
        ax.plot(centres, medians, color='white', lw=2, label='median')
        ax.legend(fontsize=9, loc='upper left')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im

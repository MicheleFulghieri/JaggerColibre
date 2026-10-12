import argparse
import os
import sys
from datetime import datetime
from operator import attrgetter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import h5py
import unyt
import astropy.units as u
from astropy import constants as const 
import swiftsimio as sw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_SNAP_DIR = "/cosma8/data/dp004/colibre/Runs"
DEFAULT_SAVE_DIR = "/cosma8/data/do019/dc-fulg1/outputs/swiftsimio_analysis"

def parse_args():
    """Command-line options: which run/snapshot to analyse and where to save."""
    p = argparse.ArgumentParser(description="Analyse a COLIBRE snapshot for JAGGER")
    p.add_argument("--run", default="L0025N0188/Thermal",   # default light
                   help="run directory relative to --snap-dir (e.g. L0025N0188/Thermal)")
    p.add_argument("--snap", type=int, required=True,
                   help="snapshot number (L025_m5: 44 z=5.5, 48 z=5.0, 52 z=4.5, 56 z=4.0)")
    p.add_argument("--snap-dir", default=DEFAULT_SNAP_DIR, help="root of the COLIBRE runs")
    p.add_argument("--save", default=DEFAULT_SAVE_DIR, help="root output directory")
    return p.parse_args()



# ---------------------------------------------------------------------------
# Useful values
# ---------------------------------------------------------------------------
hydrogen_mass = unyt.mh.to("g")    # hydrogen atom mass (unyt constant) 1.674e-24 g
Z_sun         = 0.0134             # solar metal mass fraction, Asplund et al. (2009)
N_HI_LLS      = 10**17.2           # cm^-2, Lyman limit system threshold (tau_912 = 1, e.g. Fumagalli et al, 2013)
N_HI_DLA      = 2e20               # cm^-2, damped Lyman-alpha threshold (Wolfe et al. 1986, ApJS 61, 249)


# ---------------------------------------------------------------------------
# USEFUL FUNCTIONS AND CLASSES
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------
# One sub-folder per topic, numbered in the order of the analysis
FIG_DIRS = {
    "gas"        : "01_Gas",                               # full-box maps, phase diagrams
    "HI"         : "02_HI",                                # slab HI maps, N_HI distribution
    "metal_maps" : os.path.join("03_Metals", "Maps"),      # slab column and abundance maps
    "metal_diag" : os.path.join("03_Metals", "Diagnostics"),   # distributions, absorber classes
}

def make_dirs(base):
    for d in FIG_DIRS.values():
        os.makedirs(os.path.join(base, d), exist_ok=True)

def fig_path(base, topic, filename):
    """Full path of a figure: base / FIG_DIRS[topic] / filename."""
    return os.path.join(base, FIG_DIRS[topic], filename)

def save_figure(fig, base, topic, filename, dpi=300):
    """Save fig in base / FIG_DIRS[topic] / filename, close it and print the path in the log."""
    path = fig_path(base, topic, filename)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"     [saved] {path}")


# ---------------------------------------------------------------------------
# Gas fields inspection (metadata only)
# ---------------------------------------------------------------------------
def print_gas_fields(meta):
    """Print in alphabetical order the gas fields available in the snapshot
       (swiftsimio name and HDF5 dataset) and, for the fields stored as named
       columns, the names of their sub-fields. Reads metadata only."""

    gas_props  = meta.gas_properties
    named_cols = gas_props.named_columns  
    fields     = sorted(zip(gas_props.field_names, gas_props.field_paths))
    w = max(len(n) for n, _ in fields)    # space printing according to the swiftsimio name length

    print(f"\n[+] Available gas fields ({len(fields)}):")
    print(f"     {'#':>3}  {'swiftsimio name':<{w}} {'HDF5 dataset'}")
    print("     " + "-"*95)
    for i, (name, path) in enumerate(fields, start=1):
        print(f"     {i:>3}  {name:<{w}} {path}")
        columns = named_cols.get(path)
        if columns:
            print(f"     {'':>3}    -> columns: {', '.join(columns)}")


# ---------------------------------------------------------------------------
# Gas Projection Maps
# ---------------------------------------------------------------------------
def compute_projection_map(snap, field="masses", weight=None, resolution=256, region=None,
                           weight_map=None, physical=True, parallel=False, periodic=True):
    """2D projection along z of a gas field, smoothed with the SPH kernel.

       weight=None  -> surface density of field in each pixel i:
                       Sigma_i = sum_j q_j W_ij / A_pix               [q / length^2]
       weight='w'   -> w-weighted average of field in each pixel i:
                       <q>_i = sum_j q_j w_j W_ij / sum_j w_j W_ij     [q]

       field and weight are swiftsimio names of snap.gas (dotted names allowed,
       e.g. 'element_mass_fractions.oxygen').
       region     : [x_min, x_max, y_min, y_max, (z_min, z_max)] cosmo_array, None = whole box.
       weight_map : already computed projection of weight (comoving, same resolution and
                    region), used as denominator instead of projecting weight again.
       Returns a cosmo_array (resolution x resolution) indexed as [ix, iy]:
       transpose it before imshow."""

    project_kwargs = dict(resolution=resolution, region=region, parallel=parallel, periodic=periodic)

    if weight is None:
        proj_map = sw.visualisation.projection.project_gas(snap, project=field, **project_kwargs)  # **kwargs  to allow functions to accept a unknown number of arguments
    else:
        # Temporary particle field q*w attached to snap.gas, so that project_gas can find it by name
        tmp_name = f"_tmp_{field}_x_{weight}".replace(".", "_")
        setattr(snap.gas, tmp_name, attrgetter(field)(snap.gas) * attrgetter(weight)(snap.gas))
        try:
            num = sw.visualisation.projection.project_gas(snap, project=tmp_name, **project_kwargs)
            if weight_map is None:
                den = sw.visualisation.projection.project_gas(snap, project=weight, **project_kwargs)
            else:
                den = weight_map
        finally:
            delattr(snap.gas, tmp_name)                       # free the memory of the temporary field
        with np.errstate(divide="ignore", invalid="ignore"):  # no nan due to division by 0 errors
            proj_map = num / den                              # empty pixels (den = 0) -> nan

    if physical:  # converting from coomoving to physical coords
        proj_map = proj_map.to_physical()

    return proj_map


def plot_projection_map(proj_map, ax, Lbox, units, cbar_label, cmap="magma", log=True,
                        vmin=None, vmax=None, title=None):
    """Draw on ax a map returned by compute_projection_map, converted to units.
       Lbox is the comoving box size in cMpc. Returns the image object."""

    image = proj_map.to(units).value.T       # .T since in proj_map rows are x, cols y: imshow wants rows = vertical ax

    im = ax.imshow(image,
                   origin = 'lower',
                   extent = [0, Lbox[0], 0, Lbox[1]],
                   cmap   = cmap,
                   norm   = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax))

    ax.set_box_aspect(1)
    ax.set_xlabel(r'$\rm X \ [ cMpc ]$')
    ax.set_ylabel(r'$\rm Y \ [ cMpc ]$')
    if title is not None:
        ax.set_title(title, fontsize=11)

    cbar = ax.figure.colorbar(mappable=im, ax=ax, orientation='vertical', pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)

    return im


# ---------------------------------------------------------------------------
# Neutral hydrogen
# ---------------------------------------------------------------------------
def compute_hydrogen_fields(gas):
    """Hydrogen quantities of each gas particle (comoving, as stored in the snapshot):
       m_HI = f_HI * X_H * m          [Msun]   HI mass: weight for the absorber diagnostics
       n_H  = X_H * rho / m_H         [cm^-3]  total hydrogen number density
       f_HI = n_HI / n_H (CHIMES species fraction), X_H = hydrogen mass fraction.
       (n_HI = f_HI * n_H, if needed.)"""

    X_H  = gas.element_mass_fractions.hydrogen
    f_HI = gas.species_fractions.HI

    m_HI = (f_HI * X_H * gas.masses).to("Msun")       # Msun, not g: in g the values would overflow float32
    rho  = gas.densities.to("g/cm**3")                # convert first: in g/Mpc**3 the values would overflow float32
    n_H  = (X_H * rho / hydrogen_mass).to("cm**-3")
    return m_HI, n_H


def print_HI_budget(gas, m_HI, cosmo, boxsize):
    """Global gas and HI budget of the box. Omega_HI = rho_HI,comoving / rho_crit,0."""

    M_gas = gas.masses.to("Msun").value.sum(dtype=np.float64)              # accumulate in float64
    M_H   = (gas.element_mass_fractions.hydrogen * gas.masses).to("Msun").value.sum(dtype=np.float64)
    M_HI  = m_HI.to("Msun").value.sum(dtype=np.float64)

    V_box     = np.prod(boxsize.to("Mpc").value)                           # comoving volume [cMpc^3]
    rho_crit0 = cosmo.critical_density0.to(u.Msun / u.Mpc**3).value        # astropy quantity -> float [Msun/Mpc^3]
    Omega_HI  = M_HI / V_box / rho_crit0

    print(f"\n[+] Global HI budget:")
    print(f"     Total gas mass                 :     {M_gas:.3e} Msun")
    print(f"     Total hydrogen mass            :     {M_H:.3e} Msun")
    print(f"     Total HI mass                  :     {M_HI:.3e} Msun")
    print(f"     Neutral fraction M_HI / M_H    :     {M_HI / M_H:.3e}")
    print(f"     Omega_HI                       :     {Omega_HI:.3e}")


def slab_absorption_distance(thickness_cMpc, cosmo, z):
    # Relation in Fumagalli et al. (2013), from Bahcall & Peebles (1969)
    """Absorption distance of a slab of comoving thickness Delta_chi at redshift z:
       dX = (1+z)^2 H0/H(z) dz  and  dz = H(z) dchi / c   ->   Delta_X = (1+z)^2 H0 Delta_chi / c"""
    delta_chi = thickness_cMpc * u.Mpc                       # float -> astropy Quantity
    # .decompose() simplify the units
    return ((1 + z)**2 * cosmo.H0 * delta_chi / const.c).decompose().value


# ---------------------------------------------------------------------------
# Chemical and metal inspection
# ---------------------------------------------------------------------------
# Solar photospheric abundances 12 + log10(n_X/n_H), Asplund et al. (2009), Table 1
SOLAR_LOG_EPS = {"hydrogen": 12.00, "helium": 10.93, "carbon": 8.43, "nitrogen": 7.83, "oxygen": 8.69,
                "neon": 7.93, "magnesium": 7.60, "silicon": 7.51, "iron": 7.50}
# Standard atomic weights [amu]
ATOMIC_MASS = {"hydrogen": 1.008, "helium": 4.003, "carbon": 12.011, "nitrogen": 14.007, "oxygen": 15.999,
            "neon": 20.180, "magnesium": 24.305, "silicon": 28.085, "iron": 55.845}
METALS = ["carbon", "nitrogen", "oxygen", "neon", "magnesium", "silicon", "iron"]

def print_metal_budget(gas, snap_filename):
    """Per metal: total mass, dust fraction, closure X_tot = X_diffuse + X_dust, enriched mass fraction.
    X_dust,el = sum_g D_g * M[g, el], M = GrainToElementMapping (grain -> element mass fractions)."""
    with h5py.File(snap_filename, "r") as f:
        grain_to_element = f["SubgridScheme/GrainToElementMapping"][:]            # (6 grains, 12 elements)
    elements = gas.element_mass_fractions.named_columns                          # same order as the matrix columns
    D = np.column_stack([getattr(gas.dust_mass_fractions, g).value.astype(np.float64)
                        for g in gas.dust_mass_fractions.named_columns])        # (N particles, 6 grains)
    m = gas.masses.to("Msun").value.astype(np.float64)

    print(f"\n[+] Metal budget (mass weighted)")
    print(f"     {'element':<10} {'M_el [Msun]':>12} {'f_dust':>9} {'closure':>9} {'f_enriched':>11}")
    for i, el in enumerate(elements):
        if el not in METALS:
            continue
        X_tot  = getattr(gas.element_mass_fractions, el).value.astype(np.float64)
        X_dif  = getattr(gas.element_mass_fractions_diffuse, el).value.astype(np.float64)
        X_dust = D @ grain_to_element[:, i]                                        # matrix-vector product
        M_tot  = np.sum(m * X_tot)
        print(f"     {el:<10} {M_tot:12.3e} {np.sum(m * X_dust) / M_tot:9.3e} "
            f"{np.sum(m * (X_dif + X_dust)) / M_tot:9.4f} {np.sum(m[X_tot > 0]) / np.sum(m):11.3e}")

    Z = gas.metal_mass_fractions.value.astype(np.float64)
    M_Z = np.sum(m * Z)
    for label, field in [("SNII", gas.metal_mass_fractions_from_snii), ("SNIa", gas.metal_mass_fractions_from_snia),
                        ("AGB", gas.metal_mass_fractions_from_agb)]:
        print(f"     Metal mass from {label:<5}         :     {np.sum(m * field.value.astype(np.float64)) / M_Z:.3e}")

# Abundances per particles
def number_ratio(gas, element, diffuse=False):
    """n_X / n_H of each particle (float64). diffuse=False: gas + dust; diffuse=True: gas phase only."""
    fractions = gas.element_mass_fractions_diffuse if diffuse else gas.element_mass_fractions
    X_el = getattr(fractions, element).value.astype(np.float64)
    X_H  = gas.element_mass_fractions.hydrogen.value.astype(np.float64)
    return (X_el / ATOMIC_MASS[element]) / (X_H / ATOMIC_MASS["hydrogen"])

def bracket(ratio, element, reference="hydrogen"):
    """[X/Y] = log10(n_X/n_Y) - log10(n_X/n_Y)_sun. Pristine gas (ratio = 0) -> -inf, 0/0 -> nan."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(ratio) - (SOLAR_LOG_EPS[element] - SOLAR_LOG_EPS[reference])

def depletion(gas, element):
    """delta_X = log10(X_diffuse / X_total) of each particle: 0 = no dust, -1 = 90% of X locked in grains.
       Pristine gas (0/0) -> nan."""
    X_tot = getattr(gas.element_mass_fractions, element).value.astype(np.float64)
    X_dif = getattr(gas.element_mass_fractions_diffuse, element).value.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(X_dif / X_tot)

def element_column_density(snap, element, diffuse=True, neutral=False, **slab_kwargs):
    """Column density N_X [cm^-2, physical] of an element in the slab.
       neutral=True: only the X atoms associated with HI (m_X * f_HI). For oxygen this is the OI proxy
       (charge exchange keeps x_OI ~ x_HI); for other elements it is NOT an ion column."""
    fractions = snap.gas.element_mass_fractions_diffuse if diffuse else snap.gas.element_mass_fractions
    mass_X = getattr(fractions, element) * snap.gas.masses
    if neutral:
        mass_X = mass_X * snap.gas.species_fractions.HI
    tmp_name = f"_tmp_m_{element}"
    setattr(snap.gas, tmp_name, mass_X)
    try:
        sigma = compute_projection_map(snap, field=tmp_name, weight=None, physical=False, **slab_kwargs)
    finally:
        delattr(snap.gas, tmp_name)
    return (sigma.to_physical().to("g/cm**2") / (ATOMIC_MASS[element] * unyt.amu)).to("cm**-2")


# ---------------------------------------------------------------------------
# Weighted statistics of particle quantities
# ---------------------------------------------------------------------------
def weighted_percentiles(values, weights, percentiles=(16, 50, 84)):
    """Percentiles of values in which each element counts proportionally to its weight."""

    ok = np.isfinite(values) & (weights > 0)
    values, weights = values[ok], weights[ok]
    order = np.argsort(values)                    # indices that sort values
    cum_w = np.cumsum(weights[order])             # cumulative weight along the sorted values
    cum_w /= cum_w[-1]                            # normalised to [0, 1]
    return np.interp(np.asarray(percentiles) / 100, cum_w, values[order])


def print_weighted_statistics(quantities, weights):
    """quantities, weights: {label: float64 numpy array}.
       Print the 16th, 50th, 84th weighted percentiles of each quantity, for each weight."""

    w = max(len(q) for q in quantities)
    print(f"\n[+] Weighted percentiles of particle quantities:")
    for w_label, w_values in weights.items():
        print(f"\n     weight = {w_label}")
        print(f"     {'quantity':<{w}}  {'16th':>10} {'50th':>10} {'84th':>10} {'f_fin':>7}")
        print("     " + "-"*(w + 43))
        for q_label, q_values in quantities.items():
            ok    = np.isfinite(q_values) & (w_values > 0)
            f_fin = w_values[ok].sum() / w_values.sum()      # weight fraction with a finite value (e.g. enriched gas)
            if not ok.any():
                print(f"     {q_label:<{w}}  {'no finite values':>32}")
                continue
            p16, p50, p84 = weighted_percentiles(q_values, w_values)
            print(f"     {q_label:<{w}}  {p16:10.3e} {p50:10.3e} {p84:10.3e} {f_fin:7.3f}")


ABSORBER_CLASSES = {"forest": (-np.inf, 17.2), "LLS": (17.2, 19.0), "subDLA": (19.0, 20.3), "DLA": (20.3, np.inf)}

def print_absorber_classes(logN_HI, maps):
    """Covering fraction and 16/50/84th percentiles of each map (e.g. [O/H]_HI, [Si/O]) per N_HI class.
       maps: {label: 2D numpy array}, same shape as logN_HI. Non-finite pixels (pristine) are excluded and counted."""
    print(f"\n[+] Absorber classes (pixel statistics)")
    for name, (lo, hi) in ABSORBER_CLASSES.items():
        sel = (logN_HI >= lo) & (logN_HI < hi)
        print(f"\n     {name}: covering fraction = {sel.mean():.3e}")
        for label, m in maps.items():
            v = m[sel]
            v = v[np.isfinite(v)]
            if v.size == 0:
                print(f"       {label:<12} no enriched pixels")
                continue
            p16, p50, p84 = np.percentile(v, [16, 50, 84])
            print(f"       {label:<12} {p16:8.2f} {p50:8.2f} {p84:8.2f}   (enriched: {v.size / sel.sum():.2f})")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
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



# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # ---------------------------------------------------------------------------
    # Directory setting up
    # ---------------------------------------------------------------------------
    args = parse_args()
    snap_filename = os.path.join(args.snap_dir, args.run, "SOAP-HBT",
                                f"colibre_with_SOAP_membership_{args.snap:04d}.hdf5")
    if not os.path.isfile(snap_filename):
        sys.exit(f"Snapshot not found: {snap_filename}")      # message on stderr, exit code 1
    save_path = os.path.join(args.save, args.run.replace("/", "_"), f"snap_{args.snap:04d}")
    make_dirs(save_path)

    # ---------------------------------------------------------------------------
    # Virtual snapshot load and metadata
    # ---------------------------------------------------------------------------
    snap = sw.load(snap_filename)
    meta = snap.metadata                       # metadata method
    L_box    = meta.boxsize.to('Mpc').value
    cosmo    = meta.cosmology                  # astropy w0waCDM object
    H_z = meta.cosmology.H(meta.z)             # Hubble parameter H(z_snap) km/s/Mpc
    redshift = meta.z
    time     = meta.t.to('Myr').value
    res      = 1024                            # pixels per side of the maps

    

    print(f"\n----------- Inspecting the Colibre snapshot {args.run} -----------")
    print(f"     Snapshot file                  :     {snap_filename}")
    print(f"     Figures saved in               :     {save_path}")
    print(f"     Available groups               :     {snap}")
    print(f"     Box size                       :     {meta.boxsize}")
    print(f"     Cosmic Time                    :     {time:.3f} Myr")
    print(f"     Redshift (scale factor)        :     {meta.z:.2f} ({meta.a:.2f})")
    print(f"     Hubble parameter H(z)          :     {H_z:.2f}")
    print(f"     Number of Gas particles        :     {meta.n_gas}")
    print(f"     Number of DM particles         :     {meta.n_dark_matter}")
    print(f"     Number of stellar particles    :     {meta.n_stars}")
    print(f"     Number of black hole particles :     {meta.n_black_holes}")


    # ---------------------------------------------------------------------------
    # Gas Mapping analysis
    # ---------------------------------------------------------------------------
    print_gas_fields(meta)

    # ---- Gas surface density (weight=None) ----
    sigma_gas = compute_projection_map(snap, field="masses", weight=None, resolution=res, parallel=True)
    fig, ax = plt.subplots()
    plot_projection_map(sigma_gas, ax, Lbox=L_box, units='Msun/kpc**2', cbar_label=r'$\Sigma_{\rm gas} \ [\rm M_\odot \ pkpc^{-2}]$')
    save_figure(fig, save_path, "gas", "Gas_xy2d_mass_projection.png")


    # ---- Mass-weighted gas temperature ----
    T_gas = compute_projection_map(snap, field="temperatures", weight="masses", resolution=res, parallel=True)
    fig, ax = plt.subplots()
    plot_projection_map(T_gas, ax, Lbox=L_box, units='K', cbar_label=r'$\langle T \rangle_{\rm mass} \ [\rm K]$', cmap='inferno')
    save_figure(fig, save_path, "gas", "Gas_xy2d_temperature_projection.png")


    # ---------------------------------------------------------------------------
    # Neutral hydrogen analysis
    # ---------------------------------------------------------------------------
    m_HI, n_H = compute_hydrogen_fields(snap.gas)

    # new particle fields attached to snap.gas (compute_projection_map needs them)
    snap.gas.m_HI        = m_HI       # absorber mass, useful weight
    snap.gas.n_H         = n_H
    snap.gas.Z_over_Zsun = snap.gas.metal_mass_fractions / Z_sun

    print_HI_budget(snap.gas, m_HI, cosmo, meta.boxsize)

    # ---- Particle quantities as float64 numpy arrays ----
    rho_bar_b = cosmo.Ob0 * cosmo.critical_density0.to(u.g / u.cm**3).value    # mean comoving baryon density [g/cm^3]
    quantities = {
        "T [K]"              : snap.gas.temperatures.to("K").value.astype(np.float64),
        "n_H phys [cm^-3]"   : n_H.to_physical().to("cm**-3").value.astype(np.float64),
        "Delta = rho/rho_b"  : snap.gas.densities.to_comoving().to("g/cm**3").value.astype(np.float64) / rho_bar_b,
        "Z / Z_sun"          : snap.gas.Z_over_Zsun.value.astype(np.float64),
        "f_HI = n_HI/n_H"    : snap.gas.species_fractions.HI.value.astype(np.float64),
    }
    weights = {
        "gas mass" : snap.gas.masses.to("Msun").value.astype(np.float64),
        "HI mass"  : m_HI.to("Msun").value.astype(np.float64),
    }
    print_weighted_statistics(quantities, weights)

    # ---- Phase diagram n_H - T: where is the gas, where is the HI ----
    fig, (ax_m, ax_HI) = plt.subplots(1, 2, figsize=(13, 5))
    plot_phase_diagram(ax_m,  quantities["n_H phys [cm^-3]"], quantities["T [K]"], weights["gas mass"], cbar_label='Gas mass fraction')
    plot_phase_diagram(ax_HI, quantities["n_H phys [cm^-3]"], quantities["T [K]"], weights["HI mass"],  cbar_label='HI mass fraction')
    ax_m.set_title('Mass-weighted')
    ax_HI.set_title('HI-mass-weighted')
    fig.tight_layout()
    save_figure(fig, save_path, "gas", "Gas_phase_diagram_nH_T.png")

    # Single ax for gas and HI mass weight
    fig, ax = plt.subplots()
    plot_phase_diagram(ax, quantities["n_H phys [cm^-3]"], quantities["T [K]"], weights["gas mass"], cbar_label='Gas mass fraction')
    fig.tight_layout()
    save_figure(fig, save_path, "gas", "Gas_mass_weighted_phase_diagram_nH_T.png")
    
    fig, ax = plt.subplots()
    plot_phase_diagram(ax, quantities["n_H phys [cm^-3]"], quantities["T [K]"], weights["HI mass"],  cbar_label='HI mass fraction')
    fig.tight_layout()
    save_figure(fig, save_path, "gas", "Gas_massHI_weighted_phase_diagram_nH_T.png")

    # ---- HI maps in a slab: N_HI and HI-weighted properties ----
    slab_thickness = 2.0    # cMpc: avoid summing unrelated absorbers along the whole box
    z_box_c = 0.5 * L_box[2]
    slab_region = sw.cosmo_array([0, L_box[0], 0, L_box[1], z_box_c - slab_thickness/2, z_box_c + slab_thickness/2], "Mpc",
                                 comoving=True, scale_factor=meta.a, scale_exponent=1)
    slab_kwargs = dict(resolution=res, region=slab_region, parallel=True)

    # Sigma_HI (comoving), also used as denominator of the HI-weighted maps
    sigma_HI = compute_projection_map(snap, field="m_HI", weight=None, physical=False, **slab_kwargs)
    # ---- Check: HI mass conservation and N_HI conversion in the slab ----
    z_pos   = snap.gas.coordinates[:, 2].to_comoving().to("Mpc").value
    N_HI     = (sigma_HI.to_physical().to("g/cm**2") / hydrogen_mass).to("cm**-2")        # HI atoms per physical cm^2

    nH_HI = compute_projection_map(snap, field="n_H",          weight="m_HI", weight_map=sigma_HI, **slab_kwargs)
    T_HI  = compute_projection_map(snap, field="temperatures", weight="m_HI", weight_map=sigma_HI, **slab_kwargs)
    Z_HI  = compute_projection_map(snap, field="Z_over_Zsun",  weight="m_HI", weight_map=sigma_HI, **slab_kwargs)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_projection_map(N_HI,  axs[0, 0], Lbox=L_box, units='cm**-2', cbar_label=r'$N_{\rm HI} \ [\rm cm^{-2}]$', cmap='viridis', vmin=1e12)
    plot_projection_map(nH_HI, axs[0, 1], Lbox=L_box, units='cm**-3', cbar_label=r'$\langle n_{\rm H} \rangle_{\rm HI} \ [\rm cm^{-3}]$', cmap='magma')
    plot_projection_map(T_HI,  axs[1, 0], Lbox=L_box, units='K',      cbar_label=r'$\langle T \rangle_{\rm HI} \ [\rm K]$', cmap='inferno')
    plot_projection_map(Z_HI,  axs[1, 1], Lbox=L_box, units='dimensionless', cbar_label=r'$\langle Z \rangle_{\rm HI} \ / \ Z_\odot$', cmap='cividis')
    fig.suptitle(rf'Slab: $\Delta L$ = {slab_thickness:.1f} cMpc, centred at $z_{{\rm box}}$ = {z_box_c:.1f} cMpc '
             rf'(redshift $z$ = {redshift:.2f})')
    fig.tight_layout()
    save_figure(fig, save_path, "HI", "HI_xy2d_slab_maps.png")

    # ---- Pixel distribution of N_HI in the slab ----
    fig, ax = plt.subplots()
    f_LLS, f_DLA = plot_column_density_pixels(ax, N_HI)
    ax.set_title(rf'Slab $\Delta L$ = {slab_thickness} cMpc, {res}$^2$ pixels')
    save_figure(fig, save_path, "HI", "HI_NHI_pixel_distribution.png")

    print(f"\n[+] Slab N_HI map ({slab_thickness} cMpc, {res}^2 pixels):")
    print(f"     Pixel fraction N_HI >= LLS     :     {f_LLS:.3e}")
    print(f"     Pixel fraction N_HI >= DLA     :     {f_DLA:.3e}")
    in_slab = (z_pos >= z_box_c - slab_thickness/2) & (z_pos <= z_box_c + slab_thickness/2)
    M_HI_part = m_HI.to("Msun").value[in_slab].sum(dtype=np.float64)
    A_pix     = (L_box[0] / res) * (L_box[1] / res)                            # comoving Mpc^2
    M_HI_map  = sigma_HI.to_comoving().to("Msun/Mpc**2").value.sum(dtype=np.float64) * A_pix
    print(f"     M_HI map / particles           :     {M_HI_map / M_HI_part:.4f}")

    # ---- Absorption distance for LLS and the DLA (assuming random poissonian distribution of absorbers) ----
    dX    = slab_absorption_distance(slab_thickness, cosmo, redshift)
    l_LLS = -np.log(1 - f_LLS) / dX        # Poisson: P(>=1 absorber in dX) = 1 - exp(-l dX)
    l_DLA = -np.log(1 - f_DLA) / dX
    print(f"     Slab absorption distance dX    :     {dX:.4f}")
    print(f"     l_LLS(X) = dN/dX               :     {l_LLS:.3e}")
    print(f"     l_DLA(X) = dN/dX               :     {l_DLA:.3e}")


    # --- Chemical and metal inspection ----
    print_metal_budget(snap.gas, snap_filename)

    # ---- M2: particle abundances (gas phase = diffuse, i.e. what absorption lines see before dust correction) ----
    ratio = {el: number_ratio(snap.gas, el, diffuse=True) for el in ["oxygen", "carbon", "silicon", "magnesium", "iron"]}
    z_enrich = snap.gas.mean_metal_weighted_redshifts.value.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):        # 0/0 in pristine gas -> nan
        abundances = {
            "[O/H] total"   : bracket(number_ratio(snap.gas, "oxygen", diffuse=False), "oxygen"),
            "[O/H] gas"     : bracket(ratio["oxygen"],    "oxygen"),
            "[C/H] gas"     : bracket(ratio["carbon"],    "carbon"),
            "[Si/H] gas"    : bracket(ratio["silicon"],   "silicon"),
            "[Mg/H] gas"    : bracket(ratio["magnesium"], "magnesium"),
            "[Fe/H] gas"    : bracket(ratio["iron"],      "iron"),
            "[C/O] gas"     : bracket(ratio["carbon"]  / ratio["oxygen"], "carbon",  "oxygen"),
            "[Si/O] gas"    : bracket(ratio["silicon"] / ratio["oxygen"], "silicon", "oxygen"),
            "[Fe/O] gas"    : bracket(ratio["iron"]    / ratio["oxygen"], "iron",    "oxygen"),
            "delta_O"       : depletion(snap.gas, "oxygen"),
            "delta_Si"      : depletion(snap.gas, "silicon"),
            "delta_Fe"      : depletion(snap.gas, "iron"),
            "z_enrich"      : np.where(z_enrich >= 0, z_enrich, np.nan),    # -1 = never enriched -> nan
        }
    m_gas = snap.gas.masses.to("Msun").value.astype(np.float64)
    abundance_weights = {
        "gas mass"   : m_gas,
        "HI mass"    : weights["HI mass"],
        "metal mass" : m_gas * snap.gas.metal_mass_fractions.value.astype(np.float64),   # natural weight for z_enrich
    }
    print_weighted_statistics(abundances, abundance_weights)

    was_in_ISM = snap.gas.last_ismscale_factors.value > 0     # -1 = never part of the ISM
    print(f"\n[+] Gas that has been part of the ISM (recycled / outflow tracer):")
    for w_label, w_values in abundance_weights.items():
        print(f"     {w_label + ' fraction':<31}:     {w_values[was_in_ISM].sum() / w_values.sum():.3e}")

    # M2 figures: where are the metals among the particles, and in which gas phase
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    plot_weighted_pdf(axs[0], abundances["[O/H] gas"], abundance_weights, bins=np.arange(-8, 1.01, 0.1),
                      xlabel=r'$[\rm O/H]_{\rm gas}$ (particles)')
    plot_weighted_pdf(axs[1], abundances["[Fe/O] gas"], abundance_weights, bins=np.arange(-2, 1.01, 0.05),
                      xlabel=r'$[\rm Fe/O]_{\rm gas}$ (particles)')
    plot_weighted_pdf(axs[2], abundances["z_enrich"], {"metal mass": abundance_weights["metal mass"]},
                      bins=np.arange(5, 20.01, 0.25), xlabel=r'metal-mass-weighted enrichment redshift $\bar z_Z$')
    fig.suptitle(rf'Particle abundances (redshift $z$ = {redshift:.2f})')
    fig.tight_layout()
    save_figure(fig, save_path, "metal_diag", "Metals_particle_abundance_pdfs.png")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, w_label in zip(axs, ["gas mass", "HI mass"]):
        plot_hist2d(ax, np.log10(quantities["n_H phys [cm^-3]"]), abundances["[O/H] gas"],
                    x_bins=np.arange(-6, 4.01, 0.1), y_bins=np.arange(-8, 1.01, 0.1),
                    xlabel=r'$\log_{10} \ n_{\rm H} \ [\rm cm^{-3}]$', ylabel=r'$[\rm O/H]_{\rm gas}$',
                    weights=abundance_weights[w_label])
        ax.set_title(f'{w_label}-weighted (lowest row: [O/H] < -8 or pristine)')
    fig.tight_layout()
    save_figure(fig, save_path, "metal_diag", "Metals_nH_vs_OH_particles.png")
    del ratio, abundances

    # ---- M3: element column densities in the slab ----
    N_H     = element_column_density(snap, "hydrogen",  diffuse=False, **slab_kwargs)
    N_O     = element_column_density(snap, "oxygen",    diffuse=False, **slab_kwargs)   # all O atoms (gas + dust)
    N_O_gas = element_column_density(snap, "oxygen",    diffuse=True,  **slab_kwargs)
    N_OI    = element_column_density(snap, "oxygen",    diffuse=True, neutral=True, **slab_kwargs)   # OI proxy (x_OI = x_HI)
    N_C     = element_column_density(snap, "carbon",    diffuse=True,  **slab_kwargs)
    N_Si    = element_column_density(snap, "silicon",   diffuse=True,  **slab_kwargs)
    N_Fe    = element_column_density(snap, "iron",      diffuse=True,  **slab_kwargs)
    N_Fe_tot = element_column_density(snap, "iron",     diffuse=False, **slab_kwargs)

    # plain float64 numpy maps [cm^-2] for the ratios
    col = {name: N.to("cm**-2").value.astype(np.float64) for name, N in
           [("HI", N_HI), ("H", N_H), ("O", N_O), ("O_gas", N_O_gas), ("OI", N_OI),
            ("C", N_C), ("Si", N_Si), ("Fe", N_Fe), ("Fe_tot", N_Fe_tot)]}
    with np.errstate(divide="ignore", invalid="ignore"):
        logN_HI = np.log10(col["HI"])
        abundance_maps = {
            "[O/H]_HI"  : bracket(col["OI"] / col["HI"], "oxygen"),           # neutral-gas metallicity (OI/HI-like)
            "[O/H]_col" : bracket(col["O"]  / col["H"],  "oxygen"),           # true column metallicity
            "[C/O]"     : bracket(col["C"]  / col["O_gas"], "carbon",  "oxygen"),
            "[Si/O]"    : bracket(col["Si"] / col["O_gas"], "silicon", "oxygen"),
            "[Fe/O]"    : bracket(col["Fe"] / col["O_gas"], "iron",    "oxygen"),
            "delta_Fe"  : np.log10(col["Fe"] / col["Fe_tot"]),
        }

    # Checks: OI proxy cannot exceed all oxygen; oxygen mass is conserved by the projection
    A_pix_phys = (A_pix * unyt.Mpc**2 * meta.a**2).to("cm**2").value                # physical pixel area
    X_O_all    = snap.gas.element_mass_fractions.oxygen.value.astype(np.float64)
    M_O_part   = np.sum((m_gas * X_O_all)[in_slab])
    M_O_map    = (col["O"].sum() * A_pix_phys * ATOMIC_MASS["oxygen"] * unyt.amu).to("Msun").value
    print(f"\n[+] Slab element columns ({slab_thickness} cMpc, {res}^2 pixels):")
    print(f"     M_O map / particles            :     {M_O_map / M_O_part:.4f}")
    print(f"     Pixels with N_OI > N_O         :     {np.mean(col['OI'] > col['O'] * (1 + 1e-4)):.3e}")
    for name in ["OI", "C", "Si", "Fe"]:
        print(f"     max log N_{name:<6} [cm^-2]          :     {np.log10(col[name].max()):.2f}")

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plot_projection_map(N_HI, axs[0, 0], Lbox=L_box, units='cm**-2', cbar_label=r'$N_{\rm HI} \ [\rm cm^{-2}]$', cmap='viridis', vmin=1e12)
    plot_projection_map(N_OI, axs[0, 1], Lbox=L_box, units='cm**-2', cbar_label=r'$N_{\rm OI}^{\rm proxy} \ [\rm cm^{-2}]$', cmap='viridis', vmin=1e6)
    plot_abundance_map(abundance_maps["[O/H]_HI"], axs[0, 2], Lbox=L_box, cbar_label=r'$[\rm O/H]_{\rm HI}$')
    plot_projection_map(N_C,  axs[1, 0], Lbox=L_box, units='cm**-2', cbar_label=r'$N_{\rm C}^{\rm gas} \ [\rm cm^{-2}]$', cmap='cividis', vmin=1e8)
    plot_projection_map(N_Si, axs[1, 1], Lbox=L_box, units='cm**-2', cbar_label=r'$N_{\rm Si}^{\rm gas} \ [\rm cm^{-2}]$', cmap='cividis', vmin=1e7)
    plot_projection_map(N_Fe, axs[1, 2], Lbox=L_box, units='cm**-2', cbar_label=r'$N_{\rm Fe}^{\rm gas} \ [\rm cm^{-2}]$', cmap='cividis', vmin=1e7)
    fig.suptitle(rf'Metals in the slab: $\Delta L$ = {slab_thickness:.1f} cMpc (redshift $z$ = {redshift:.2f}); white/grey = pristine (X = 0)')
    fig.tight_layout()
    save_figure(fig, save_path, "metal_maps", "Metals_xy2d_slab_columns.png")

    # Enrichment redshift map, weighted by the metal mass that has a recorded enrichment event
    # (metals received only by diffusion have z_enrich = -1 and get weight 0)
    # The snapshot field is flagged as physical (comoving=False) and cannot be multiplied by comoving masses:
    # re-wrap the same values as a comoving, scale-independent (a^0) dimensionless cosmo_array.
    z_values = snap.gas.mean_metal_weighted_redshifts.value
    snap.gas.z_enrich  = sw.cosmo_array(z_values, "dimensionless", comoving=True, scale_factor=meta.a, scale_exponent=0)
    dated = (z_values >= 0).astype(np.float32)
    snap.gas.m_Z_dated = snap.gas.masses * snap.gas.metal_mass_fractions * dated
    zZ_map = compute_projection_map(snap, field="z_enrich", weight="m_Z_dated", **slab_kwargs)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plot_abundance_map(abundance_maps["[O/H]_col"], axs[0, 0], Lbox=L_box, cbar_label=r'$[\rm O/H]_{\rm col}$ (all O, all H)')
    plot_abundance_map(abundance_maps["[C/O]"],     axs[0, 1], Lbox=L_box, cbar_label=r'$[\rm C/O]_{\rm gas}$',  vmin=-1.0, vmax=0.0)
    plot_abundance_map(abundance_maps["[Si/O]"],    axs[0, 2], Lbox=L_box, cbar_label=r'$[\rm Si/O]_{\rm gas}$', vmin=-0.5, vmax=0.5)
    plot_abundance_map(abundance_maps["[Fe/O]"],    axs[1, 0], Lbox=L_box, cbar_label=r'$[\rm Fe/O]_{\rm gas}$', vmin=-1.5, vmax=0.5)
    plot_abundance_map(abundance_maps["delta_Fe"],  axs[1, 1], Lbox=L_box, cbar_label=r'$\delta_{\rm Fe}$ (column)', vmin=-0.5, vmax=0.0, cmap="viridis")
    plot_projection_map(zZ_map, axs[1, 2], Lbox=L_box, units='dimensionless', log=False, vmin=5, vmax=12,
                        cmap='plasma', cbar_label=r'$\bar z_Z$ (metal-mass weighted)')
    fig.suptitle(rf'Abundance ratios in the slab: $\Delta L$ = {slab_thickness:.1f} cMpc (redshift $z$ = {redshift:.2f}); grey/white = pristine')
    fig.tight_layout()
    save_figure(fig, save_path, "metal_maps", "Metals_xy2d_slab_ratios.png")
    del snap.gas.m_Z_dated, snap.gas.z_enrich

    # ---- M4: absorber classes ----
    print_absorber_classes(logN_HI, abundance_maps)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    plot_class_histograms(axs[0], logN_HI, abundance_maps["[O/H]_HI"], r'$[\rm O/H]_{\rm HI}$')
    plot_class_histograms(axs[1], logN_HI, abundance_maps["[Si/O]"],   r'$[\rm Si/O]_{\rm gas}$', bins=np.arange(-2, 1.01, 0.05))
    plot_class_histograms(axs[2], logN_HI, abundance_maps["[Fe/O]"],   r'$[\rm Fe/O]_{\rm gas}$', bins=np.arange(-2, 1.01, 0.05))
    fig.suptitle(rf'Metallicity of optically thick absorbers (slab {slab_thickness:.1f} cMpc, redshift $z$ = {redshift:.2f})')
    fig.tight_layout()
    save_figure(fig, save_path, "metal_diag", "Metals_abundances_by_absorber_class.png")

    # Key JAGGER-like diagnostic: absorber metallicity as a function of N_HI (one point per pixel)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, label in zip(axs, ["[O/H]_HI", "[O/H]_col"],
                              [r'$[\rm O/H]_{\rm HI}$ (neutral gas, OI/HI-like)', r'$[\rm O/H]_{\rm col}$ (all gas)']):
        plot_hist2d(ax, logN_HI.ravel(), abundance_maps[key].ravel(),
                    x_bins=np.arange(14, 22.01, 0.1), y_bins=np.arange(-8, 1.01, 0.1),
                    xlabel=r'$\log_{10} \ N_{\rm HI} \ [\rm cm^{-2}]$', ylabel=label, conditional=True)
        for logN in [17.2, 19.0, 20.3]:                      # LLS, subDLA, DLA boundaries
            ax.axvline(logN, color='red', ls='--', lw=1)
    fig.suptitle(rf'Metallicity vs $N_{{\rm HI}}$ per pixel (lowest row: [O/H] < -8 or pristine; redshift $z$ = {redshift:.2f})')
    fig.tight_layout()
    save_figure(fig, save_path, "metal_diag", "Metals_OH_vs_NHI_pixels.png")






    print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()


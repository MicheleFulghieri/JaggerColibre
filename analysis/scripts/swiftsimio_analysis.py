import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LogNorm
import numpy as np
import os
from operator import attrgetter
import h5py
import unyt
import astropy.units as u
from datetime import datetime
import swiftsimio as sw


# ---------------------------------------------------------------------------
# Useful paths
# ---------------------------------------------------------------------------
snap_dir      = '/cosma8/data/dp004/colibre/Runs'
run_dir       = 'L0025N0188/Thermal'
snap_nr       = 48  # z = 5.0
snap_filename = f'{snap_dir}/{run_dir}/SOAP-HBT/colibre_with_SOAP_membership_{snap_nr:04}.hdf5'
save_path     = '/cosma8/data/do019/dc-fulg1/outputs/swiftsimio_analysis'



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
def make_dirs(base):
    subdirs = ["Visualization2D", "Diagnostics"]
    for d in subdirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)


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
                   norm   = LogNorm(vmin=vmin, vmax=vmax) if log else None)

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


# ---------------------------------------------------------------------------
# Weighted statistics of particle quantities
# ---------------------------------------------------------------------------
def weighted_percentiles(values, weights, percentiles=(16, 50, 84)):
    """Percentiles of values in which each element counts proportionally to its weight."""

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
        print(f"     {'quantity':<{w}}  {'16th':>10} {'50th':>10} {'84th':>10}")
        print("     " + "-"*(w + 35))
        for q_label, q_values in quantities.items():
            p16, p50, p84 = weighted_percentiles(q_values, w_values)
            print(f"     {q_label:<{w}}  {p16:10.3e} {p50:10.3e} {p84:10.3e}")


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



# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # ---------------------------------------------------------------------------
    # Directory setting up
    # ---------------------------------------------------------------------------
    make_dirs(save_path)

    # ---------------------------------------------------------------------------
    # Virtual snapshot load and metadata
    # ---------------------------------------------------------------------------
    snap = sw.load(snap_filename)
    meta = snap.metadata                       # metadata method
    L_box    = meta.boxsize.to('Mpc').value
    cosmo    = meta.cosmology                  # astropy w0waCDM object
    redshift = meta.z
    time     = meta.t.to('Myr').value
    res      = 1024                            # pixels per side of the maps

    print(f"\n----------- Inspecting the Colibre snapshot {run_dir} -----------")
    print(f"     Available groups               :     {snap}")
    print(f"     Box size                       :     {meta.boxsize}")
    print(f"     Cosmic Time                    :     {time:.3f} Myr")
    print(f"     Redshift (scale factor)        :     {meta.z:.2f} ({meta.a:.2f})")
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
    fig.savefig(os.path.join(save_path, "Visualization2D", "Gas_xy2d_mass_projection.png"), dpi=300)
    plt.close(fig)


    # ---- Mass-weighted gas temperature ----
    T_gas = compute_projection_map(snap, field="temperatures", weight="masses", resolution=res, parallel=True)
    fig, ax = plt.subplots()
    plot_projection_map(T_gas, ax, Lbox=L_box, units='K', cbar_label=r'$\langle T \rangle_{\rm mass} \ [\rm K]$', cmap='inferno')
    fig.savefig(os.path.join(save_path, "Visualization2D", "Gas_xy2d_temperature_projection.png"), dpi=300)
    plt.close(fig)


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
    fig.savefig(os.path.join(save_path, "Diagnostics", "Gas_phase_diagram_nH_T.png"), dpi=300)
    plt.close(fig)

    # Single ax for gas and HI mass weight
    fig, ax = plt.subplots()
    plot_phase_diagram(ax, quantities["n_H phys [cm^-3]"], quantities["T [K]"], weights["gas mass"], cbar_label='Gas mass fraction')
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "Diagnostics", "Gas_mass_weighted_phase_diagram_nH_T.png"), dpi=300)
    plt.close(fig)
    
    fig, ax = plt.subplots()
    plot_phase_diagram(ax, quantities["n_H phys [cm^-3]"], quantities["T [K]"], weights["HI mass"],  cbar_label='HI mass fraction')
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "Diagnostics", "Gas_massHI_weighted_phase_diagram_nH_T.png"), dpi=300)
    plt.close(fig)

    # ---- HI maps in a slab: N_HI and HI-weighted properties ----
    slab_thickness = 2.0    # cMpc: avoid summing unrelated absorbers along the whole box
    z_c = 0.5 * L_box[2]
    slab_region = sw.cosmo_array([0, L_box[0], 0, L_box[1], z_c - slab_thickness/2, z_c + slab_thickness/2], "Mpc",
                                 comoving=True, scale_factor=meta.a, scale_exponent=1)
    slab_kwargs = dict(resolution=res, region=slab_region, parallel=True)

    # Sigma_HI (comoving), also used as denominator of the HI-weighted maps
    sigma_HI = compute_projection_map(snap, field="m_HI", weight=None, physical=False, **slab_kwargs)
    N_HI     = (sigma_HI.to_physical().to("g/cm**2") / hydrogen_mass).to("cm**-2")        # HI atoms per physical cm^2

    nH_HI = compute_projection_map(snap, field="n_H",          weight="m_HI", weight_map=sigma_HI, **slab_kwargs)
    T_HI  = compute_projection_map(snap, field="temperatures", weight="m_HI", weight_map=sigma_HI, **slab_kwargs)
    Z_HI  = compute_projection_map(snap, field="Z_over_Zsun",  weight="m_HI", weight_map=sigma_HI, **slab_kwargs)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_projection_map(N_HI,  axs[0, 0], Lbox=L_box, units='cm**-2', cbar_label=r'$N_{\rm HI} \ [\rm cm^{-2}]$', cmap='viridis', vmin=1e12)
    plot_projection_map(nH_HI, axs[0, 1], Lbox=L_box, units='cm**-3', cbar_label=r'$\langle n_{\rm H} \rangle_{\rm HI} \ [\rm cm^{-3}]$', cmap='magma')
    plot_projection_map(T_HI,  axs[1, 0], Lbox=L_box, units='K',      cbar_label=r'$\langle T \rangle_{\rm HI} \ [\rm K]$', cmap='inferno')
    plot_projection_map(Z_HI,  axs[1, 1], Lbox=L_box, units='dimensionless', cbar_label=r'$\langle Z \rangle_{\rm HI} \ / \ Z_\odot$', cmap='cividis')
    fig.suptitle(rf'Slab $\Delta z$ = {slab_thickness} cMpc centred at z = {z_c:.1f} cMpc  (redshift {redshift:.2f})')
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "Visualization2D", "Gas_xy2d_HI_slab_maps.png"), dpi=300)
    plt.close(fig)

    # ---- Pixel distribution of N_HI in the slab ----
    fig, ax = plt.subplots()
    f_LLS, f_DLA = plot_column_density_pixels(ax, N_HI)
    ax.set_title(rf'Slab $\Delta z$ = {slab_thickness} cMpc, {res}$^2$ pixels')
    fig.savefig(os.path.join(save_path, "Diagnostics", "Gas_NHI_pixel_distribution.png"), dpi=300)
    plt.close(fig)

    print(f"\n[+] Slab N_HI map ({slab_thickness} cMpc, {res}^2 pixels):")
    print(f"     Pixel fraction N_HI >= LLS     :     {f_LLS:.3e}")
    print(f"     Pixel fraction N_HI >= DLA     :     {f_DLA:.3e}")


    print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()


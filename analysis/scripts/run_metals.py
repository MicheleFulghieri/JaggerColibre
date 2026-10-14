"""Driver: block M of the roadmap. Metal budget and dust (M1), particle abundances (M2),
element column densities in the slab (M3), metallicity of the absorber classes (M4)."""

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import unyt
import swiftsimio as sw

from colibre_io import parse_args, open_snapshot, print_snapshot_header, save_figure
from projections import compute_projection_map, plot_projection_map
from hydrogen import compute_hydrogen_fields, make_slab, HI_column_density
from metals import (ATOMIC_MASS, print_metal_budget, number_ratio, bracket, depletion,
                    element_column_density, print_absorber_classes)
from weighted_stats import print_weighted_statistics
from plotting import plot_abundance_map, plot_class_histograms, plot_weighted_pdf, plot_hist2d


def main():
    args = parse_args("Metals and absorber metallicity of a COLIBRE snapshot")
    snap, snap_filename, save_path = open_snapshot(args)
    print_snapshot_header(snap, args.run, snap_filename, save_path)

    meta     = snap.metadata
    L_box    = meta.boxsize.to('Mpc').value
    redshift = meta.z
    res      = 1024                            # pixels per side of the maps

    # ---- Hydrogen quantities needed here: HI mass (weight), n_H, slab N_HI ----
    m_HI, n_H = compute_hydrogen_fields(snap.gas)
    snap.gas.m_HI = m_HI
    n_H_phys = n_H.to_physical().to("cm**-3").value.astype(np.float64)

    slab_thickness = 2.0    # cMpc
    slab_kwargs, z_box_c = make_slab(meta, slab_thickness, res)
    sigma_HI, N_HI = HI_column_density(snap, slab_kwargs)
    z_pos   = snap.gas.coordinates[:, 2].to_comoving().to("Mpc").value
    in_slab = (z_pos >= z_box_c - slab_thickness/2) & (z_pos <= z_box_c + slab_thickness/2)
    A_pix   = (L_box[0] / res) * (L_box[1] / res)                              # comoving Mpc^2

    # ---- M1: metal budget, dust, enrichment channels ----
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
        "HI mass"    : m_HI.to("Msun").value.astype(np.float64),
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
        plot_hist2d(ax, np.log10(n_H_phys), abundances["[O/H] gas"],
                    x_bins=np.arange(-6, 4.01, 0.1), y_bins=np.arange(-8, 1.01, 0.1),
                    xlabel=r'$\log_{10} \ n_{\rm H} \ [\rm cm^{-3}]$', ylabel=r'$[\rm O/H]_{\rm gas}$',
                    weights=abundance_weights[w_label])
        ax.set_title(f'{w_label}-weighted (lowest row: [O/H] < -8 or pristine)')
    fig.tight_layout()
    save_figure(fig, save_path, "metal_diag", "Metals_nH_vs_OH_particles.png")
    del ratio, abundances

    # ---- M3: element column densities in the slab ----
    N_H      = element_column_density(snap, "hydrogen",  diffuse=False, **slab_kwargs)
    N_O      = element_column_density(snap, "oxygen",    diffuse=False, **slab_kwargs)   # all O atoms (gas + dust)
    N_O_gas  = element_column_density(snap, "oxygen",    diffuse=True,  **slab_kwargs)
    N_OI     = element_column_density(snap, "oxygen",    diffuse=True, neutral=True, **slab_kwargs)   # OI proxy (x_OI = x_HI)
    N_C      = element_column_density(snap, "carbon",    diffuse=True,  **slab_kwargs)
    N_Si     = element_column_density(snap, "silicon",   diffuse=True,  **slab_kwargs)
    N_Fe     = element_column_density(snap, "iron",      diffuse=True,  **slab_kwargs)
    N_Fe_tot = element_column_density(snap, "iron",      diffuse=False, **slab_kwargs)

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

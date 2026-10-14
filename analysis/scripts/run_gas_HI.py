"""Driver: gas maps, phase diagrams, global HI budget, slab HI columns, LLS/DLA incidence l(X)."""

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from colibre_io import parse_args, open_snapshot, print_snapshot_header, print_gas_fields, save_figure
from projections import compute_projection_map, plot_projection_map
from hydrogen import (compute_hydrogen_fields, print_HI_budget, slab_absorption_distance,
                      make_slab, HI_column_density)
from metals import Z_sun
from weighted_stats import print_weighted_statistics
from plotting import plot_phase_diagram, plot_column_density_pixels


def main():
    args = parse_args("Gas and HI analysis of a COLIBRE snapshot")
    snap, snap_filename, save_path = open_snapshot(args)
    print_snapshot_header(snap, args.run, snap_filename, save_path)

    meta     = snap.metadata
    L_box    = meta.boxsize.to('Mpc').value
    cosmo    = meta.cosmology                  # astropy w0waCDM object
    redshift = meta.z
    res      = 1024                            # pixels per side of the maps

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
    slab_kwargs, z_box_c = make_slab(meta, slab_thickness, res)

    # Sigma_HI (comoving), also used as denominator of the HI-weighted maps
    sigma_HI, N_HI = HI_column_density(snap, slab_kwargs)

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

    # ---- Check: HI mass conservation in the slab ----
    z_pos     = snap.gas.coordinates[:, 2].to_comoving().to("Mpc").value
    in_slab   = (z_pos >= z_box_c - slab_thickness/2) & (z_pos <= z_box_c + slab_thickness/2)
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

    print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()

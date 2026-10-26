"""Driver: neutral hydrogen, equilibrium models vs CHIMES (steps I0b and I1 of the roadmap, validation).
   Neutral fraction f_HI from analytic equilibrium (HM12 or COLIBRE/PS20 UV background, with/without
   Rahmati self-shielding), from the COLIBRE equilibrium tables and from the Trident/CLOUDY HM12 tables,
   compared with the non-equilibrium CHIMES f_HI stored in the snapshot."""

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from colibre_io import parse_args, open_snapshot, print_snapshot_header, save_figure
from hydrogen import compute_hydrogen_fields
from weighted_stats import weighted_percentiles, binned_percentiles
from plotting import plot_hist2d
from hydrogen_equilibrium import gamma_HI_hm12, gamma_HI_colibre, rahmati_shielding, equilibrium_f_HI
from ion_tables import load_eqm_table, eqm_ion_fraction, trident_ion_fraction


# Neutral-hydrogen models: short internal key -> label shown in the figures (raw strings: LaTeX inside $...$)
HI_LABELS = {
    "chimes"              : r"COLIBRE (CHIMES, non-equilibrium)",
    "hm12_thin"           : r"Equilibrium, $\Gamma_{\rm HI}^{\rm HM12}$, optically thin",
    "hm12_r13"            : r"Equilibrium, $\Gamma_{\rm HI}^{\rm HM12}$ + R13 shielding",
    "ps20_r13"            : r"Equilibrium, $\Gamma_{\rm HI}^{\rm PS20}$ + R13 shielding",
    "colibre_eqm"         : r"Equilibrium, COLIBRE tables (CHIMES full network)",
    "trident_cloudy_thin" : r"TRIDENT (CLOUDY, HM12, optically thin)",
    "trident_cloudy_ss"   : r"TRIDENT (CLOUDY, HM12, self-shielded)",
}


def main():
    args = parse_args("Neutral hydrogen: equilibrium models vs CHIMES")
    snap, snap_filename, save_path = open_snapshot(args)
    print_snapshot_header(snap, args.run, snap_filename, save_path)

    meta  = snap.metadata
    cosmo = meta.cosmology
    z     = meta.z
    gas   = snap.gas

    # ---- Particle quantities (physical, float64) ----
    m_HI, n_H_c = compute_hydrogen_fields(gas)
    n_H   = n_H_c.to_physical().to("cm**-3").value.astype(np.float64)
    T     = gas.temperatures.to("K").value.astype(np.float64)
    m     = gas.masses.to("Msun").value.astype(np.float64)
    X_H   = gas.element_mass_fractions.hydrogen.value.astype(np.float64)
    Y     = gas.element_mass_fractions.helium.value.astype(np.float64)
    Z     = gas.metal_mass_fractions.value.astype(np.float64)   # total (gas + dust) metal mass fraction
    eta   = 1.0 + (Y / 4.003) / (X_H / 1.008)                  # n_e / n_HII with He singly ionised
    f_HI_chimes = gas.species_fractions.HI.value.astype(np.float64)
    f_g   = cosmo.Ob0 / cosmo.Om0                               # universal baryon fraction (Rahmati fit)

    # ---- UV background photoionisation rates ----
    gamma_hm12               = gamma_HI_hm12(z)
    gamma_colibre, sigma_col = gamma_HI_colibre(z)
    print(f"\n[+] HI photoionisation rate of the UV background at z = {z:.2f}:")
    print(f"     Gamma_HI HM12                  :     {gamma_hm12:.3e} s^-1")
    print(f"     Gamma_HI COLIBRE (PS20)        :     {gamma_colibre:.3e} s^-1   (sigma_HI = {sigma_col:.3e} cm^2)")

    # ---- Neutral fraction: models ----
    f_HI = {
        "chimes"              : f_HI_chimes,
        "hm12_thin"           : equilibrium_f_HI(n_H, T, gamma_hm12, eta),
        "hm12_r13"            : equilibrium_f_HI(n_H, T, gamma_hm12 * rahmati_shielding(n_H, T, gamma_hm12, f_g=f_g), eta),
        "ps20_r13"            : equilibrium_f_HI(n_H, T, gamma_colibre * rahmati_shielding(n_H, T, gamma_colibre,
                                                                                           sigma_nu=sigma_col, f_g=f_g), eta),
        "colibre_eqm"         : eqm_ion_fraction(load_eqm_table("HI", z), n_H, T, Z),
        "trident_cloudy_thin" : trident_ion_fraction("HM12 thin", "H", 1, z, n_H, T),
        "trident_cloudy_ss"   : trident_ion_fraction("HM12 ss",   "H", 1, z, n_H, T),
    }

    V_box     = np.prod(meta.boxsize.to("Mpc").value)                        # comoving volume [cMpc^3]
    rho_crit0 = cosmo.critical_density0.to(u.Msun / u.Mpc**3).value
    M_H       = np.sum(m * X_H)
    w_HI      = m * X_H * f_HI_chimes                                        # CHIMES HI mass, weight
    print(f"\n[+] Neutral hydrogen: models vs CHIMES")
    print(f"     {'model':<20} {'M_HI/M_H':>10} {'Omega_HI':>10} {'med log(f/f_CHIMES)':>22} {'(HI-weighted)':>14}")
    for key, f in f_HI.items():
        M_HI = np.sum(m * X_H * f)
        with np.errstate(divide="ignore", invalid="ignore"):                 # f_HI = 0 in some particles
            log_r = np.log10(f / f_HI_chimes)
        med_gas = weighted_percentiles(log_r, m,    (50,))[0]
        med_HI  = weighted_percentiles(log_r, w_HI, (50,))[0]
        print(f"     {key:<20} {M_HI / M_H:10.3e} {M_HI / V_box / rho_crit0:10.3e} {med_gas:22.3f} {med_HI:14.3f}")
    print("     (keys: " + "; ".join(f"{k} = {v}" for k, v in HI_LABELS.items()) + ")")

    # ---- Figure 1: Gamma_HI(z) of the two UV backgrounds ----
    z_plot = np.linspace(3.5, 6.5, 61)
    fig, ax = plt.subplots()
    ax.plot(z_plot, gamma_HI_hm12(z_plot), label='HM12 (Haardt & Madau 2012)')
    ax.plot(z_plot, gamma_HI_colibre(z_plot)[0], label='PS20, COLIBRE (Ploeckinger & Schaye 2020)')
    ax.axvline(z, color='grey', ls='--', lw=1, label=rf'$z_{{\rm snap}}$ = {z:.2f}')
    ax.set_yscale('log')
    ax.set_xlabel('redshift $z$')
    ax.set_ylabel(r'$\Gamma_{\rm HI}\ [\rm s^{-1}]$')
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, save_path, "HI_models", "HI_GammaHI_vs_z.png")

    # ---- Figure 2: f_HI vs n_H (gas-mass-weighted median per n_H bin; band = CHIMES 16-84th) ----
    log_nH = np.log10(n_H)
    n_bins = np.arange(-6, 3.01, 0.2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, f in f_HI.items():
        is_sim = key == "chimes"                                # the simulation: thick black line + 16-84th band
        with np.errstate(divide="ignore"):
            centres, p = binned_percentiles(log_nH, np.log10(f), m, n_bins)
        ax.plot(centres, p[:, 1], lw=2.5 if is_sim else 1.5, color='k' if is_sim else None, label=HI_LABELS[key])
        if is_sim:
            ax.fill_between(centres, p[:, 0], p[:, 2], color='grey', alpha=0.3, label='COLIBRE 16th-84th percentile')
    ax.set_xlabel(r'$\log_{10}\ n_{\rm H}\ [\rm cm^{-3}]$')
    ax.set_ylabel(r'median $\log_{10}\ f_{\rm HI}$')
    ax.set_ylim(-7, 0.3)
    ax.legend(fontsize=8)
    ax.set_title(rf'Neutral fraction vs density (redshift $z$ = {z:.2f})')
    fig.tight_layout()
    save_figure(fig, save_path, "HI_models", "HI_fHI_vs_nH.png")

    # ---- Figure 3: equilibrium models vs CHIMES, particle by particle (gas-mass weighted) ----
    bins = np.arange(-7, 0.51, 0.05)
    fig, axs = plt.subplots(1, 3, figsize=(19, 5))
    for ax, key in zip(axs, ["colibre_eqm", "ps20_r13", "trident_cloudy_ss"]):
        with np.errstate(divide="ignore"):                    # f_HI = 0 in some particles -> log10 = -inf
            x, y = np.log10(f_HI_chimes), np.log10(f_HI[key])
        plot_hist2d(ax, x, y, x_bins=bins, y_bins=bins,
                    xlabel=r'$\log_{10} f_{\rm HI}$, ' + HI_LABELS["chimes"],
                    ylabel=r'$\log_{10} f_{\rm HI}$, model', weights=m)
        ax.plot(bins, bins, color='red', lw=1, ls='--', label='1:1')
        ax.set_title(HI_LABELS[key], fontsize=11)
        ax.legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    save_figure(fig, save_path, "HI_models", "HI_fHI_models_vs_CHIMES.png")

    # ---- Figure 4: where in density the equilibrium tables depart from CHIMES ----
    with np.errstate(divide="ignore", invalid="ignore"):      # 0/0 -> nan, excluded by plot_hist2d
        log_ratio = np.log10(f_HI["colibre_eqm"] / f_HI_chimes)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_hist2d(ax, log_nH, log_ratio,
                x_bins=np.arange(-6, 3.01, 0.1), y_bins=np.arange(-4, 4.01, 0.05),
                xlabel=r'$\log_{10}\ n_{\rm H}\ [\rm cm^{-3}]$',
                ylabel=r'$\log_{10}\ f_{\rm HI}^{\rm eqm\ table} / f_{\rm HI}^{\rm CHIMES}$',
                weights=m, conditional=True)
    ax.axhline(0, color='red', lw=1, ls='--')
    ax.set_title(rf'COLIBRE equilibrium tables vs CHIMES (redshift $z$ = {z:.2f})', fontsize=11)
    fig.tight_layout()
    save_figure(fig, save_path, "HI_models", "HI_fHI_eqm_over_CHIMES_vs_nH.png")

    print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()

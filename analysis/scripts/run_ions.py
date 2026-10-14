"""Driver: step I0b, model 0 of the ionisation state.
   Neutral hydrogen fraction from analytic equilibrium (HM12 or COLIBRE UV background, with/without
   Rahmati self-shielding) and from Trident/CLOUDY HM12 tables, compared with CHIMES (snapshot).
   First look at the metal ions with the Trident HM12 tables."""

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from colibre_io import parse_args, open_snapshot, print_snapshot_header, save_figure
from hydrogen import compute_hydrogen_fields
from weighted_stats import weighted_percentiles
from plotting import plot_hist2d
from ions import (gamma_HI_hm12, gamma_HI_colibre, rahmati_shielding, equilibrium_f_HI, trident_ion_fraction)


# (label, Trident element, ionisation stage, COLIBRE element) of the metal ions observed by JAGGER
METAL_IONS = [("OI",   "O",  1, "oxygen"),
              ("CII",  "C",  2, "carbon"),
              ("CIV",  "C",  4, "carbon"),
              ("SiII", "Si", 2, "silicon"),
              ("SiIV", "Si", 4, "silicon"),
              ("MgII", "Mg", 2, "magnesium"),
              ("FeII", "Fe", 2, "iron")]


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


def main():
    args = parse_args("Model 0 of the ionisation state (HM12 / COLIBRE UVB) vs CHIMES")
    snap, snap_filename, save_path = open_snapshot(args)
    print_snapshot_header(snap, args.run, snap_filename, save_path)

    meta  = snap.metadata
    cosmo = meta.cosmology
    z     = meta.z
    gas   = snap.gas

    np.seterr(divide="ignore", invalid="ignore")              # f_HI = 0 exactly in some particles: log10(0) = -inf is expected

    # ---- Particle quantities (physical, float64) ----
    m_HI, n_H_c = compute_hydrogen_fields(gas)
    n_H   = n_H_c.to_physical().to("cm**-3").value.astype(np.float64)
    T     = gas.temperatures.to("K").value.astype(np.float64)
    m     = gas.masses.to("Msun").value.astype(np.float64)
    X_H   = gas.element_mass_fractions.hydrogen.value.astype(np.float64)
    Y     = gas.element_mass_fractions.helium.value.astype(np.float64)
    eta   = 1.0 + (Y / 4.003) / (X_H / 1.008)                # n_e / n_HII with He singly ionised
    f_HI_chimes = gas.species_fractions.HI.value.astype(np.float64)
    f_g   = cosmo.Ob0 / cosmo.Om0                            # universal baryon fraction (Rahmati fit)

    # ---- UV background photoionisation rates ----
    gamma_hm12               = gamma_HI_hm12(z)
    gamma_colibre, sigma_col = gamma_HI_colibre(z)
    print(f"\n[+] HI photoionisation rate of the UV background at z = {z:.2f}:")
    print(f"     Gamma_HI HM12                  :     {gamma_hm12:.3e} s^-1")
    print(f"     Gamma_HI COLIBRE (PS20)        :     {gamma_colibre:.3e} s^-1   (sigma_HI = {sigma_col:.3e} cm^2)")

    # ---- Neutral fraction: models ----
    f_HI = {
        "CHIMES (snapshot)"      : f_HI_chimes,
        "HM12 thin (analytic)"   : equilibrium_f_HI(n_H, T, gamma_hm12, eta),
        "HM12 + Rahmati"         : equilibrium_f_HI(n_H, T, gamma_hm12 * rahmati_shielding(n_H, T, gamma_hm12, f_g=f_g), eta),
        "COLIBRE G + Rahmati"    : equilibrium_f_HI(n_H, T, gamma_colibre * rahmati_shielding(n_H, T, gamma_colibre,
                                                                                               sigma_nu=sigma_col, f_g=f_g), eta),
        "Trident HM12 thin"      : trident_ion_fraction("HM12 thin", "H", 1, z, n_H, T),
        "Trident HM12 ss"        : trident_ion_fraction("HM12 ss",   "H", 1, z, n_H, T),
    }

    V_box     = np.prod(meta.boxsize.to("Mpc").value)                        # comoving volume [cMpc^3]
    rho_crit0 = cosmo.critical_density0.to(u.Msun / u.Mpc**3).value
    M_H       = np.sum(m * X_H)
    w_HI      = m * X_H * f_HI_chimes                                        # CHIMES HI mass, weight
    print(f"\n[+] Neutral hydrogen: models vs CHIMES")
    print(f"     {'model':<24} {'M_HI/M_H':>10} {'Omega_HI':>10} {'med log(f/f_CHIMES)':>22} {'(HI-weighted)':>14}")
    for label, f in f_HI.items():
        M_HI     = np.sum(m * X_H * f)
        log_r    = np.log10(f / f_HI_chimes)
        med_gas  = weighted_percentiles(log_r, m,    (50,))[0]
        med_HI   = weighted_percentiles(log_r, w_HI, (50,))[0]
        print(f"     {label:<24} {M_HI / M_H:10.3e} {M_HI / V_box / rho_crit0:10.3e} {med_gas:22.3f} {med_HI:14.3f}")

    # ---- Metal ions, first look with the Trident HM12 tables (fraction of the gas-phase element in the ion) ----
    print(f"\n[+] Fraction of the gas-phase element in each ion (mass weighted, whole box), Trident HM12 tables")
    print(f"     {'ion':<6} {'thin':>10} {'self-shielded':>14}")
    for ion, el_trident, stage, el in METAL_IONS:
        m_el = m * getattr(gas.element_mass_fractions_diffuse, el).value.astype(np.float64)
        fractions = [np.sum(m_el * trident_ion_fraction(t, el_trident, stage, z, n_H, T)) / np.sum(m_el)
                     for t in ["HM12 thin", "HM12 ss"]]
        print(f"     {ion:<6} {fractions[0]:10.3e} {fractions[1]:14.3e}")

    # ---- Figure 1: Gamma_HI(z) of the two UV backgrounds ----
    z_plot = np.linspace(3.5, 6.5, 61)
    fig, ax = plt.subplots()
    ax.plot(z_plot, gamma_HI_hm12(z_plot), label='HM12 (Haardt & Madau 2012)')
    ax.plot(z_plot, gamma_HI_colibre(z_plot)[0], label='COLIBRE (PS20 spectrum)')
    ax.axvline(z, color='grey', ls='--', lw=1, label=f'snapshot z = {z:.2f}')
    ax.set_yscale('log')
    ax.set_xlabel('redshift $z$')
    ax.set_ylabel(r'$\Gamma_{\rm HI}\ [\rm s^{-1}]$')
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, save_path, "ions_model0", "Model0_GammaHI_vs_z.png")

    # ---- Figure 2: f_HI vs n_H (gas-mass-weighted median per n_H bin; band = CHIMES 16-84th) ----
    log_nH = np.log10(n_H)
    n_bins = np.arange(-6, 3.01, 0.2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, f in f_HI.items():
        centres, p = binned_percentiles(log_nH, np.log10(f), m, n_bins)
        ax.plot(centres, p[:, 1], lw=2.5 if label.startswith("CHIMES") else 1.5,
                color='k' if label.startswith("CHIMES") else None, label=label)
        if label.startswith("CHIMES"):
            ax.fill_between(centres, p[:, 0], p[:, 2], color='grey', alpha=0.3)
    ax.set_xlabel(r'$\log_{10}\ n_{\rm H}\ [\rm cm^{-3}]$')
    ax.set_ylabel(r'median $\log_{10}\ f_{\rm HI}$')
    ax.set_ylim(-7, 0.3)
    ax.legend(fontsize=8)
    ax.set_title(rf'Neutral fraction vs density (redshift $z$ = {z:.2f})')
    fig.tight_layout()
    save_figure(fig, save_path, "ions_model0", "Model0_fHI_vs_nH.png")

    # ---- Figure 3: model vs CHIMES particle by particle (gas-mass weighted) ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.arange(-7, 0.51, 0.05)
    for ax, label in zip(axs, ["COLIBRE G + Rahmati", "Trident HM12 ss"]):
        plot_hist2d(ax, np.log10(f_HI_chimes), np.log10(f_HI[label]), x_bins=bins, y_bins=bins,
                    xlabel=r'$\log_{10} f_{\rm HI}$ CHIMES (snapshot)', ylabel=rf'$\log_{{10}} f_{{\rm HI}}$ {label}',
                    weights=m)
        ax.plot(bins, bins, color='red', lw=1, ls='--')
    fig.tight_layout()
    save_figure(fig, save_path, "ions_model0", "Model0_fHI_model_vs_CHIMES.png")

    print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()

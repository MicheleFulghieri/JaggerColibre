"""Driver: metal ions observed by JAGGER.
   Fraction of each gas-phase element in the ion, from the COLIBRE equilibrium tables (prediction)
   and from the Trident/CLOUDY HM12 tables (model systematic). No non-equilibrium reference exists:
   in the COLIBRE runs CHIMES follows only H and He (CHIMESCooling:Include<Element> = 0)."""

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from colibre_io import parse_args, open_snapshot, print_snapshot_header, save_figure
from hydrogen import compute_hydrogen_fields
from weighted_stats import binned_percentiles
from ion_tables import load_eqm_table, eqm_ion_fraction, trident_ion_fraction


# (ion, Trident element, ionisation stage, COLIBRE element) of the metal ions observed by JAGGER
METAL_IONS = [("OI",   "O",  1, "oxygen"),
              ("CII",  "C",  2, "carbon"),
              ("CIV",  "C",  4, "carbon"),
              ("SiII", "Si", 2, "silicon"),
              ("SiIV", "Si", 4, "silicon"),
              ("MgII", "Mg", 2, "magnesium"),
              ("FeII", "Fe", 2, "iron")]

# Ion-fraction models: short internal key -> label shown in the figures
ION_LABELS = {
    "colibre_eqm"  : r"COLIBRE equilibrium tables (CHIMES)",
    "trident_thin" : r"TRIDENT (CLOUDY, HM12, optically thin)",
    "trident_ss"   : r"TRIDENT (CLOUDY, HM12, self-shielded)",
}


def main():
    args = parse_args("Metal ion fractions of a COLIBRE snapshot (COLIBRE tables vs Trident)")
    snap, snap_filename, save_path = open_snapshot(args)
    print_snapshot_header(snap, args.run, snap_filename, save_path)

    z   = snap.metadata.z
    gas = snap.gas

    # ---- Particle quantities (physical, float64) ----
    _, n_H_c = compute_hydrogen_fields(gas)                          # '_' = returned value not needed
    n_H    = n_H_c.to_physical().to("cm**-3").value.astype(np.float64)
    T      = gas.temperatures.to("K").value.astype(np.float64)
    m      = gas.masses.to("Msun").value.astype(np.float64)
    Z      = gas.metal_mass_fractions.value.astype(np.float64)       # total (gas + dust) metal mass fraction
    log_nH = np.log10(n_H)
    n_bins = np.arange(-6, 3.01, 0.2)

    print(f"\n[+] Fraction of the gas-phase element in each ion (element-mass weighted, whole box)")
    print(f"     {'ion':<6} " + " ".join(f"{key:>14}" for key in ION_LABELS))

    fig, axs = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    for ax, (ion, el_trident, stage, el) in zip(axs.flat, METAL_IONS):
        m_el  = m * getattr(gas.element_mass_fractions_diffuse, el).value.astype(np.float64)   # gas-phase element mass
        f_ion = {
            "colibre_eqm"  : eqm_ion_fraction(load_eqm_table(ion, z), n_H, T, Z),
            "trident_thin" : trident_ion_fraction("HM12 thin", el_trident, stage, z, n_H, T),
            "trident_ss"   : trident_ion_fraction("HM12 ss",   el_trident, stage, z, n_H, T),
        }
        box_fraction = {key: np.sum(m_el * f) / np.sum(m_el) for key, f in f_ion.items()}
        print(f"     {ion:<6} " + " ".join(f"{box_fraction[key]:14.3e}" for key in ION_LABELS))

        for key, f in f_ion.items():
            with np.errstate(divide="ignore"):
                centres, p = binned_percentiles(log_nH, np.log10(f), m_el, n_bins, percentiles=(50,))
            ax.plot(centres, p[:, 0], lw=2 if key == "colibre_eqm" else 1.2, label=ION_LABELS[key])
        ax.set_title(ion)

    axs.flat[-1].axis("off")                                         # 7 ions in 8 panels: hide the last one
    for ax in axs[1, :]:
        ax.set_xlabel(r'$\log_{10}\ n_{\rm H}\ [\rm cm^{-3}]$')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'median $\log_{10}\ f_{\rm ion}$')
    axs[0, 0].set_ylim(-6, 0.2)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(rf'Ion fractions vs density, element-mass weighted (redshift $z$ = {z:.2f})')
    fig.tight_layout()
    save_figure(fig, save_path, "metal_ions", "Ions_fion_vs_nH.png")

    print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()

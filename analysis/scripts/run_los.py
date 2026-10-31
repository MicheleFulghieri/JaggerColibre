"""Driver: read one COLIBRE line of sight and summarise its particles."""

import os
import argparse

import numpy as np
from astropy import constants as const

from los_io import LOS_DIR, read_sightline
from weighted_stats import weighted_percentiles

MSUN_G = const.M_sun.cgs.value  # g
M_H_G = const.m_p.cgs.value    # g, proton mass (H atom mass to 0.05%)

def parse_args():
    p = argparse.ArgumentParser(description="Read one COLIBRE sightline and print a summary")
    p.add_argument("--file", type=int, default=51, help="LOS output number (51 -> los_0051.hdf5, z = 5.02)")
    p.add_argument("--los",  type=int, default=0,  help="sightline index in the file (0-104)")
    return p.parse_args()

def main():
    args = parse_args()
    filename = os.path.join(LOS_DIR, f"los_{args.file:04d}.hdf5")
    los = read_sightline(filename, args.los)

    # ---- Geometry: impact parameter of each particle, in units of its smoothing length ----
    ax_x, ax_y, ax_z = los["axes"]
    cm_per_cMpc = los["unit_length"] * los["a"]                # comoving Mpc -> physical cm
    box = los["box_cMpc"] * cm_per_cMpc
    dx  = los["pos"][:, ax_x] - los["xpos_cMpc"] * cm_per_cMpc
    dy  = los["pos"][:, ax_y] - los["ypos_cMpc"] * cm_per_cMpc
    dx -= box * np.round(dx / box)                             # periodic box: nearest image
    dy -= box * np.round(dy / box)
    b_over_h = np.hypot(dx, dy) / los["h"]                     # np.hypot compute the hypotenuse


    # ---- Hydrogen: n_H, HI mass from CHIMES (non-equilibrium) and from the equilibrium tables ----
    X_H     = los["X"][:, los["element_names"].index("Hydrogen")]
    f_HI    = los["species"][:, los["species_names"].index("HI")]   # n_HI / n_H
    m       = los["mass"] / MSUN_G                                   # Msun
    n_H     = X_H * los["rho"] / M_H_G                               # cm^-3, physical
    m_HI    = f_HI * X_H * m                                         # Msun, CHIMES
    m_HI_eq = los["m_HI_eq"] / MSUN_G                                # Msun, equilibrium tables

    print(f"\n[+] Sightline {args.los} of {filename}")
    print(f"     Redshift (scale factor)        :     {los['z']:.3f} ({los['a']:.4f})")
    print(f"     Direction, crossing point      :     {'xyz'[ax_z]}, ({los['xpos_cMpc']:.3f}, {los['ypos_cMpc']:.3f}) cMpc")
    print(f"     Number of particles            :     {m.size}")
    print(f"     Max b / h (kernel gamma {los['kernel_gamma']:.4f}):     {b_over_h.max():.4f}")
    print(f"     Position along the LOS [cMpc]  :     {los['pos'][:, ax_z].min() / cm_per_cMpc:.3f} - {los['pos'][:, ax_z].max() / cm_per_cMpc:.3f}")
    print(f"     Peculiar v along the LOS [km/s]:     {los['vel'][:, ax_z].min() / 1e5:.1f} - {los['vel'][:, ax_z].max() / 1e5:.1f}")

    print(f"\n[+] Hydrogen along the sightline")
    print(f"     Gas mass                       :     {m.sum():.3e} Msun")
    print(f"     HI mass, CHIMES                :     {m_HI.sum():.3e} Msun")
    print(f"     HI mass, equilibrium tables    :     {m_HI_eq.sum():.3e} Msun   (ratio CHIMES/eq = {m_HI.sum() / m_HI_eq.sum():.3f})")

    with np.errstate(divide="ignore", invalid="ignore"):              # log10(0) for fully ionised particles
        log_nH, log_T = np.log10(n_H), np.log10(los["T"])
        log_ratio     = np.log10(m_HI / m_HI_eq)
    print(f"     16th / 50th / 84th percentiles :")
    for label, values, weights in [("log n_H, gas-mass weighted",   log_nH,    m),
                                   ("log T, gas-mass weighted",     log_T,     m),
                                   ("log n_H, HI-mass weighted",    log_nH,    m_HI),
                                   ("log(m_HI CHIMES/eq), HI w.",   log_ratio, m_HI)]:
        p16, p50, p84 = weighted_percentiles(values, weights)
        print(f"     {label:<31}:     {p16:7.2f} {p50:7.2f} {p84:7.2f}")

if __name__ == "__main__":
    main()

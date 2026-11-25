"""Driver: read one or more COLIBRE lines of sight of the same output and summarise each in a table row."""

import os
import argparse

import numpy as np
from astropy import constants as const

from los_io import LOS_DIR, los_catalogue, read_sightline
from weighted_stats import weighted_percentiles

MSUN_G = const.M_sun.cgs.value      # g
M_H_G  = const.m_p.cgs.value        # g, proton mass (H atom mass to 0.05%)


def parse_args():
    p = argparse.ArgumentParser(description="Summarise COLIBRE sightlines of one LOS output")
    p.add_argument("--file", type=int, default=51, help="LOS output number (51 -> los_0051.hdf5, z = 5.02)")
    p.add_argument("--list", action="store_true", help="list the LOS files in 4 <= z <= 5.5 and exit")
    p.add_argument("--los",  type=int, nargs="+", default=[0], help="one or more sightline indices (0-104)")
    p.add_argument("--save", default=None, help="text file for the summary table (optional)")
    return p.parse_args()

def summarise_sightline(filename, index):
    """Geometry and hydrogen content of one sightline, as a dictionary of numbers (one table row)."""
    los = read_sightline(filename, index)

    # Geometry: impact parameter in units of the smoothing length (periodic box)
    ax_x, ax_y, ax_z = los["axes"]
    cm_per_cMpc = los["unit_length"] * los["a"]
    box = los["box_cMpc"] * cm_per_cMpc
    dx  = los["pos"][:, ax_x] - los["xpos_cMpc"] * cm_per_cMpc
    dy  = los["pos"][:, ax_y] - los["ypos_cMpc"] * cm_per_cMpc
    dx -= box * np.round(dx / box)
    dy -= box * np.round(dy / box)
    b_over_h = np.hypot(dx, dy) / los["h"]

    # Hydrogen
    X_H     = los["X"][:, los["element_names"].index("Hydrogen")]
    f_HI    = los["species"][:, los["species_names"].index("HI")]
    m       = los["mass"] / MSUN_G
    n_H     = X_H * los["rho"] / M_H_G
    m_HI    = f_HI * X_H * m
    m_HI_eq = los["m_HI_eq"] / MSUN_G
    with np.errstate(divide="ignore", invalid="ignore"):
        log_nH, log_T = np.log10(n_H), np.log10(los["T"])

    return {
        "los"          : index,
        "axis"         : ax_z,
        "z"            : los["z"],
        "n_part"       : m.size,
        "max_b_over_h" : b_over_h.max(),
        "M_gas_Msun"   : m.sum(),
        "M_HI_Msun"    : m_HI.sum(),
        "HI_chimes_eq" : m_HI.sum() / m_HI_eq.sum(),
        "lognH_gas_med": weighted_percentiles(log_nH, m,    (50,))[0],
        "logT_gas_med" : weighted_percentiles(log_T,  m,    (50,))[0],
        "lognH_HI_med" : weighted_percentiles(log_nH, m_HI, (50,))[0],
    }


def main():
    args     = parse_args()
    if args.list:
        numbers, z, per_axis = los_catalogue()
        print(f"\n[+] {numbers.size} LOS files in {LOS_DIR}")
        for n, zi, c in zip(numbers, z, per_axis):
            print(f"     los_{n:04d}   z = {zi:.3f}   LOS along x/y/z = {c}")
        a = 1.0 / (1.0 + z)
        print(f"     Cadence a_(i+1)/a_i = 1.01, no gaps: {np.allclose(a[1:] / a[:-1], 1.01, rtol=1e-3)}")
        return

    filename = os.path.join(LOS_DIR, f"los_{args.file:04d}.hdf5")
    rows     = [summarise_sightline(filename, i) for i in args.los]
    columns  = list(rows[0].keys())

    print(f"\n[+] Sightline summary of {filename} ({len(rows)} sightlines)")
    print("     " + " ".join(f"{c:>13}" for c in columns))
    for row in rows:
        print("     " + " ".join(f"{row[c]:13.4g}" for c in columns))

    if args.save is not None:
        table = np.array([[row[c] for c in columns] for row in rows])
        np.savetxt(args.save, table, header=" ".join(columns), fmt="%.6e")
        print(f"\n     [saved] {args.save}")

    
if __name__ == "__main__":
    main()
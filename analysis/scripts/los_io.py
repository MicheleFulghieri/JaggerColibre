"""Gas for sightline analyses, from two sources.

   1) Native COLIBRE LOS files (legacy, kept for consistency checks): each los_XXXX.hdf5 holds 105 sightlines
      (35 per axis); group LOS_YYYY contains the gas particles whose kernel intersects it. Read in physical CGS.
   2) Snapshots (built sightlines): all gas particles, in internal comoving units (cMpc, km/s, 1e10 Msun)."""

import os
import glob

import numpy as np
import h5py

LOS_DIR    = "/cosma8/data/dp004/colibre/Runs/L025_m5/Thermal/los"
CGS_FACTOR = "Conversion factor to physical CGS (including cosmological corrections)"

# Particle fields read for each sightline: dataset name in the file -> short key used in the code
LOS_FIELDS = {
    "Coordinates"                 : "pos",        # cm, physical, (N, 3)
    "Velocities"                  : "vel",        # cm/s, peculiar, (N, 3)
    "SmoothingLengths"            : "h",          # cm, physical
    "Masses"                      : "mass",       # g
    "Densities"                   : "rho",        # g/cm^3, physical
    "Temperatures"                : "T",          # K
    "SpeciesFractions"            : "species",    # n_i / n_H (CHIMES, non-equilibrium), (N, 10)
    "AtomicHydrogenMasses"        : "m_HI_eq",    # g, from the equilibrium cooling tables
    "ElementMassFractions"        : "X",          # gas + dust, (N, 12)
    "ElementMassFractionsDiffuse" : "X_gas",      # gas phase only, (N, 12)
    "MetalMassFractions"          : "Z",          # total metal mass fraction
}

# ---------------------------------------------------------------------------
# Native COLIBRE LOS files (legacy: kept for consistency checks)
# ---------------------------------------------------------------------------
def los_catalogue(los_dir=LOS_DIR, zmin=4.0, zmax=5.5):
    """Header-only scan of the LOS files with zmin <= z <= zmax.
       Returns file numbers, redshifts and number of sightlines along x, y, z (shape (n, 3))."""
    numbers, z, per_axis = [], [], []
    for fn in sorted(glob.glob(os.path.join(los_dir, "los_*.hdf5"))):
        with h5py.File(fn, "r") as f:
            z_file = f["Header"].attrs["Redshift"][0]
            if not zmin <= z_file <= zmax:
                continue
            axes = [f[g].attrs["Zaxis"][0] for g in f if g.startswith("LOS_")]
        numbers.append(int(os.path.basename(fn)[4:8]))
        z.append(z_file)
        per_axis.append(np.bincount(axes, minlength=3))
    return np.array(numbers), np.array(z), np.array(per_axis)

def read_sightline(filename, index):
    """One sightline as a dictionary: metadata plus particle arrays (float64, physical CGS).
       The sightline runs along the axis whose index is axes[2] (0 = x, 1 = y, 2 = z) and
       crosses the perpendicular plane at (xpos, ypos) along (axes[0], axes[1])."""
    with h5py.File(filename, "r") as f:
        header = f["Header"].attrs
        group  = f[f"LOS_{index:04d}"]
        los = {
            "z"            : header["Redshift"][0],
            "a"            : header["Scale-factor"][0],
            "box_cMpc"     : header["BoxSize"][0],                                # comoving, internal units
            "unit_length"  : f["Units"].attrs["Unit length in cgs (U_L)"][0],     # cm per internal length unit
            "kernel_gamma" : f["HydroScheme"].attrs["Kernel gamma"][0],           # kernel support H = gamma * h
            "axes"         : (group.attrs["Xaxis"][0], group.attrs["Yaxis"][0], group.attrs["Zaxis"][0]),
            "xpos_cMpc"    : group.attrs["Xpos"][0],
            "ypos_cMpc"    : group.attrs["Ypos"][0],
            "species_names": [s.decode() for s in f["SubgridScheme/NamedColumns/SpeciesFractions"][:]],
            "element_names": [s.decode() for s in f["SubgridScheme/NamedColumns/ElementMassFractions"][:]],
        }
        for name, key in LOS_FIELDS.items():  # unpack the dic { filename : short_key}, e.g. name = "Coordinates", key = "pos"
            dset     = group[name]            # create the h5py.Dataset
            los[key] = dset[:].astype(np.float64) * dset.attrs[CGS_FACTOR][0] # adds a new entry to the dict with key "pos" and the array in cgs
    return los # dic with 10 medatadata and 11 array

# ---------------------------------------------------------------------------
# Gas from snapshots (built sightlines)
# ---------------------------------------------------------------------------
def read_gas(filename):
    """Gas fields needed for sightlines, in internal comoving units (cMpc, km/s, 1e10 Msun), plus metadata."""
    with h5py.File(filename, "r") as f:
        cosmo = f["Cosmology"].attrs
        gas = {
            "z"        : cosmo["Redshift"][0],
            "a"        : cosmo["Scale-factor"][0],
            "Hz"       : cosmo["H [internal units]"][0],                                     # km/s/Mpc
            "rho_crit0": cosmo["Critical density at redshift zero [internal units]"][0],     # 1e10 Msun / cMpc^3
            "L"        : f["Header"].attrs["BoxSize"][0],                                    # cMpc
            "gamma"    : f["HydroScheme"].attrs["Kernel gamma"][0],
        }
        elements = [s.decode() for s in f["SubgridScheme/NamedColumns/ElementMassFractions"][:]]
        species  = [s.decode() for s in f["SubgridScheme/NamedColumns/SpeciesFractions"][:]]

        part = f["PartType0"]
        gas["pos"]  = part["Coordinates"][:]                                          # cMpc, (N, 3), float64
        gas["vel"]  = part["Velocities"][:]                                           # km/s, peculiar, (N, 3)
        gas["mass"] = part["Masses"][:]                                               # 1e10 Msun
        gas["T"]    = part["Temperatures"][:]                                         # K
        h           = part["SmoothingLengths"][:]                                     # cMpc
        X_H         = part["ElementMassFractions"][:, elements.index("Hydrogen")]     # one column only
        x_HI        = part["SpeciesFractions"][:, species.index("HI")]                # n_HI / n_H

    gas["H"]    = gas["gamma"] * h                     # kernel radius, cMpc
    gas["m_HI"] = gas["mass"] * X_H * x_HI             # HI mass, 1e10 Msun
    return gas

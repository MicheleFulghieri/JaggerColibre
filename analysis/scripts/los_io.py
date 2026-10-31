"""Reading of the COLIBRE line-of-sight (LOS) outputs.

   Each file los_XXXX.hdf5 holds, for one output time, 105 sightlines (35 parallel to each axis).
   Group LOS_YYYY contains all gas particles whose SPH kernel intersects that sightline.
   Every dataset carries the factor that converts it to physical CGS units (cosmological a-factors included)."""

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

# Return (redshift array, file list) sorted by the name of the file
def los_redshifts(los_dir=LOS_DIR):
    """Redshift of every LOS file in los_dir and the file names, sorted by file number."""
    files = sorted(glob.glob(os.path.join(los_dir, "los_*.hdf5")))
    z = np.empty(len(files))
    for i, fn in enumerate(files):
        with h5py.File(fn, "r") as f:
            z[i] = f["Header"].attrs["Redshift"][0]
    return z, files


def read_sightline(filename, index):
    """One sightline as a dictionary: metadata plus particle arrays (float64, physical CGS).
       The sightline uns along the axis whose index is axes[2] (0 = x, 1 = y, 2 = z) and
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


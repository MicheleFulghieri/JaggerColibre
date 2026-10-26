"""Tabulated equilibrium ion fractions n_ion / n_element (any species, HI included).

   1) COLIBRE tables of S. Ploeckinger (02/2026): full CHIMES network (157 species), same UV background,
      shielding, ISRF, cosmic rays and dust as the COLIBRE runs; H and He in equilibrium.
      Abundances[T, n_H, Z, species] = log10(n_ion / n_element).
   2) Trident tables (CLOUDY, HM12 background; optically thin or self-shielded).
      Dataset per element [stage, n_H, z, T] = log10(n_ion / n_element)."""

import os
import glob

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator


EQM_TABLE_DIR = "/cosma8/data/dp004/colibre/Tables/equilibrium_rates_and_speciesfractions/abundances_full"
Z_SUN_TABLE   = 0.013088709    # SolarMetallicity of cooling_tables/cooling_rates_z*.hdf5 (Wiersma+2009b);
                               # assumed identical for abundances_full (no attribute there): to confirm with S. Ploeckinger

# Position of each species along the last axis of Abundances (chimes_dict of read_chimes.py, copied in the README)
SPECIES_INDEX = {"HI": 1, "CII": 8, "CIV": 10, "OI": 23, "MgII": 45, "SiII": 58, "SiIV": 60, "FeII": 111}

TRIDENT_TABLES = {
    "HM12 thin" : "/cosma8/data/do019/dc-fulg1/tables/trident/hm2012_hr.h5",      # optically thin
    "HM12 ss"   : "/cosma8/data/do019/dc-fulg1/tables/trident/hm2012_ss_hr.h5",   # self-shielded
}


# ---------------------------------------------------------------------------
# COLIBRE equilibrium tables (CHIMES full network)
# ---------------------------------------------------------------------------
def table_redshifts():
    """Redshifts of the table files, sorted ('z5.000_eqm.hdf5' -> 5.0), and the matching file names."""
    files   = glob.glob(os.path.join(EQM_TABLE_DIR, "z*_eqm.hdf5"))
    z_files = np.array([float(os.path.basename(fn)[1:6]) for fn in files])
    order   = np.argsort(z_files)
    return z_files[order], [files[i] for i in order]


def load_eqm_table(species, z):
    """log10 ion fraction of one species on the (log T, log n_H, log Z/Z_sun) grid, interpolated
       linearly in z between the two files that bracket z. Returns a RegularGridInterpolator."""
    z_files, files = table_redshifts()
    i = np.clip(np.searchsorted(z_files, z) - 1, 0, len(z_files) - 2)    # z_files[i] <= z <= z_files[i+1]
    w = (z - z_files[i]) / (z_files[i + 1] - z_files[i])
    k = SPECIES_INDEX[species]
    with h5py.File(files[i], "r") as f:
        log_T = f["TableBins/Temperatures"][:].astype(np.float64)
        log_n = f["TableBins/Densities"][:].astype(np.float64)
        log_Z = f["TableBins/Metallicities"][:].astype(np.float64)
        lo    = f["Abundances"][:, :, :, k].astype(np.float64)              # reads only this species
    with h5py.File(files[i + 1], "r") as f:
        hi    = f["Abundances"][:, :, :, k].astype(np.float64)
    return RegularGridInterpolator((log_T, log_n, log_Z), (1 - w) * lo + w * hi)


def eqm_ion_fraction(interp, n_H, T, Z):
    """Ion fraction n_ion / n_element for arrays n_H [cm^-3, physical], T [K], Z (metal mass fraction).
       Points outside the grid take the edge value; pristine gas (Z = 0) goes to the lowest Z bin."""
    log_T, log_n, log_Z = interp.grid
    with np.errstate(divide="ignore"):                                     # log10(0) = -inf, then clipped
        points = np.column_stack([np.clip(np.log10(T),               log_T[0], log_T[-1]),
                                  np.clip(np.log10(n_H),             log_n[0], log_n[-1]),
                                  np.clip(np.log10(Z / Z_SUN_TABLE), log_Z[0], log_Z[-1])])
    return 10**interp(points)


# ---------------------------------------------------------------------------
# Trident (CLOUDY, HM12) tables
# ---------------------------------------------------------------------------
def trident_ion_fraction(table_name, element, stage, z, n_H, T):
    """Ion fraction n_ion / n_element from a Trident table, at redshift z, for arrays n_H [cm^-3] and T [K].
       element: dataset name ('H', 'C', 'O', 'Mg', 'Al', 'Si', 'Fe', ...); stage: 1 = neutral (I), 2 = II, ...
       Table axes: [stage, log n_H, z, log T]; values: log10 of the ion fraction.
       The table is first interpolated linearly in z, then bilinearly in (log n_H, log T);
       points outside the grid take the value at the edge."""
    with h5py.File(TRIDENT_TABLES[table_name], "r") as f:
        dset   = f[element]
        log_n  = dset.attrs["Parameter1"]
        z_grid = dset.attrs["Parameter2"]
        log_T  = dset.attrs["Temperature"]
        i = np.clip(np.searchsorted(z_grid, z) - 1, 0, len(z_grid) - 2)   # z_grid[i] <= z < z_grid[i+1]
        w = (z - z_grid[i]) / (z_grid[i + 1] - z_grid[i])
        table = (1 - w) * dset[stage - 1, :, i, :] + w * dset[stage - 1, :, i + 1, :]   # (n_H, T) at redshift z

    interp = RegularGridInterpolator((log_n, log_T), table)
    points = np.column_stack([np.clip(np.log10(n_H), log_n[0], log_n[-1]),
                              np.clip(np.log10(T),   log_T[0], log_T[-1])])
    return 10**interp(points)

"""Ionisation state of the gas, model 0 (step I0b of the roadmap).

   1) Analytic hydrogen equilibrium with photoionisation, collisional ionisation and recombination,
      for a given HI photoionisation rate Gamma (HM12 or COLIBRE UV background), optionally reduced
      by self-shielding (Rahmati et al. 2013).
   2) Tabulated ion fractions of Trident (CLOUDY, HM12 background; optically thin or self-shielded)."""

import os
import glob

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator


DATA_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
HM12_RATES_FILE = os.path.join(DATA_DIR, "hm12_photorates.txt")    # Haardt & Madau 2012, Table 3 (from the rabacus package)
COLIBRE_UVB_DIR = "/cosma8/data/dp004/jlvc76/COLIBRE/Tables/chimes-data/SP20_cross_sections"
TRIDENT_TABLES  = {
    "HM12 thin" : "/cosma8/data/do019/dc-fulg1/tables/trident/hm2012_hr.h5",      # optically thin
    "HM12 ss"   : "/cosma8/data/do019/dc-fulg1/tables/trident/hm2012_ss_hr.h5",   # self-shielded
}

SPEED_OF_LIGHT = 2.99792458e10     # cm/s
T_ION_HI       = 157807.0          # K, HI ionisation potential (13.6 eV) / k_B


# ---------------------------------------------------------------------------
# HI photoionisation rate of the UV background
# ---------------------------------------------------------------------------
def gamma_HI_hm12(z):
    """Gamma_HI [s^-1] of Haardt & Madau (2012) at redshift z (linear interpolation of log Gamma in z).
       Table columns: z, Gamma_HI, heating_HI, Gamma_HeI, heating_HeI, Gamma_HeII, heating_HeII, Compton."""
    table = np.loadtxt(HM12_RATES_FILE, comments="#")
    return 10**np.interp(z, table[:, 0], np.log10(table[:, 1]))


def gamma_HI_colibre(z):
    """Gamma_HI = c * n_gamma * sigma_HI [s^-1] of the COLIBRE UV background (Ploeckinger & Schaye 2020 spectrum),
       from the CHIMES cross-section files (one per z, step 0.2): n_gamma = isotropic photon density above 13.6 eV,
       sigma_HI = spectrum-averaged HI cross section. Returns (Gamma_HI, sigma_HI), interpolated linearly in z."""
    files = glob.glob(os.path.join(COLIBRE_UVB_DIR, "z*_cross_sections.hdf5"))
    z_grid, gamma, sigma = [], [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            z_grid.append(float(os.path.basename(fn)[1:6]))          # 'z5.000_cross_sections.hdf5' -> 5.0
            n_gamma = f["isotropic_photon_density"][0]
            sigma.append(f["photoion_euv/sigmaPhot"][0])             # reaction 0 = HI photoionisation
            gamma.append(SPEED_OF_LIGHT * n_gamma * sigma[-1])
    order = np.argsort(z_grid)
    z_grid, gamma, sigma = np.array(z_grid)[order], np.array(gamma)[order], np.array(sigma)[order]
    return 10**np.interp(z, z_grid, np.log10(gamma)), np.interp(z, z_grid, sigma)


# ---------------------------------------------------------------------------
# Rate coefficients for hydrogen
# ---------------------------------------------------------------------------
def alpha_A_HII(T):
    """Case-A radiative recombination coefficient of HII [cm^3 s^-1], fit of Hui & Gnedin (1997).
       Case A (recombinations to the ground state included) is appropriate for optically thin gas."""
    lam = 2.0 * T_ION_HI / T
    return 1.269e-13 * lam**1.503 / (1.0 + (lam / 0.522)**0.470)**1.923


def beta_HI(T):
    """Collisional ionisation coefficient of HI by electrons [cm^3 s^-1], fit of Cen (1992)."""
    return 5.85e-11 * np.sqrt(T) * np.exp(-T_ION_HI / T) / (1.0 + np.sqrt(T / 1e5))


def rahmati_shielding(n_H, T, gamma_uvb, sigma_nu=2.49e-18, f_g=0.17):
    """Gamma_eff / Gamma_UVB from the self-shielding fit of Rahmati et al. (2013), MNRAS 430, 2427:
       n_SSh = 6.73e-3 (sigma/2.49e-18)^(-2/3) (T/1e4)^0.17 (Gamma/1e-12)^(2/3) (f_g/0.17)^(-1/3)  [cm^-3]
       Gamma_eff/Gamma_UVB = 0.98 [1 + (n_H/n_SSh)^1.64]^(-2.28) + 0.02 [1 + n_H/n_SSh]^(-0.84)."""
    n_SSh = (6.73e-3 * (sigma_nu / 2.49e-18)**(-2/3) * (T / 1e4)**0.17
             * (gamma_uvb / 1e-12)**(2/3) * (f_g / 0.17)**(-1/3))
    x = n_H / n_SSh
    return 0.98 * (1.0 + x**1.64)**(-2.28) + 0.02 * (1.0 + x)**(-0.84)


def equilibrium_f_HI(n_H, T, gamma, eta):
    """Neutral fraction f = n_HI/n_H in ionisation equilibrium (all arrays in cgs, physical n_H):
           f (Gamma + beta n_e) = alpha n_e (1 - f),   n_e = eta n_H (1 - f)
       eta = 1 + Y/(4 X_H): electrons from H plus singly ionised He, assumed to follow H.
       This is the quadratic  A f^2 - B f + C = 0  with
           A = (alpha + beta) eta n_H,  B = (2 alpha + beta) eta n_H + Gamma,  C = alpha eta n_H,
       whose physical (smaller) root is written as f = 2C / (B + sqrt(B^2 - 4AC)) to avoid cancellation."""
    alpha = alpha_A_HII(T)
    beta  = beta_HI(T)
    k = eta * n_H
    A = (alpha + beta) * k
    B = (2.0 * alpha + beta) * k + gamma
    C = alpha * k
    return 2.0 * C / (B + np.sqrt(np.maximum(B**2 - 4.0 * A * C, 0.0)))


# ---------------------------------------------------------------------------
# Trident (CLOUDY) ion tables
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

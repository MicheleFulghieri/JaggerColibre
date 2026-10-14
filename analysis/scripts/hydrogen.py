"""Neutral hydrogen: particle HI quantities, global budget, slab HI columns, absorber classes, X."""

import numpy as np
import unyt
import astropy.units as u
from astropy import constants as const
import swiftsimio as sw

from projections import compute_projection_map


hydrogen_mass = unyt.mh.to("g")    # hydrogen atom mass (unyt constant) 1.674e-24 g
N_HI_LLS      = 10**17.2           # cm^-2, Lyman limit system threshold (tau_912 = 1, e.g. Fumagalli et al, 2013)
N_HI_DLA      = 2e20               # cm^-2, damped Lyman-alpha threshold (Wolfe et al. 1986, ApJS 61, 249)

# log10 N_HI [cm^-2] boundaries of the absorber classes
ABSORBER_CLASSES = {"forest": (-np.inf, 17.2), "LLS": (17.2, 19.0), "subDLA": (19.0, 20.3), "DLA": (20.3, np.inf)}


def compute_hydrogen_fields(gas):
    """Hydrogen quantities of each gas particle (comoving, as stored in the snapshot):
       m_HI = f_HI * X_H * m          [Msun]   HI mass: weight for the absorber diagnostics
       n_H  = X_H * rho / m_H         [cm^-3]  total hydrogen number density
       f_HI = n_HI / n_H (CHIMES species fraction), X_H = hydrogen mass fraction.
       (n_HI = f_HI * n_H, if needed.)"""

    X_H  = gas.element_mass_fractions.hydrogen
    f_HI = gas.species_fractions.HI

    m_HI = (f_HI * X_H * gas.masses).to("Msun")       # Msun, not g: in g the values would overflow float32
    rho  = gas.densities.to("g/cm**3")                # convert first: in g/Mpc**3 the values would overflow float32
    n_H  = (X_H * rho / hydrogen_mass).to("cm**-3")
    return m_HI, n_H


def print_HI_budget(gas, m_HI, cosmo, boxsize):
    """Global gas and HI budget of the box. Omega_HI = rho_HI,comoving / rho_crit,0."""

    M_gas = gas.masses.to("Msun").value.sum(dtype=np.float64)              # accumulate in float64
    M_H   = (gas.element_mass_fractions.hydrogen * gas.masses).to("Msun").value.sum(dtype=np.float64)
    M_HI  = m_HI.to("Msun").value.sum(dtype=np.float64)

    V_box     = np.prod(boxsize.to("Mpc").value)                           # comoving volume [cMpc^3]
    rho_crit0 = cosmo.critical_density0.to(u.Msun / u.Mpc**3).value        # astropy quantity -> float [Msun/Mpc^3]
    Omega_HI  = M_HI / V_box / rho_crit0

    print(f"\n[+] Global HI budget:")
    print(f"     Total gas mass                 :     {M_gas:.3e} Msun")
    print(f"     Total hydrogen mass            :     {M_H:.3e} Msun")
    print(f"     Total HI mass                  :     {M_HI:.3e} Msun")
    print(f"     Neutral fraction M_HI / M_H    :     {M_HI / M_H:.3e}")
    print(f"     Omega_HI                       :     {Omega_HI:.3e}")


def slab_absorption_distance(thickness_cMpc, cosmo, z):
    # Relation in Fumagalli et al. (2013), from Bahcall & Peebles (1969)
    """Absorption distance of a slab of comoving thickness Delta_chi at redshift z:
       dX = (1+z)^2 H0/H(z) dz  and  dz = H(z) dchi / c   ->   Delta_X = (1+z)^2 H0 Delta_chi / c"""
    delta_chi = thickness_cMpc * u.Mpc                       # float -> astropy Quantity
    # .decompose() simplify the units
    return ((1 + z)**2 * cosmo.H0 * delta_chi / const.c).decompose().value


def make_slab(meta, thickness_cMpc, resolution):
    """Slab of comoving thickness thickness_cMpc centred in the box along z, covering the full x-y area.
       Returns (slab_kwargs for compute_projection_map, z_box_c = slab centre [cMpc])."""
    L_box   = meta.boxsize.to('Mpc').value
    z_box_c = 0.5 * L_box[2]
    slab_region = sw.cosmo_array([0, L_box[0], 0, L_box[1], z_box_c - thickness_cMpc/2, z_box_c + thickness_cMpc/2], "Mpc",
                                 comoving=True, scale_factor=meta.a, scale_exponent=1)
    slab_kwargs = dict(resolution=resolution, region=slab_region, parallel=True)
    return slab_kwargs, z_box_c


def HI_column_density(snap, slab_kwargs):
    """Sigma_HI map (comoving, also used as denominator of HI-weighted maps) and N_HI map
       [HI atoms per physical cm^2]. Needs snap.gas.m_HI attached."""
    sigma_HI = compute_projection_map(snap, field="m_HI", weight=None, physical=False, **slab_kwargs)
    N_HI     = (sigma_HI.to_physical().to("g/cm**2") / hydrogen_mass).to("cm**-2")
    return sigma_HI, N_HI

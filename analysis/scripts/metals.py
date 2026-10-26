"""Elements, dust and metallicity diagnostics """

import numpy as np
import h5py
import unyt

from projections import compute_projection_map
from hydrogen import ABSORBER_CLASSES


Z_sun = 0.0134             # solar metal mass fraction, Asplund et al. (2009)

# Solar photospheric abundances 12 + log10(n_X/n_H), Asplund et al. (2009), Table 1
SOLAR_LOG_EPS = {"hydrogen": 12.00, "helium": 10.93, "carbon": 8.43, "nitrogen": 7.83, "oxygen": 8.69,
                 "neon": 7.93, "magnesium": 7.60, "silicon": 7.51, "iron": 7.50}
# Standard atomic weights [amu]
ATOMIC_MASS = {"hydrogen": 1.008, "helium": 4.003, "carbon": 12.011, "nitrogen": 14.007, "oxygen": 15.999,
               "neon": 20.180, "magnesium": 24.305, "silicon": 28.085, "iron": 55.845}
METALS = ["carbon", "nitrogen", "oxygen", "neon", "magnesium", "silicon", "iron"]


def print_metal_budget(gas, snap_filename):
    """Per metal: total mass, dust fraction, closure X_tot = X_diffuse + X_dust, enriched mass fraction,
       and global abundance [X/H]_glob = log10( sum m X_X/A_X / sum m X_H/A_H ) - log10(n_X/n_H)_sun
       for all atoms (tot) and for the gas phase only (gas).
       X_dust,el = sum_g D_g * M[g, el], M = GrainToElementMapping (grain -> element mass fractions)."""
    with h5py.File(snap_filename, "r") as f:
        grain_to_element = f["SubgridScheme/GrainToElementMapping"][:]           # (6 grains, 12 elements)
    elements = gas.element_mass_fractions.named_columns                          # same order as the matrix columns
    D = np.column_stack([getattr(gas.dust_mass_fractions, g).value.astype(np.float64)
                         for g in gas.dust_mass_fractions.named_columns])        # (N particles, 6 grains)
    m = gas.masses.to("Msun").value.astype(np.float64)
    n_H_atoms = np.sum(m * gas.element_mass_fractions.hydrogen.value.astype(np.float64)) / ATOMIC_MASS["hydrogen"]

    print(f"\n[+] Metal budget (mass weighted)")
    print(f"     {'element':<10} {'M_el [Msun]':>12} {'f_dust':>9} {'closure':>9} {'f_enriched':>11} "
          f"{'[X/H]glob_tot':>14} {'[X/H]glob_gas':>14}")
    for i, el in enumerate(elements):
        if el not in METALS:
            continue
        X_tot  = getattr(gas.element_mass_fractions, el).value.astype(np.float64)
        X_dif  = getattr(gas.element_mass_fractions_diffuse, el).value.astype(np.float64)
        X_dust = D @ grain_to_element[:, i]                                        # matrix-vector product
        M_tot  = np.sum(m * X_tot)
        solar  = SOLAR_LOG_EPS[el] - 12.0                                          # log10(n_X/n_H)_sun
        glob_tot = np.log10(M_tot / ATOMIC_MASS[el] / n_H_atoms) - solar
        glob_gas = np.log10(np.sum(m * X_dif) / ATOMIC_MASS[el] / n_H_atoms) - solar
        print(f"     {el:<10} {M_tot:12.3e} {np.sum(m * X_dust) / M_tot:9.3e} "
              f"{np.sum(m * (X_dif + X_dust)) / M_tot:9.4f} {np.sum(m[X_tot > 0]) / np.sum(m):11.3e} "
              f"{glob_tot:14.3f} {glob_gas:14.3f}")

    Z = gas.metal_mass_fractions.value.astype(np.float64)
    M_Z = np.sum(m * Z)
    for label, field in [("SNII", gas.metal_mass_fractions_from_snii), ("SNIa", gas.metal_mass_fractions_from_snia),
                         ("AGB", gas.metal_mass_fractions_from_agb)]:
        print(f"     Metal mass from {label:<5}         :     {np.sum(m * field.value.astype(np.float64)) / M_Z:.3e}")


# Abundances per particles
def number_ratio(gas, element, diffuse=False):
    """n_X / n_H of each particle (float64). diffuse=False: gas + dust; diffuse=True: gas phase only."""
    fractions = gas.element_mass_fractions_diffuse if diffuse else gas.element_mass_fractions
    X_el = getattr(fractions, element).value.astype(np.float64)
    X_H  = gas.element_mass_fractions.hydrogen.value.astype(np.float64)
    return (X_el / ATOMIC_MASS[element]) / (X_H / ATOMIC_MASS["hydrogen"])


def bracket(ratio, element, reference="hydrogen"):
    """[X/Y] = log10(n_X/n_Y) - log10(n_X/n_Y)_sun. Pristine gas (ratio = 0) -> -inf, 0/0 -> nan."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(ratio) - (SOLAR_LOG_EPS[element] - SOLAR_LOG_EPS[reference])


def depletion(gas, element):
    """delta_X = log10(X_diffuse / X_total) of each particle: 0 = no dust, -1 = 90% of X locked in grains.
       Pristine gas (0/0) -> nan."""
    X_tot = getattr(gas.element_mass_fractions, element).value.astype(np.float64)
    X_dif = getattr(gas.element_mass_fractions_diffuse, element).value.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(X_dif / X_tot)


def element_column_density(snap, element, diffuse=True, neutral=False, **slab_kwargs):
    """Column density N_X [cm^-2, physical] of an element in the slab.
       neutral=True: only the X atoms associated with HI (m_X * f_HI). For oxygen this is the OI proxy
       (charge exchange keeps x_OI ~ x_HI); for other elements it is NOT an ion column."""
    fractions = snap.gas.element_mass_fractions_diffuse if diffuse else snap.gas.element_mass_fractions
    mass_X = getattr(fractions, element) * snap.gas.masses
    if neutral:
        mass_X = mass_X * snap.gas.species_fractions.HI
    tmp_name = f"_tmp_m_{element}"
    setattr(snap.gas, tmp_name, mass_X)
    try:
        sigma = compute_projection_map(snap, field=tmp_name, weight=None, physical=False, **slab_kwargs)
    finally:
        delattr(snap.gas, tmp_name)
    return (sigma.to_physical().to("g/cm**2") / (ATOMIC_MASS[element] * unyt.amu)).to("cm**-2")


def print_absorber_classes(logN_HI, maps):
    """Covering fraction and 16/50/84th percentiles of each map (e.g. [O/H]_HI, [Si/O]) per N_HI class.
       maps: {label: 2D numpy array}, same shape as logN_HI. Non-finite pixels (pristine) are excluded and counted."""
    print(f"\n[+] Absorber classes (pixel statistics)")
    for name, (lo, hi) in ABSORBER_CLASSES.items():
        sel = (logN_HI >= lo) & (logN_HI < hi)
        print(f"\n     {name}: covering fraction = {sel.mean():.3e}")
        for label, m in maps.items():
            v = m[sel]
            v = v[np.isfinite(v)]
            if v.size == 0:
                print(f"       {label:<12} no enriched pixels")
                continue
            p16, p50, p84 = np.percentile(v, [16, 50, 84])
            print(f"       {label:<12} {p16:8.2f} {p50:8.2f} {p84:8.2f}   (enriched: {v.size / sel.sum():.2f})")

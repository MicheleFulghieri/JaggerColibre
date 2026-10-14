"""SPH projections of gas fields with swiftsimio, and their plotting."""

from operator import attrgetter

import numpy as np
from matplotlib.colors import LogNorm, Normalize
import swiftsimio as sw


def compute_projection_map(snap, field="masses", weight=None, resolution=256, region=None,
                           weight_map=None, physical=True, parallel=False, periodic=True):
    """2D projection along z of a gas field, smoothed with the SPH kernel.

       weight=None  -> surface density of field in each pixel i:
                       Sigma_i = sum_j q_j W_ij / A_pix               [q / length^2]
       weight='w'   -> w-weighted average of field in each pixel i:
                       <q>_i = sum_j q_j w_j W_ij / sum_j w_j W_ij     [q]

       field and weight are swiftsimio names of snap.gas (dotted names allowed,
       e.g. 'element_mass_fractions.oxygen').
       region     : [x_min, x_max, y_min, y_max, (z_min, z_max)] cosmo_array, None = whole box.
       weight_map : already computed projection of weight (comoving, same resolution and
                    region), used as denominator instead of projecting weight again.
       Returns a cosmo_array (resolution x resolution) indexed as [ix, iy]:
       transpose it before imshow."""

    project_kwargs = dict(resolution=resolution, region=region, parallel=parallel, periodic=periodic)

    if weight is None:
        proj_map = sw.visualisation.projection.project_gas(snap, project=field, **project_kwargs)  # **kwargs  to allow functions to accept a unknown number of arguments
    else:
        # Temporary particle field q*w attached to snap.gas, so that project_gas can find it by name
        tmp_name = f"_tmp_{field}_x_{weight}".replace(".", "_")
        setattr(snap.gas, tmp_name, attrgetter(field)(snap.gas) * attrgetter(weight)(snap.gas))
        try:
            num = sw.visualisation.projection.project_gas(snap, project=tmp_name, **project_kwargs)
            if weight_map is None:
                den = sw.visualisation.projection.project_gas(snap, project=weight, **project_kwargs)
            else:
                den = weight_map
        finally:
            delattr(snap.gas, tmp_name)                       # free the memory of the temporary field
        with np.errstate(divide="ignore", invalid="ignore"):  # no nan due to division by 0 errors
            proj_map = num / den                              # empty pixels (den = 0) -> nan

    if physical:  # converting from coomoving to physical coords
        proj_map = proj_map.to_physical()

    return proj_map


def plot_projection_map(proj_map, ax, Lbox, units, cbar_label, cmap="magma", log=True,
                        vmin=None, vmax=None, title=None):
    """Draw on ax a map returned by compute_projection_map, converted to units.
       Lbox is the comoving box size in cMpc. Returns the image object."""

    image = proj_map.to(units).value.T       # .T since in proj_map rows are x, cols y: imshow wants rows = vertical ax

    im = ax.imshow(image,
                   origin = 'lower',
                   extent = [0, Lbox[0], 0, Lbox[1]],
                   cmap   = cmap,
                   norm   = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax))

    ax.set_box_aspect(1)
    ax.set_xlabel(r'$\rm X \ [ cMpc ]$')
    ax.set_ylabel(r'$\rm Y \ [ cMpc ]$')
    if title is not None:
        ax.set_title(title, fontsize=11)

    cbar = ax.figure.colorbar(mappable=im, ax=ax, orientation='vertical', pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)

    return im

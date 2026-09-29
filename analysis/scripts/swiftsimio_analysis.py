import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
matplotlib.use("Agg")
import numpy as np
import os
import h5py
import unyt 
from datetime import datetime
import swiftsimio as sw


# ---------------------------------------------------------------------------
# Useful paths
# ---------------------------------------------------------------------------
snap_dir      = '/cosma8/data/dp004/colibre/Runs'
run_dir       = 'L0025N0188/Thermal'
snap_nr       = 48  # z = 5.0
snap_filename = f'{snap_dir}/{run_dir}/SOAP-HBT/colibre_with_SOAP_membership_{snap_nr:04}.hdf5'
save_path     = '/cosma8/data/do019/dc-fulg1/outputs/swiftsimio_analysis'



# ---------------------------------------------------------------------------
# USEFUL FUNCTIONS AND CLASSES
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------
def make_dirs(base):
    subdirs = ["Visualization2D"]
    for d in subdirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)



# ---------------------------------------------------------------------------
# MAIN 
# ---------------------------------------------------------------------------
def main():
    # ---------------------------------------------------------------------------
    # Directory setting up
    # ---------------------------------------------------------------------------
    make_dirs(save_path)

    # ---------------------------------------------------------------------------
    # Virtual snapshot load and metadata
    # ---------------------------------------------------------------------------
    snap = sw.load(snap_filename)
    meta = snap.metadata                       # metadata method
    L_box    = meta.boxsize.to('Mpc').value
    cosmo    = meta.cosmology
    redshift = meta.z
    time     = meta.t.to('Myr').value
    
    print(f"\n----------- Inspecting the Colibre snapshot {run_dir} -----------")
    print(f"     Available_groups               :     {snap}")
    print(f"     Box size                       :     {meta.boxsize}")
    print(f"     Cosmic Time                    :     {time:.3f} Myr")
    print(f"     Redshift (scale factor)        :     {meta.z:.2f} ({meta.a:.2f})")
    print(f"     Number of Gas particles        :     {meta.n_gas}")
    print(f"     Number of DM particles         :     {meta.n_dark_matter}")
    print(f"     Number of stellar particles    :     {meta.n_stars}")
    print(f"     Number of black hole particles :     {meta.n_black_holes}")


    # ---------------------------------------------------------------------------
    # Mass Mapping
    # ---------------------------------------------------------------------------
    extent = [0, L_box[0], 0, L_box[1]]

    mass_map = sw.visualisation.projection.project_gas(   # projection smoothed via Wendland-C2 kernel
        snap,
        resolution=256,
        project="masses",
        parallel=True,       # construct the image in (thread) parallel
        periodic=True,       # PBCs
    )

    mass_map = mass_map.to('Msun/kpc**2').value

    fig_proj_gas, ax_proj_gas = plt.subplots()
    im_histo = ax_proj_gas.imshow(mass_map,       
                              origin = 'lower',
                              extent = extent,
                              cmap = 'magma',
                              norm = LogNorm())
    ax_proj_gas.set_box_aspect(1)
    ax_proj_gas.set_xlabel(rf'$\rm X \ [Mpc / h^{-1}]$')
    ax_proj_gas.set_ylabel(rf'$\rm Y \ [Mpc / h^{-1}]$')

    fig_proj_gas.savefig(os.path.join(save_path, "Visualization2D", "Gas_xy2d_hist.png"), dpi=300)
    plt.close(fig_proj_gas)





    print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()


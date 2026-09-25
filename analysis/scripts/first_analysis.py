import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
matplotlib.use("Agg")
import numpy as np
import os
import h5py
from datetime import datetime


# ---------------------------------------------------------------------------
# Useful paths
# ---------------------------------------------------------------------------
file_path = '/cosma8/data/dp004/colibre/Runs/L0012N0094/Thermal/snapshots/colibre_0000'
save_path = '/cosma8/data/do019/dc-fulg1/outputs/first_analysis'


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
# Class for inspecting and loading the dataset
# ---------------------------------------------------------------------------
class ColibreSnapshot:
   def __init__(self, filepath):
      self.filepath = filepath
      self.f = h5py.File(self.filepath, 'r')     
      self.header = self.f['Header'].attrs
      self.redshift = self.header['Redshift'][0]
      self.box_size = self.header['BoxSize']

   def print_snap_info(self):
      header = self.f['Header'].attrs
      
      redshift  = header['Redshift'][0]  # [0] to have the float 3.0
      L_box_Mpc = header['BoxSize']
      pgas_count  = self.f['PartType0/Coordinates'].shape[0]
      pdm_count   = self.f['PartType1/Coordinates'].shape[0] 
      pstar_count = self.f['PartType4/Coordinates'].shape[0]

      # Clean print if run_name is in bytes
      run_name = self.header['RunName']
      if isinstance(run_name, bytes):
         run_name = run_name.decode('utf-8')

      print(f"---- Inspecting the Colibre snap {header['RunName']} ----")
      print(f"     Redshift                    :     {redshift}")
      print(f"     Box dimension (Mpc/h)       :     {L_box_Mpc}")
      print(f"     Number of Gas particles     :     {pgas_count}")
      print(f"     Number of DM particles      :     {pdm_count}")
      print(f"     Number of stellar particles :     {pstar_count}")
      

   def get_dataset(self, part_type, dataset_name):
      """ Extract a generic dataset (e.g. 'Coordinates', 'Masses', 'Temperatures')"""
      path = f"{part_type}/{dataset_name}"
      if path in self.f:
         return self.f[path][:]
      else:
         print(f"Dataset {dataset_name} not found for {part_type}!")
         return None

   def get_sim_parameter(self, attr_name):
      header = self.f['Header'].attrs
      if attr_name in header:
         return header[attr_name]
      else:
         print(f"Dataset {attr_name} not found in header!")
         return None

   def close(self): # close the file in the end
      self.f.close()


# ---------------------------------------------------------------------------
#  Gas, dm and star particles visualization
# ---------------------------------------------------------------------------
fig, (ax_gas, ax_dm, ax_star) = plt.subplots(1, 3, figsize=(15, 5))

def sample_and_plot_parts(nplot, pcoords, ax_n, ls='k.', ms=0.5, label=None):
  """Sample nplots random points from x and y array of coordinates of a particle and plot
     them in the ax_n of a (1,3) subplot (set n = gas, dm, star)  """
  num_parts = pcoords.shape[0]
  nplot     = np.minimum(nplot, num_parts)  # to be sure 
  idx = np.random.choice(pcoords.shape[0], size=nplot, replace=False)
  pcoords_plot = pcoords[idx]
  ax_n.plot(pcoords_plot[:,0], pcoords_plot[:,1], ls, markersize=ms, label=label)
  ax_n.set_xlabel(r'$\rm X \ [Mpc / h^{-1}]$')
  ax_n.set_ylabel(r'$\rm Y \ [Mpc / h^{-1}]$')
  ax_n.set_box_aspect(1)
  ax_n.legend()


# ---------------------------------------------------------------------------
#  Gas, dm and star 2D histo
# ---------------------------------------------------------------------------
def histo2d_and_show_parts(pcoords, fig_histo, ax_histo, ptype, Lbox, nbins=200, absc=0, ord=1):
   """Extract the (non-)weighted 2d nummpy histogram of the box distribution of particles and
      plot them"""
   
   histo_2d, xedges, yedges = np.histogram2d(
      pcoords[:, absc],
      pcoords[:, ord],
      range = [[0, Lbox[absc]], [0, Lbox[ord]]],
      bins = nbins
   )

   im_histo = ax_histo.imshow(histo_2d,
                              origin = 'lower',
                              extent = [0, Lbox[absc], 0, Lbox[ord]],
                              cmap = 'magma',
                              norm = LogNorm())

   ax_histo.set_box_aspect(1)
   coord_names = {0: 'X', 1: 'Y', 2: 'Z'}
   x = coord_names[absc]
   y = coord_names[ord]

   ax_histo.set_xlabel(rf'$\rm {x} \ [Mpc / h^{-1}]$')
   ax_histo.set_ylabel(rf'$\rm {y} \ [Mpc / h^{-1}]$')

   cbar_histo = fig_histo.colorbar(mappable=im_histo, ax=ax_histo, orientation='vertical', shrink=1.0, pad=0.04)
   cbar_histo.set_label(rf'$\rm Number \ of \ {ptype} \ Particles$', fontsize=12)



# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------
def main():

   # ---------------------------------------------------------------------------
   # Directories setup
   # ---------------------------------------------------------------------------
   make_dirs("/data/mfulghieri/jagger/outputs/first_analysis")

   # ---------------------------------------------------------------------------
   # Inspect and load the snapshot
   # --------------------------------------------------------------------------- 
   snap = ColibreSnapshot(file_path)                   # instantiating the class object
   snap.print_snap_info()
   L_box_Mpc = snap.box_size
   redshift  = snap.redshift
   pgas  = snap.get_dataset('PartType0', 'Coordinates')  # (6573595, 3) array with (x, y, z) for each part
   pdm   = snap.get_dataset('PartType1', 'Coordinates')  # (26578688, 3)
   pstar = snap.get_dataset('PartType4', 'Coordinates')  # (70698, 3)
   snap.close()

   # ---------------------------------------------------------------------------
   # Plot of a sample of particles: Gas, DM, Stars
   # --------------------------------------------------------------------------- 
   sample_and_plot_parts(int(2*1e4), pgas, ax_gas, ls='k.', ms=0.5, label='Gas Particles')
   sample_and_plot_parts(int(2*1e4), pdm, ax_dm, ls='k.', ms=0.5, label='DM Particles')
   sample_and_plot_parts(int(1e4), pstar, ax_star, ls='y.', ms=1, label='Stellar Particles')

   fig.savefig(os.path.join(save_path, "Visualization2D", "Gas_dm_stars_xy.png"), dpi=200)
   plt.close(fig)


   # ---- 2D histo: Gas, DM, Stars ----
   fig_ghist, ax_ghist = plt.subplots()
   histo2d_and_show_parts(pgas, fig_ghist, ax_ghist, ptype='Gas', nbins=200, absc=0, ord=1, Lbox=L_box_Mpc)
   fig_ghist.savefig(os.path.join(save_path, "Visualization2D", "Gas_xy2d_hist.png"), dpi=300)
   plt.close(fig_ghist)

   fig_dmhist, ax_dmhist = plt.subplots()
   histo2d_and_show_parts(pdm, fig_dmhist, ax_dmhist, Lbox=L_box_Mpc, ptype='DM', nbins=200, absc=0, ord=1)
   fig_dmhist.savefig(os.path.join(save_path, "Visualization2D", "DM_xy2d_hist.png"), dpi=300)
   plt.close(fig_dmhist)

   fig_shist, ax_shist = plt.subplots()
   histo2d_and_show_parts(pstar, fig_shist, ax_shist, Lbox=L_box_Mpc, ptype='Stars', nbins=200, absc=0, ord=1)
   fig_shist.savefig(os.path.join(save_path, "Visualization2D", "Stars_xy2d_hist.png"), dpi=300)
   plt.close(fig_shist)


   print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()



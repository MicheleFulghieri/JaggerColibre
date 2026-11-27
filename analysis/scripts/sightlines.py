"""Own sightlines through the gas of a snapshot (internal comoving units: 
   cMpc, km/s, 1e10 Msun)."""

import numpy as np

def select_particles(gas, axis, x0, y0):
    """Gas particles whose kernel (radius H = gamma h) intersects the line parallel to `axis` (0 = x, 1 = y, 2 = z)
       through (x0, y0) [cMpc] on the two perpendicular axes.
       Returns the particle indices and their impact parameters b [cMpc] (periodic box)."""
    #   axis = "z": the line runs along z, (x0, y0) are its (x, y)
    #   axis = "x": the line runs along x, (x0, y0) are its (y, z)
    #   axis = "y": the line runs along y, (x0, y0) are its (x, z)
    t1, t2 = [i for i in range(3) if i != axis]
    L  = gas["L"]
    dx = gas["pos"][:, t1] - x0                # component of the distance in the plane
    dy = gas["pos"][:, t2] - y0
    dx -= L * np.round(dx / L)                 # minimum image
    dy -= L * np.round(dy / L)
    b   = np.hypot(dx, dy)                     # impact parameter
    idx = np.flatnonzero(b < gas["H"])
    return idx, b[idx]

# Tranverse axis in cyclic order: t1, t2 = (axis + 1) % 3, (axis + 2) % 3
#  t1, t2 = (axis + 1) % 3, (axis + 2) % 3    # e.g. axis = 2 -> (0, 1) -> x and y cyclic order
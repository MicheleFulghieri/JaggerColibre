"""SPH kernel of COLIBRE/SWIFT: quartic spline (M5), Dehnen & Aly (2012), as in SWIFT src/kernel_hydro.h.

   W(r, h) = C w(x) / H^3,  H = gamma h,  x = r / H,
   w(x) = (1-x)^4_+ - 5 (3/5-x)^4_+ + 10 (1/5-x)^4_+,  (y)_+ = max(y, 0),  C = 15625 / (512 pi)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from colibre_io import save_figure


KERNEL_CONSTANT = 15625.0 / (512.0 * np.pi)      # 3D normalisation C

def quartic_spline(x):
    """Dimensionless M5 kernel shape w(x), x = r / H. Zero for x >= 1."""
    x = np.asarray(x, dtype=np.float64)
    return (np.maximum(1.0 - x, 0.0)**4
            - 5.0  * np.maximum(0.6 - x, 0.0)**4
            + 10.0 * np.maximum(0.2 - x, 0.0)**4)

def kernel_3d(r, h, gamma):
    """W(r, h) [1/length^3] for distance r and smoothing length h (same length units)."""
    H = gamma * h
    return KERNEL_CONSTANT * quartic_spline(r / H) / H**3

if __name__ == "__main__":
    # Normalisation: int_0^1 4 pi x^2 C w(x) dx must be 1 (H cancels)    
    norm, err = quad(lambda x: 4.0 * np.pi * x**2 * KERNEL_CONSTANT * quartic_spline(x), 0.0, 1.0,
                     points=[0.2, 0.6]) # points identifies the sub-intervals where fourth derivative exhibits a jump, to make adaptive integration more precise

    print(f"scipy quad                 : {norm:.10f}  (error estimate {err:.1e})")
    print(f"w(0), w(0.6), w(1), w(1.5) : {quartic_spline([0.0, 0.6, 1.0, 1.5])}")

    # # Kernel visualization
    # x = np.linspace(0.0, 1.0, 500)
    # quartic_spline = quartic_spline(x)
    # kernel = 4.0 * np.pi * x**2 * KERNEL_CONSTANT * quartic_spline(x)

    # fig, ax = plt.subplots()
    # ax.plot(x, quartic_spline, 'b--', linewidth=1.6, label='Quartic Spline (M5)')
    # ax.plot(x, kernel, color='red', label='Quartic Spline (M5)')
    # ax.legend(fontsize=9)
    # ax.grid(True, which='both', alpha=0.3)
    # fig.tight_layout()

    # save_figure(fig, save_path, "LoS", "M5_kernel.png")
    


# srun -p cosma-analyse -A do019 -t 00:10:00 --pty /bin/bash

# source /cosma/local/anaconda3/202512/etc/profile.d/conda.sh && conda activate jagger

# "$CONDA_PREFIX/bin/python" /cosma8/data/do019/dc-fulg1/analysis/scripts/sph_kernel.py
"""SPH kernel of COLIBRE/SWIFT: quartic spline (M5), Dehnen & Aly (2012), as in SWIFT src/kernel_hydro.h.

   W(r, h) = C w(x) / H^3,  H = gamma h,  x = r / H,
   w(x) = (1-x)^4_+ - 5 (3/5-x)^4_+ + 10 (1/5-x)^4_+,  (y)_+ = max(y, 0),  C = 15625 / (512 pi)."""

import numpy as np
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

def kernel_column(beta):
    """Dimensionless column of the M5 kernel through the whole line at impact parameter beta = b / H:
       Sigma(beta) = 2 int_0^sqrt(1-beta^2) C w(sqrt(beta^2 + s^2)) ds,  zero for beta >= 1.
       The physical column of a particle of mass m is  m * Sigma(b / H) / H^2."""
    if beta >= 1.0:
        return 0.0
    s_max = np.sqrt(1.0 - beta**2)
    knots = [np.sqrt(xk**2 - beta**2) for xk in (0.2, 0.6) if xk > beta]   # kernel knots seen along the line
    value, _ = quad(lambda s: KERNEL_CONSTANT * float(quartic_spline(np.sqrt(beta**2 + s**2))),
                    0.0, s_max, points=knots or None)
    return 2.0 * value

if __name__ == "__main__":
    # 3D Kernel Normalisation: int_0^1 4 pi x^2 C w(x) dx must be 1 (H cancels)    
    norm_3d, err_3d = quad(lambda x: 4.0 * np.pi * x**2 * KERNEL_CONSTANT * quartic_spline(x), 0.0, 1.0,
                           points=[0.2, 0.6]) # points identifies the sub-intervals where fourth derivative exhibits a jump, to make adaptive integration more precise

    print(f"3D scipy quad normalisation     : {norm_3d:.10f}  (error estimate {err_3d:.1e})")
    print(f"w(0), w(0.6), w(1), w(1.5)      : {quartic_spline([0.0, 0.6, 1.0, 1.5])}")

    # 2D Kernel Normalisation: 2 pi int_0^1 Sigma(beta) beta dbeta must be 1 (mass conservation along sightlines)
    norm_2d, err_2d = quad(lambda b: 2.0 * np.pi * b * kernel_column(b), 0.0, 1.0, points=[0.2, 0.6])
    print("\n")
    print(f"2D scipy quad normalisation     : {norm_2d:.10f}  (error estimate {err_2d:.1e})")
    print(f"Sigma(0) vs 15 / (2 pi)         : {kernel_column(0.0):.10f}  {15.0 / (2.0 * np.pi):.10f}")



    


# srun -p cosma-analyse -A do019 -t 00:10:00 --pty /bin/bash

# source /cosma/local/anaconda3/202512/etc/profile.d/conda.sh && conda activate jagger

# "$CONDA_PREFIX/bin/python" /cosma8/data/do019/dc-fulg1/analysis/scripts/sph_kernel.py
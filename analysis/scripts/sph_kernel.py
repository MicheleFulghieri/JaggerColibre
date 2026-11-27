"""SPH kernel of COLIBRE/SWIFT: quartic spline (M5), Dehnen & Aly (2012), as in SWIFT src/kernel_hydro.h.

   W(r, h) = C w(x) / H^3,  H = gamma h,  x = r / H,
   w(x) = (1-x)^4_+ - 5 (3/5-x)^4_+ + 10 (1/5-x)^4_+,  (y)_+ = max(y, 0),  C = 15625 / (512 pi)."""

import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

KERNEL_CONSTANT = 15625.0 / (512.0 * np.pi)      # 3D normalisation C
BETA_MAX = 0.999 # Sigma(0.999) / Sigma(0) = 1.5e-13: particles beyond contribute nothing, < 1 to avoid degenerate zero-length chord (0/0) at beta = 1

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

def column_tables(n_beta=256, n_u=1001):
    """Tables of the M5 kernel along a line, in units of the kernel radius H.
       beta  : impact parameters b / H in [0, BETA_MAX]                       (n_beta,)
       sigma : total column Sigma(beta), as kernel_column                      (n_beta,)
       u     : line coordinate s / s_max in [-1, 1], s_max = sqrt(1 - beta^2)  (n_u,)
       P     : fraction of the column at coordinate < u                        (n_beta, n_u), P(-1) = 0, P(1) = 1"""
    # 1D grids: beta capped at BETA_MAX < 1 to avoid degenerate zero-length chord (0/0) at beta = 1
    beta  = np.linspace(0.0, BETA_MAX, n_beta)  # (n_beta,)
    u     = np.linspace(-1.0, 1.0, n_u)         # (n_u,)
    # Total dimensionless 2D projected column Sigma(beta) via adaptive quadrature for each impact parameter
    sigma = np.array([kernel_column(b) for b in beta])  # (n_beta,)
    # Broadcasting: (n_beta, 1) and (1, n_u) -> (n_beta, n_u)
    # 3D normalized distance r/H = sqrt(beta^2 + s^2), with chord coordinate s = u * s_max = u * sqrt(1 - beta^2)
    r     = np.sqrt(beta[:, None]**2 + u[None, :]**2 * (1.0 - beta[:, None]**2))  # (n_beta, n_u)
    # Cumulative integral int_{-1}^{u} w(r) du' along line axis (axis=1); initial=0.0 preserves (n_beta, n_u) shape with P(..., -1) = 0
    cum   = cumulative_trapezoid(quartic_spline(r), u, axis=1, initial=0.0)  # (n_beta, n_u)
    # Normalize each row by its total integral at u = +1; slicing [:, -1:] keeps 2D shape (n_beta, 1) for broadcasting -> P(..., 1) = 1
    P     = cum / cum[:, -1:]  # (n_beta, n_u)
    return beta, sigma, u, P

if __name__ == "__main__":
    # 3D Kernel Normalisation: int_0^1 4 pi x^2 C w(x) dx must be 1 (H cancels)    
    norm_3d, err_3d = quad(lambda x: 4.0 * np.pi * x**2 * KERNEL_CONSTANT * quartic_spline(x), 0.0, 1.0,
                           points=[0.2, 0.6]) # points identifies the sub-intervals where fourth derivative exhibits a jump, to make adaptive integration more precise

    print(f"3D scipy quad normalisation     : {norm_3d:.10f}  (error estimate {err_3d:.1e})")
    print(f"w(0), w(0.6), w(1), w(1.5)      : {quartic_spline([0.0, 0.6, 1.0, 1.5])}")

    # 2D Kernel Normalisation: 2 pi int_0^1 Sigma(beta) beta dbeta must be 1 (mass conservation along sightlines)
    norm_2d, err_2d = quad(lambda b: 2.0 * np.pi * b * kernel_column(b), 0.0, 1.0, points=[0.2, 0.6])
  
    print(f"2D scipy quad normalisation     : {norm_2d:.10f}  (error estimate {err_2d:.1e})")
    print(f"Sigma(0) vs 15 / (2 pi)         : {kernel_column(0.0):.10f}  {15.0 / (2.0 * np.pi):.10f}")

    # Interpolation error estimate of the lookup table for beta, total column Sigma(beta), line coordinate u, P
    beta, sigma, u, P = column_tables()      # point of the lookup table
    rng   = np.random.default_rng(42)        # reproducibility
    b_t   = rng.uniform(0.0, BETA_MAX, 300)  # impact parameter random grid
    u_t   = rng.uniform(-1.0, 1.0, 300)      # dimensionless line coordinate random grid

    P_linear = RegularGridInterpolator((beta, u), P, method='linear')(np.column_stack([b_t, u_t])) # linear to be faster with 2d interp
    sigma_linear = np.interp(b_t, beta, sigma) # only linear

    # Scipy quad integration of the cumulative
    def P_exact(b, x):
        f = lambda v: float(quartic_spline(np.sqrt(b**2 +v**2 * (1.0 - b**2))))
        return quad(f, -1.0, x)[0] / quad(f, -1.0, 1.0)[0]

    # Maximum discrepancy 
    err_P = np.max(np.abs(P_linear - [P_exact(b, x) for b, x in zip(b_t, u_t)]))
    err_sigma = np.max(np.abs(sigma_linear - [kernel_column(b) for b in b_t])) / sigma[0]
    print(f"Table errors: max |dP| = {err_P:.1e},  max |dSigma| / Sigma(0) = {err_sigma:.1e}")



# srun -p cosma-analyse -A do019 -t 00:10:00 --pty /bin/bash

# source /cosma/local/anaconda3/202512/etc/profile.d/conda.sh && conda activate jagger

# "$CONDA_PREFIX/bin/python" /cosma8/data/do019/dc-fulg1/analysis/scripts/sph_kernel.py
# # *Global imports*
# import os
# nthreads = 8
# os.environ["OMP_NUM_THREADS"] = str(nthreads) 
# os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads) 
# os.environ["MKL_NUM_THREADS"] = str(nthreads)


import numpy as np
import scipy as sp
from numpy.lib.scimath import sqrt
from scipy.io import loadmat
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# *Local imports*
from beams_spec.beams_spec_core import *
from beams_spec.beams_spec_visualisation import *

# TODO: timing data for various steps
# TODO: experimentation integration in time.
# TODO: investigation of instabilities - integrals.
# TODO: 
#Nondimensional phyisical parameters
L = 50.0
g = 2.5

adim_params = NondimensionalBeamParameters(L=L, g=g)

#Spatial and frequencygrids.
delta_s = 0.25
s_grid = np.r_[0.0 : L + delta_s : delta_s]

w_cutoff = sqrt(g) #cutoff frequency
w_max = 2.0*sqrt(2)*w_cutoff
delta_w = ((np.pi/L)**2)/100.0
w = np.r_[0.0 : w_max : delta_w]
nw = np.size(w)

w = w.reshape((nw,1))

#tolerance for frequency and wavenumber computation
tolerance_freq = 1.0e-12

#tolerance for modal computations and solutions
tolerance_func = 1.0e-3

#define structures and compute modal wavenumbers
adim_params = NondimensionalBeamParameters(L=L, g=g)

basis_params = BasisParameters(w_max, w_cutoff, delta_w, tolerance_freq, adim_params)

modal_norms = compute_normalisation_factors(basis_params, tolerance_func)

solution_params = SolutionParameters(basis_params, modal_norms, tolerance_func)

#define temporal grid
w_basis = solution_params.basis_params.w
t_max = (2.0*np.pi)/np.min(w_basis)
delta_t = (2.0*np.pi)/w_basis[9, 0]
t_grid = np.r_[0.0 : t_max : delta_t]

#compute numerical solutions
e1_numeric = compute_discrete_e1(s_grid, t_grid, solution_params)
k2_numeric = compute_discrete_k2(s_grid, t_grid, solution_params)

phi1_numeric, phi3_numeric = compute_phi_from_e1_k2(s_grid, t_grid,
                                                    e1_numeric, k2_numeric,
                                                    solution_params)

#theta_init_numeric = theta_s0(s_grid, adim_params)

cos_theta, sin_theta = compute_theta_from_k(s_grid, t_grid, k2_numeric)

theta_exact = compute_discrete_theta(s_grid, t_grid, solution_params)

cos_theta_exact = np.cos(theta_exact)
sin_theta_exact = np.sin(theta_exact)

e_cos = norm(cos_theta - cos_theta_exact)/norm(cos_theta_exact)
e_sin = norm(sin_theta - sin_theta_exact)/norm(sin_theta_exact)

print(f'Relative error cos(theta) between exact and approx = {e_cos:1.5e}')
print(f'Relative error sin(theta) between exact and approx = {e_sin:1.5e}')

#Visualise the solution
# simulate_beam_exact_theta(phi1_numeric, phi3_numeric, s_grid, t_grid, solution_params, save=True)
# simulate_beam_approx_theta(phi1_numeric, phi3_numeric, cos_theta, sin_theta, s_grid, t_grid, save=True)

# e1 et k2 au cours du temps. a un point pres de 3*L/4

# inspection_point = (3.0*L)/4.0

# dists = np.abs(s_grid - inspection_point)
# inspection_point_id = np.argmin(dists)

# print(f'Grid inspection point:{inspection_point_id}')

# fig_inspection, axes = plt.subplots(2, 1)


# t_plot_grid = np.linspace(0.0, t_max, 10000)

# k2_t = np.zeros(np.shape(t_plot_grid))
# e1_t = np.zeros(np.shape(t_plot_grid))

# for t_i in range(np.size(t_plot_grid)):
#     k2_t[t_i] = k2(inspection_point, t_plot_grid[t_i], solution_params)
#     e1_t[t_i] = e1(inspection_point, t_plot_grid[t_i], solution_params)
    


# axes[0].plot(t_plot_grid, k2_t)
# axes[1].plot(t_plot_grid, e1_t)
# fig_inspection.tight_layout()
# plt.show()

# s_ = rc * s et t_ = tc * t
# rc = r/sqrt(2)
# L_ = rc * L
# tc = rc * sqrt(rho/E) ; rho ~ 4000 ; E ~ 200 * 10^9
# r = 5 * 10^(-2)
# L_ = 15 m
# L = L_/(r / sqrt(2))
# A =  pi * r^2
# I3 = 2 (pi * r^4)







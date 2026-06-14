# *Global imports*
import os
nthreads = 8
os.environ["OMP_NUM_THREADS"] = str(nthreads) 
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads) 
os.environ["MKL_NUM_THREADS"] = str(nthreads)


import numpy as np
import scipy as sp
from numpy.lib.scimath import sqrt
from scipy.io import loadmat
from numpy.linalg import norm

# *Local imports*
from beams_spec.beams_spec_core import *
from beams_spec.beams_spec_visualisation import *


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

#Visualise the solution
#simulate_beam(phi1_numeric, phi3_numeric, s_grid, t_grid, solution_params)
simulate_beam_approx_theta(phi1_numeric, phi3_numeric, cos_theta, sin_theta, s_grid, t_grid)





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

#set parameters from the reference file

L = 50.0 #nondimensional length of beam
g = 2.5  #E/G

prms = NondimensionalBeamParameters(L=L, g=g)

#spatial and frequency space grid definition

delta_s = 0.25
s = np.r_[0.0 : L + delta_s : delta_s]
ns = np.size(s)

s = s.reshape((ns,1)) #all arrays should have two dimensions

w_cutoff = sqrt(g) #cutoff frequency
w_max = 2.0*sqrt(2)*w_cutoff
delta_w = ((np.pi/L)**2)/100.0
w = np.r_[0.0 : w_max : delta_w]
nw = np.size(w)

w = w.reshape((nw,1))

tolerance_freq = 1.0e-12
tolerance_func = 1.0e-3

adim_phys_params = NondimensionalBeamParameters(L=L, g=g)

basis_params = BasisParameters(w_max, w_cutoff, delta_w, tolerance_freq, adim_phys_params)

norms_modal = compute_normalisation_factors(basis_params, tolerance_func)

solution_params = SolutionParameters(basis_params, norms_modal, tolerance_func)
print(f'Finished basis setup.')

#phi1, phi3, t = time_integration_phi(s, solution_params)
phi1, phi3, t = time_integration_phi2(s, solution_params)
print(f'Finished time integration.')

simulate_beam_exact_theta(phi1, phi3, s, t, solution_params)
print(f'The end.')

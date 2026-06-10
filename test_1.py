
# *Global imports*
import numpy as np
import scipy as sp
from numpy.lib.scimath import sqrt
from scipy.io import loadmat
from numpy.linalg import norm

# *Local imports*
from beams_spec.BeamProblem import TimoshenkoAdimParams
from beams_spec.InitCondtions import *
from beams_spec.UtilityFunctions import Ef_params, calculate_coeffs_eigenfuncs, compute_normalisation_factors, determine_eigenfrequencies, dispersion_relation, test_bs_bc
from beams_spec.UtilityFunctions import f_function
from beams_spec.Visualisation import  visual_check_f_functions, visually_check_initial_conditions, visually_inspect_efuncs

#set parameters from the reference file

L = 50.0 #nondimensional length of beam
g = 2.5  #E/G

prms = TimoshenkoAdimParams(L=L, g=g)

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

k_1,k_2 = dispersion_relation(w, prms)

k1_test_file = loadmat('test_rd_1_1.mat')
k2_test_file = loadmat('test_rd_2_1.mat')

k1_test = k1_test_file['t1']
k2_test = k2_test_file['t2']

print(f'Test value norm dispersion_relation:{norm(k_1-k1_test) + norm(k_2-k2_test):1.5e}')

f_function_test_file = loadmat('f_out_test.mat')
f_out_test = f_function_test_file['f_out']

f_test = f_function(w, prms)

print(f'Test value norm f_function:{norm(f_test - f_out_test):1.5e}')

#visual_check_f_functions(f_test, f_out_test, w)

#Low frequency domain
w_lf = np.r_[delta_w : w_cutoff - delta_w : delta_w]
#High frequency domain
w_hf = np.r_[w_cutoff + delta_w : w_max : delta_w]

w_lf = w_lf.reshape((np.size(w_lf), 1))
w_hf = w_hf.reshape((np.size(w_hf), 1))

tolerance_ef = 1.0e-12

wn_lf, kpn_lf, kmn_lf = determine_eigenfrequencies(w_lf, tolerance_ef, prms)
wn_hf, kpn_hf, kmn_hf = determine_eigenfrequencies(w_hf, tolerance_ef, prms)

wn_lf_test_file = loadmat('./tests/wn_BF_test.mat')
kpn_lf_test_file = loadmat('./tests/kpn_BF_test.mat')
kmn_lf_test_file = loadmat('./tests/kmn_BF_test.mat')

wn_hf_test_file = loadmat('./tests/wn_HF_test.mat')
kpn_hf_test_file = loadmat('./tests/kpn_HF_test.mat')
kmn_hf_test_file = loadmat('./tests/kmn_HF_test.mat')

wn_lf_test = wn_lf_test_file['wn_BF'].reshape(np.shape(wn_lf))
kpn_lf_test = kpn_lf_test_file['kpn_BF'].reshape(np.shape(wn_lf))
kmn_lf_test = kmn_lf_test_file['kmn_BF'].reshape(np.shape(wn_lf))

wn_hf_test = wn_hf_test_file['wn_HF'].reshape(np.shape(wn_hf))
kpn_hf_test = kpn_hf_test_file['kpn_HF'].reshape(np.shape(wn_hf))
kmn_hf_test = kmn_hf_test_file['kmn_HF'].reshape(np.shape(wn_hf))

print(f'Test value diff in LF regime wn: {norm(wn_lf_test - wn_lf):1.5e}')
print(f'Test value diff in LF regime kpn: {norm(kpn_lf_test - kpn_lf):1.5e}')
print(f'Test value diff in LF regime kmn: {norm(kmn_lf_test - kmn_lf):1.5e}')
print(f'Test value diff in HF regime wn: {norm(wn_hf_test - wn_hf):1.5e}')
print(f'Test value diff in HF regime kpn: {norm(kpn_hf_test - kpn_hf):1.5e}')
print(f'Test value diff in HF regime kmn: {norm(kmn_hf_test - kmn_hf):1.5e}')

#combine eigenfrequencies

tolerance_efunc = 1e-3

wn = np.block([[wn_lf], [wn_hf]])
kpn = np.block([[kpn_lf], [kpn_hf]])
kmn = np.block([[kmn_lf], [kmn_hf]])

assert np.shape(wn) == np.shape(kpn) == np.shape(kmn)

apc, aps, amc, ams = calculate_coeffs_eigenfuncs(wn, kpn, kmn, prms)

coeffs_test_file = loadmat('./coeffs_test.mat')
apc_test = coeffs_test_file['apc']
aps_test = coeffs_test_file['aps']
amc_test = coeffs_test_file['amc']
ams_test = coeffs_test_file['ams']

assert np.shape(apc) == np.shape(aps_test) == np.shape(aps) == np.shape(aps_test) == np.shape(amc) == np.shape(ams) == np.shape(ams_test)

print(f'Test apc norm:{norm(apc - apc_test):1.5e}')
print(f'Test aps norm:{norm(aps - aps_test):1.5e}')
print(f'Test amc norm:{norm(amc - amc_test):1.5e}')
print(f'Test ams norm:{norm(ams - ams_test):1.5e}')

efparams = Ef_params(prms, apc, aps, amc, ams, wn, kpn, kmn)

norms_ef = compute_normalisation_factors(efparams, tolerance_efunc)

#visually_inspect_efuncs(efparams, norms_ef, s, w_cutoff)

test_bs_bc(efparams, norms_ef, tolerance_efunc)

visually_check_initial_conditions(s, efparams, norms_ef, tolerance_efunc)

#time_simulation

# #!! the outputs are lists not numpy objects
# v1, e1, o2, k2 = eigenfunctions(wn, kpn, kmn, tolerance_efunc, prms)

# #visual check eigenfunctions
# visual_check_eigenfunctions(v1, e1, o2, k2, wn, w_cutoff, s, 5)

# #projection of boundary conditions


# assert np.shape(v1) == np.shape(e1) == np.shape(o2) == np.shape(k2)

#v1_st, e1_st, o2_st, k2_st, theta_st = solution_st(v1, e1, o2, k2, wn, tolerance_efunc, prms)

#TODO: There is a problem here with k2_st - Date: 26/05/2026
#visual_check_init_conditions(v1_st, e1_st, o2_st, k2_st, s, prms)













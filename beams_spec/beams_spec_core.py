import numpy as np
import scipy as sp

from scipy.io import loadmat
from numpy.linalg import norm

#This function returns complex values a-priori
from numpy.lib.scimath import sqrt

from scipy.integrate import solve_ivp
from .beams_spec_structures import *
from .beams_spec_init_conditions import *


class SolutionParameters:
    def __init__(self, basis_params:BasisParameters, norms_ef, tol):
        self.basis_params = basis_params
        self.norms_ef = norms_ef
        self.tol = tol
        self.bs, self.bc = project_boundary_conditions(basis_params, norms_ef, tol)



def scalar_product(x1, x2, x3, x4, y1, y2, y3, y4, tol:float, params:NondimensionalBeamParameters):
    """Scalar product defined by the separation of variables"""

    L = params.L
    g = params.g

    #global g L wc
    # f =@(x) x1(x).*conj(y1(x)) ...
    # + g*x2(x).*conj(y2(x)) ...
    # +   x3(x).*conj(y3(x)) ...
    # +   x4(x).*conj(y4(x));
    # xy =integral(f,0,L,'AbsTol',tolerance,'RelTol',tolerance,'ArrayValued',true);
    f = lambda x: (x1(x)*np.conjugate(y1(x))
                   + g*x2(x)*np.conjugate(y2(x))
                   + x3(x)*np.conjugate(y3(x))
                   + x4(x)*np.conjugate(y4(x)))
    f_real = lambda x: np.real(f(x))
    f_imag = lambda x: np.imag(f(x))


    dp_real, err = sp.integrate.quad(f_real, 0.0, L, epsabs=0, epsrel=tol)
    dp_imag, err = sp.integrate.quad(f_imag, 0.0, L, epsabs=0, epsrel=tol)
    return dp_real + (dp_imag * 1.0j)








def v1_mode_i_unnormalised(x, i, efparams:BasisParameters):
    g = efparams.adim_params.g
    km = efparams.km
    kp = efparams.kp
    apc = efparams.apc
    aps = efparams.aps
    amc = efparams.amc
    ams = efparams.ams
    
    return -g*( km[i,0]*(ams[i,0]*np.cos(km[i,0]*x) - amc[i,0]*np.sin(km[i,0]*x)) + kp[i,0]*(aps[i,0]*np.cos(kp[i,0]*x) - apc[i, 0]*np.sin(kp[i,0]*x)) )

def e1_mode_i_unnormalised(x, i, efparams:BasisParameters):
    g = efparams.adim_params.g
    km = efparams.km
    kp = efparams.kp
    apc = efparams.apc
    aps = efparams.aps
    amc = efparams.amc
    ams = efparams.ams
    w = efparams.w

    return w[i,0]*(amc[i,0]*np.cos(km[i,0]*x) + apc[i,0]*np.cos(kp[i,0]*x) + ams[i,0]*np.sin(km[i,0]*x) + aps[i,0]*np.sin(kp[i,0]*x))

def o2_mode_i_unnormalised(x, i, efparams:BasisParameters):
    g = efparams.adim_params.g
    km = efparams.km
    kp = efparams.kp
    apc = efparams.apc
    aps = efparams.aps
    amc = efparams.amc
    ams = efparams.ams
    w = efparams.w
    W = w**2
    Kp = kp**2
    Km = km**2

    return (g*Km[i,0]-W[i,0])*(amc[i,0]*np.cos(km[i,0]*x) + ams[i,0]*np.sin(km[i,0]*x)) + (g*Kp[i,0]-W[i,0])*(apc[i,0]*np.cos(kp[i,0]*x) + aps[i,0]*np.sin(kp[i,0]*x))

def k2_mode_i_unnormalised(x, i, efparams:BasisParameters):
    g = efparams.adim_params.g
    km = efparams.km
    kp = efparams.kp
    apc = efparams.apc
    aps = efparams.aps
    amc = efparams.amc
    ams = efparams.ams
    w = efparams.w
    W = w**2
    Kp = kp**2
    Km = km**2


    return ( (g*Km[i,0]-W[i,0])*km[i,0]*(ams[i,0]*np.cos(km[i,0]*x) - amc[i,0]*np.sin(km[i,0]*x)) + (g*Kp[i,0]-W[i,0])*kp[i,0]*(aps[i,0]*np.cos(kp[i,0]*x) - apc[i,0]*np.sin(kp[i,0]*x)) )/w[i,0]

def compute_ef_params(w, kp, km, params:NondimensionalBeamParameters):
    g = params.g
    L = params.L

    W = w**2
    Kp = kp**2
    Km = km**2

    # apc = (g*Km-W).*(-kp.*sin(km*L)+km.*sin(kp*L));
    # aps = km.*(+(g*Kp-W).*cos(km*L) - (g*Km-W).*cos(kp*L));
    # amc = (g*Kp-W).*(+kp.*sin(km*L)-km.*sin(kp*L));
    # ams = kp.*(-(g*Kp-W).*cos(km*L) + (g*Km-W).*cos(kp*L));

    #Compute modal amplitutes
    apc = (g*Km - W)*(-kp*np.sin(km*L) + km*np.sin(kp*L));
    aps = km*(+(g*Kp - W)*np.cos(km*L) - (g*Km - W)*np.cos(kp*L));
    amc = (g*Kp - W)*(+kp*np.sin(km*L)-km*np.sin(kp*L));
    ams = kp*(-(g*Kp - W)*np.cos(km*L) + (g*Km - W)*np.cos(kp*L));

    assert np.shape(apc) == np.shape(w)
    assert np.shape(aps) == np.shape(w)
    assert np.shape(amc) == np.shape(w)
    assert np.shape(ams) == np.shape(w)

    return BasisParameters(params, apc, aps, amc, ams, w, kp, km)

def compute_normalisation_factors(efparams:BasisParameters, tol:float):

    Tparams = efparams.adim_params
    n_shape = np.shape(efparams.w)
    n_size = np.size(efparams.w)
    norms_ef = np.zeros(n_shape, dtype=complex)

    for i in range(n_size):
        v1 = lambda x, i=i: v1_mode_i_unnormalised(x, i, efparams)
        e1 = lambda x, i=i: e1_mode_i_unnormalised(x, i, efparams)
        o2 = lambda x, i=i: o2_mode_i_unnormalised(x, i, efparams)
        k2 = lambda x, i=i: k2_mode_i_unnormalised(x, i, efparams)
        norms_ef[i,0] = sqrt(scalar_product(v1, e1, o2, k2, v1, e1, o2, k2, tol, Tparams))
    return norms_ef
        
        
def v1_mode_i(x, i, efparams:BasisParameters, norms_ef):
    return v1_mode_i_unnormalised(x, i, efparams)/norms_ef[i, 0]
def e1_mode_i(x, i, efparams:BasisParameters, norms_ef):
    return e1_mode_i_unnormalised(x, i, efparams)/norms_ef[i, 0]
def o2_mode_i(x, i, efparams:BasisParameters, norms_ef):
    return o2_mode_i_unnormalised(x, i, efparams)/norms_ef[i, 0]
def k2_mode_i(x, i, efparams:BasisParameters, norms_ef):
    return k2_mode_i_unnormalised(x, i, efparams)/norms_ef[i, 0]
    

def project_boundary_conditions(efparams:BasisParameters, norms_ef, tol):
    """returns the arrays bc and bs from the matlab code"""
    params = efparams.adim_params
    w = efparams.w
    nn = np.size(w)
    bc = np.zeros((nn, 1), dtype=complex)
    bs = np.zeros((nn, 1), dtype=complex)

    zf = lambda x: 0.0*x
    v11_s0 = lambda x: v1_init(x, params)
    e11_s0 = lambda x: e1_init(x, params)
    o22_s0 = lambda x: o2_init(x, params)
    k22_s0 = lambda x: k2_init(x, params)

    for i in range(nn):
        v1_efi = lambda x, i=i: v1_mode_i(x, i, efparams, norms_ef)
        e1_efi = lambda x, i=i: e1_mode_i(x, i, efparams, norms_ef)
        o2_efi = lambda x, i=i: o2_mode_i(x, i, efparams, norms_ef)
        k2_efi = lambda x, i=i: k2_mode_i(x, i, efparams, norms_ef)
        bs[i, 0] = 2.0*scalar_product(v11_s0, e11_s0, o22_s0, k22_s0,
                                      v1_efi, zf, o2_efi, zf, tol, params)
        bc[i, 0] = 2.0*scalar_product(v11_s0, e11_s0, o22_s0, k22_s0,
                                      zf, e1_efi, zf, k2_efi, tol, params)
    #end for

    return (bs, bc)

def test_bs_bc(efparams:BasisParameters, norms_ef, tol):

    bs, bc = project_boundary_conditions(efparams, norms_ef, tol)

    bc_test_file = loadmat('bc_test.mat')
    bs_test_file = loadmat('bs_test.mat')
    bc_test = bc_test_file['bc']
    bs_test = bs_test_file['bs']

    assert np.shape(bs_test) == np.shape(bs)
    assert np.shape(bc_test) == np.shape(bc)

    print(f'Test norm for bc: {norm(bc_test - bc):1.5e}')
    print(f'Test norm for bs: {norm(bs_test - bs):1.5e}')

# v1_st=@(x,t) transpose(v1_brut(x))*(-bc.*sin(wn*t)+bs.*cos(wn*t));
# e1_st=@(x,t) transpose(e1_brut(x))*(+bc.*cos(wn*t)+bs.*sin(wn*t));
# w2_st=@(x,t) transpose(w2_brut(x))*(-bc.*sin(wn*t)+bs.*cos(wn*t));
# k2_st=@(x,t) transpose(k2_brut(x))*(+bc.*cos(wn*t)+bs.*sin(wn*t));
# th_st=@(x,t) transpose(w2_brut(x))*((bc.*cos(wn*t)+bs.*sin(wn*t))./wn);


def v1(s, t, sparams:SolutionParameters):
    """Solution for v1 in moving frame"""
    w = sparams.basis_params.w
    efparams = sparams.basis_params
    norms_ef = sparams.norms_ef
    nn = np.size(w)
    bs = sparams.bs
    bc = sparams.bc

    v1st = 0.0
    for i in range(nn):
        v1st = v1st + (v1_mode_i(s, i, efparams, norms_ef) *
                       (-bc[i, 0]*np.sin(w[i, 0]*t) + bs[i, 0]*np.cos(w[i, 0]*t)))
    
    return v1st

def e1(s, t, sparams:SolutionParameters):
    """Solution for v1 in moving frame"""
    w = sparams.basis_params.w
    efparams = sparams.basis_params
    norms_ef = sparams.norms_ef
    bs = sparams.bs
    bc = sparams.bc
    nn = np.size(w)
   
    e1st = 0.0
    for i in range(nn):
        e1st = e1st + (e1_mode_i(s, i, efparams, norms_ef) *
                       (+bc[i, 0]*np.cos(w[i, 0]*t) + bs[i, 0]*np.sin(w[i, 0]*t)))
    
    return e1st

def o2(s, t, sparams:SolutionParameters):
    """Solution for v1 in moving frame"""

    w = sparams.basis_params.w
    efparams = sparams.basis_params
    norms_ef = sparams.norms_ef
    bs = sparams.bs
    bc = sparams.bc
    nn = np.size(w)

    o2st = 0.0
    for i in range(nn):
        o2st = o2st + (o2_mode_i(s, i, efparams, norms_ef) *
                       (-bc[i, 0]*np.sin(w[i, 0]*t) + bs[i, 0]*np.cos(w[i, 0]*t)))
    
    return o2st

def k2(s, t, sparams:SolutionParameters):
    """Solution for v1 in moving frame"""
    w = sparams.basis_params.w
    efparams = sparams.basis_params
    norms_ef = sparams.norms_ef
    bs = sparams.bs
    bc = sparams.bc
    nn = np.size(w)

    k2st = 0.0
    for i in range(nn):
        k2st = k2st + (k2_mode_i(s, i, efparams, norms_ef) *
                       (+bc[i, 0]*np.cos(w[i, 0]*t) + bs[i, 0]*np.sin(w[i, 0]*t)))
    
    return k2st

def theta(s, t, sparams:SolutionParameters):
    """Solution for v1 in moving frame"""

    w = sparams.basis_params.w
    efparams = sparams.basis_params
    norms_ef = sparams.norms_ef
    bs = sparams.bs
    bc = sparams.bc
    nn = np.size(w)

    thetast = 0.0
    for i in range(nn):
        thetast = thetast + ((o2_mode_i(s, i, efparams, norms_ef) *
                              (+bc[i, 0]*np.cos(w[i, 0]*t) + bs[i, 0]*np.sin(w[i, 0]*t)))/w[i,0])
    
    return thetast




# function dydx = ODE_ph_X_dis(x,y,e1_s_dis,k2_s_dis,s_dis)
# %y(1)=>phi_1(s,t(it))       pour t(it) fixé
# %y(2)=>phi_3(s,t(it))-s     pour t(it) fixé
# e1=interp1(s_dis,e1_s_dis,x);
# k2=interp1(s_dis,k2_s_dis,x);
# dydx = zeros(2,1);
# dydx(1) = e1-k2.*(x+y(2));
# dydx(2) = k2.*y(1);

def ode_phi_s(s, phi, e1_numeric_s, k2_numeric_s, s_grid):
    e1 = np.interp(s, s_grid, e1_numeric_s)
    k2 = np.interp(s, s_grid, k2_numeric_s)
    phi_prime = np.zeros((2,), dtype=complex)
    phi_prime[0] = e1 - (k2 * (s + phi[1]))
    phi_prime[1] = k2*phi[1]

    return phi_prime


def time_integration_phi(s_grid, sparams:SolutionParameters):

    w = sparams.basis_params.w
    L = sparams.basis_params.adim_params.L
    tmax = (2.0*np.pi)/np.min(w)
    dt = 2.0*np.pi/w[9, 0]
    t = np.r_[0.0 : tmax : dt]
    nt = np.size(t)
    ns = np.size(s_grid)
    xspan = (0, L)
    k2_numeric = np.zeros((ns, nt), dtype=complex)
    k2_numeric = np.zeros((ns, nt), dtype=complex)
    s_grid = s_grid.reshape((ns,))
    tol = sparams.tol
    phi1_numeric = np.zeros((ns, nt), dtype=complex)
    phi3_numeric = np.zeros((ns, nt), dtype=complex)

    #print(f'Value of nt:{nt}')
    #print(f'Value of ns:{ns}')
    #Discretized values of k2 and e1 on s-t grid
    for i in range(ns):
        for j in range(nt):
            k2_numeric[i, j] = e1(s_grid[i], t[j], sparams)
            k2_numeric[i, j] = k2(s_grid[i], t[j], sparams)
        #endfor
    #endfor

    #Time integration loop
    for it in range(nt):
        F = lambda s, phi, it=it: ode_phi_s(s, phi, k2_numeric[:, it].reshape((ns,)),
                                             k2_numeric[:, it].reshape((ns,)), s_grid)
        phi_init_s = np.array([0.0, 0.0])
        sol = solve_ivp(F, xspan, phi_init_s, t_eval=s_grid, atol=tol, rtol=tol)
        phi_numeric = sol.y
     
        #print(f'Shape of y:{y.shape}')

        phi1_numeric[:, it] = phi_numeric[0, :]
        phi3_numeric[:, it] = s_grid + phi_numeric[1, :]


    return (phi1_numeric, phi3_numeric, t)



        
        


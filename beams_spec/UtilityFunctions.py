import numpy as np
import scipy as sp

from scipy.io import loadmat
from numpy.linalg import norm

#This function returns complex values a-priori
from numpy.lib.scimath import sqrt
from scipy.optimize import fminbound
from .BeamProblem import TimoshenkoAdimParams
from .InitCondtions import *

def dispersion_relation(w, params:TimoshenkoAdimParams):
    """Returns a discretized version
    of the dispersion relation on a frequency grid (w is an array)"""

    g = params.g

    w_sq = w * w

    #DEBUG: check that nothing weird has happened due to
    #NumPy's detestable automatic reshaping.
    assert np.shape(w_sq) == np.shape(w)


    #     Kp=( (1+g)*W + sqrt(W).*sqrt((1-2*g).*W + g^2*(4+W)) )/(2*g);
    # Km=( (1+g)*W - sqrt(W).*sqrt((1-2*g).*W + g^2*(4+W)) )/(2*g);

    E = (sqrt(w_sq)*(sqrt((1-2*g)*w_sq + (g**2)*(4+w_sq))))
    kp_sq = ( ((1 + g)*w_sq) + E )/(2*g)
    km_sq = ( ((1 + g)*w_sq) - E )/(2*g)

    #DEBUG: check dimensions
    assert np.shape(w_sq) == np.shape(kp_sq)
    assert np.shape(w_sq) == np.shape(km_sq)

    return (sqrt(kp_sq), sqrt(km_sq))



def f_function(w, params:TimoshenkoAdimParams):
    """Corresponds to FonctionF in the matlab version"""

    L = params.L
    g = params.g

    kp, km = dispersion_relation(w, params)

    Kp = kp**2
    Km = km**2
    W = w**2

    #Frequency equations for bcs clamped -- free i.e cantilever
    #F=(g*km.*kp.*(g*(Km.^2+Kp.^2)-2*W.^2).*cos(km*L).*cos(kp*L) +...
    #(g*Km-W).*(g*Kp-W).*(-2*km.*kp+(1+1/g)*W.*sin(km*L).*sin(kp*L)));

    f = (g * km * kp * (g * (Km**2 + Kp**2)-(2.0*W**2))*np.cos(km * L) * np.cos(kp * L) +
                        (g*Km - W)*(g*Kp - W)*((-2.0 * km * kp) +
                                               (1.0 + (1.0/g)) * W * np.sin(km * L) *
                                               np.sin(kp*L)))


    assert np.shape(w) == np.shape(f)
    
    return f



def determine_eigenfrequencies(w, tolerance:float, params:TimoshenkoAdimParams):
    """returns eigenfrequencies"""
    #L = params.L
    #g = params.g
    delta_w = w[1, 0] - w[0, 0]
    #print('I am here!!')
    f_abs = lambda w : np.abs(f_function(w, params))**2

    #TODO: Carefully verify the following there is likely a
    #difference in behaviour between this and matlab counterparts
    ddf = np.diff(np.sign(np.diff(f_abs(w).flatten())))

    ddf = np.concat((ddf, np.array([0, 0])))

    w_n = w.flatten()[ddf==2]
    n_n = np.size(w_n)
    #print(f'this is n:{n}')
    w_n = w_n.reshape((np.size(w_n), 1))

    idx_w = 0
    w_n_refined = np.zeros(np.size(w_n))

    for idx_n in range(n_n):
        #print('Debug: I am here')
        x1 = np.max([np.min(w), w_n[idx_n, 0] - (2.0 * delta_w)])
        x2 = np.min([w_n[idx_n, 0] + (2.0 * delta_w), np.max(w)])

        f_opt_func = lambda w : f_abs(w) / f_abs(w_n[idx_n])

        x, f_val, ierr, num_eval = fminbound(f_opt_func, x1, x2, xtol=tolerance, full_output=True)

        #print(f'Debug: This is x:{x}')

        if ierr == 0:
            w_n_refined[idx_w] = x[0]
            idx_w = idx_w + 1



    #Now eliminate extraneous values in w_n_refined
    w_n_refined = w_n_refined.flatten()
    mask = np.ones(np.size(w_n_refined), dtype=bool)
    mask[idx_w:] = False
    w_n_refined = w_n_refined[mask]
    w_n_refined = w_n_refined.reshape((np.size(w_n_refined), 1))

    kp_n, km_n = dispersion_relation(w_n_refined, params)
    
    return (w_n_refined, kp_n, km_n)


def scalar_product(x1, x2, x3, x4, y1, y2, y3, y4, tol:float, params:TimoshenkoAdimParams):
    """Scalar product defined by the separation of variables"""

    L = params.L
    g = params.g

    #global g L wc
    # f =@(x) x1(x).*conj(y1(x)) ...
    # + g*x2(x).*conj(y2(x)) ...
    # +   x3(x).*conj(y3(x)) ...
    # +   x4(x).*conj(y4(x));
    # xy =integral(f,0,L,'AbsTol',tolerance,'RelTol',tolerance,'ArrayValued',true);
    f = lambda x: np.real(x1(x)*np.conjugate(y1(x))
                   + g*x2(x)*np.conjugate(y2(x))
                   + x3(x)*np.conjugate(y3(x))
                   + x4(x)*np.conjugate(y4(x)))


    dp, err = sp.integrate.quad(f, 0.0, L, epsabs=0, epsrel=tol)
    return dp


def eigenfunctions(w, kp, km, tol:float, params:TimoshenkoAdimParams):
    """return python lists (not numpy arrays!!) of eigenfunctions"""

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

    # v1_brut =@(x) -g*( km.*(ams.*cos(km*x) - amc.*sin(km*x)) + kp.*(aps.*cos(kp*x) - apc.*sin(kp*x)) );
    # e1_brut =@(x) w.*(amc.*cos(km*x) + apc.*cos(kp*x) + ams.*sin(km*x) + aps.*sin(kp*x));
    # w2_brut =@(x) (g*Km-W).*(amc.*cos(km*x) + ams.*sin(km*x)) + (g*Kp-W).*(apc.*cos(kp*x) + aps.*sin(kp*x));
    # k2_brut =@(x) ( (g*Km-W).*km.*(ams.*cos(km*x) - amc.*sin(km*x)) + (g*Kp-W).*kp.*(aps.*cos(kp*x) - apc.*sin(kp*x)) )./w;

    n = np.size(w)
    v1 = []
    e1 = []
    o2 = []
    k2 = []

    for i in range(n):
        v1_efi = lambda x: -g*( km[i,0]*(ams[i,0]*np.cos(km[i,0]*x) - amc[i,0]*np.sin(km[i,0]*x)) + kp[i,0]*(aps[i,0]*np.cos(kp[i,0]*x) - apc[i, 0]*np.sin(kp[i,0]*x)) )
        
        e1_efi = lambda x: w[i,0]*(amc[i,0]*np.cos(km[i,0]*x) + apc[i,0]*np.cos(kp[i,0]*x) + ams[i,0]*np.sin(km[i,0]*x) + aps[i,0]*np.sin(kp[i,0]*x))
        
        o2_efi = lambda x: (g*Km[i,0]-W[i,0])*(amc[i,0]*np.cos(km[i,0]*x) + ams[i,0]*np.sin(km[i,0]*x)) + (g*Kp[i,0]-W[i,0])*(apc[i,0]*np.cos(kp[i,0]*x) + aps[i,0]*np.sin(kp[i,0]*x))
        k2_efi = lambda x: ( (g*Km[i,0]-W[i,0])*km[i,0]*(ams[i,0]*np.cos(km[i,0]*x) - amc[i,0]*np.sin(km[i,0]*x)) + (g*Kp[i,0]-W[i,0])*kp[i,0]*(aps[i,0]*np.cos(kp[i,0]*x) - apc[i,0]*np.sin(kp[i,0]*x)) )/w[i,0]

        v1.append(v1_efi)
        e1.append(e1_efi)
        o2.append(o2_efi)
        k2.append(k2_efi)

    #end for
    # v1_ = []
    # e1_ = []
    # o2_ = []
    # k2_ = []
    v1_ = np.empty((n, 1), dtype='object')
    e1_ = np.empty((n, 1), dtype='object')
    o2_ = np.empty((n, 1), dtype='object')
    k2_ = np.empty((n, 1), dtype='object')
    for i in range(n):
        prod = scalar_product(v1[i], e1[i], o2[i], k2[i], v1[i], e1[i], o2[i], k2[i], tol, params)
        norm = np.sqrt(prod)
        #print(f'norm: {norm:1.5e}')
        v1_[i, 0] = lambda x: v1[i](x)/norm
        e1_[i, 0] = lambda x: e1[i](x)/norm
        o2_[i, 0] = lambda x: o2[i](x)/norm
        k2_[i, 0] = lambda x: k2[i](x)/norm
    
    return (v1_,e1_,o2_,k2_)


def solution_st(v1_ef, e1_ef, o2_ef, k2_ef, wn, tol, params):
    """return the spatio temporal solutions v1, e1, o2, k2, *and* theta"""
    nn = np.size(wn)

    bc = np.zeros((nn, 1))
    bs = np.zeros((nn, 1))

    zero_func = lambda x: 0.0*x
    v11_s0 = lambda x: v1_s0(x, params)
    e11_s0 = lambda x: e1_s0(x, params)
    w22_s0 = lambda x: w2_s0(x, params)
    k22_s0 = lambda x: k2_s0(x, params)
    

    for i in range(nn):
        bs[i, 0] = 2*scalar_product(v11_s0, e11_s0, w22_s0,
                                    k22_s0, v1_ef[i,0], zero_func,
                                    o2_ef[i,0], zero_func, tol, params)
        
        bc[i, 0] = 2*scalar_product(v11_s0, e11_s0, w22_s0,
                                    k22_s0,  zero_func, e1_ef[i,0],
                                    zero_func, k2_ef[i,0], tol, params)

    #DEBUG test directly with values obtained from matlab implementation
    bc_test_file = loadmat('bc_test.mat')
    bs_test_file = loadmat('bs_test.mat')
    bc_test = bc_test_file['bc']
    bs_test = bs_test_file['bs']

    assert np.shape(bs_test) == np.shape(bs)
    assert np.shape(bc_test) == np.shape(bc)
    

    print(f'Test norm for bc: {norm(bc_test - bc):1.5e}')
    print(f'Test norm for bs: {norm(bs_test - bs):1.5e}')
    # DEBUG note: Date 26/05/2026: There is a problem with bc. 
    



    # v1_st=@(x,t) transpose(v1_brut(x))*(-bc.*sin(wn*t)+bs.*cos(wn*t));
    # e1_st=@(x,t) transpose(e1_brut(x))*(+bc.*cos(wn*t)+bs.*sin(wn*t));
    # w2_st=@(x,t) transpose(w2_brut(x))*(-bc.*sin(wn*t)+bs.*cos(wn*t));
    # k2_st=@(x,t) transpose(k2_brut(x))*(+bc.*cos(wn*t)+bs.*sin(wn*t));
    # th_st=@(x,t) transpose(w2_brut(x))*((bc.*cos(wn*t)+bs.*sin(wn*t))./wn);


    def v1_st(s, t):

        val = 0.0
        for i in range(nn):
            val = val + (v1_ef[i,0](s) * (-bc[i,0] * np.sin(wn[i, 0] * t) +
                                       bs[i,0] * np.cos(wn[i, 0] * t)))
        return val
    
    def e1_st(s, t):

        val = 0.0
        for i in range(nn):
            val = val + (e1_ef[i,0](s) * (bc[i,0] * np.cos(wn[i, 0] * t) +
                                       bs[i,0] * np.sin(wn[i, 0] * t)))

        return val
        
    def o2_st(s, t):

        val = 0.0
        for i in range(nn):
            val = val + (o2_ef[i,0](s) * (-bc[i,0] * np.sin(wn[i, 0] * t) +
                                       bs[i,0] * np.cos(wn[i, 0] * t)))

        return val

    def k2_st(s, t):

        val = 0.0
        for i in range(nn):
            val = val + (k2_ef[i,0](s) * (bc[i,0] * np.cos(wn[i, 0] * t) +
                                       bs[i,0] * np.sin(wn[i, 0] * t)))

        return val
    
    def theta_st(s, t):

        val = 0.0
        for i in range(nn):
            val = val + (o2_ef[i,0](s) * ((bc[i,0] * np.cos(wn[i, 0] * t) +
                                        bs[i,0] * np.sin(wn[i, 0] * t))/wn[i, 0]))

        return val
    

    return (v1_st, e1_st, o2_st, k2_st, theta_st)
    
    
        

    
    
    

    
    


    

    

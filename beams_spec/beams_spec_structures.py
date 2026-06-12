from collections.abc import Callable
import numpy as np
from numpy.lib.scimath import sqrt
from scipy.optimize import fminbound
#from .beams_spec_core import *

class NondimensionalBeamParameters:
    def __init__(self, L:float = 50.0, g:float = 2.5, phi1_00:float = 0.0,
                 phi3_00:float = 0.0, theta_00:float = 0.0):
        self.L = L
        self.g = g
        self.phi1_00 = phi1_00
        self.phi3_00 = phi3_00
        self.theta_00 = theta_00

# class BasisParameters:
#     def __init__(self, adim_params:NondimensionalBeamParameters, apc, aps, amc, ams, w, kp, km):
#         self.adim_params = adim_params


        
#         self.apc = apc
#         self.aps = aps
#         self.amc = amc
#         self.ams = ams
#         self.kp = kp
#         self.km = km
#         self.w = w

class BasisParameters:
    def __init__(self, w_max, w_cutoff, delta_w, tol_frequency, adim_params:NondimensionalBeamParameters):
        self.adim_params = adim_params

        #determine high and low frequency domains
        w_lf, w_hf = low_and_high_frequency_domains(w_max, w_cutoff, delta_w)
        wn_lf, kpn_lf, kmn_lf = compute_frequencies_wavenumbers(w_lf, tol_frequency, self.adim_params)
        wn_hf, kpn_hf, kmn_hf = compute_frequencies_wavenumbers(w_hf, tol_frequency, self.adim_params)
        self.w = np.block([[wn_lf], [wn_hf]])
        self.kp = np.block([[kpn_lf], [kpn_hf]])
        self.km = np.block([[kmn_lf], [kmn_hf]])
        self.apc, self.aps, self.amc, self.ams = compute_modal_amplitutes(self.w, self.kp, self.km, self.adim_params)


        


def low_and_high_frequency_domains(w_max, w_cutoff, delta_w):
    w_lf = np.r_[delta_w : w_cutoff - delta_w : delta_w]
    w_hf = np.r_[w_cutoff + delta_w : w_max : delta_w]
    w_lf = w_lf.reshape((np.size(w_lf), 1))
    w_hf = w_hf.reshape((np.size(w_hf), 1))
    return (w_lf, w_hf)


def dispersion_relation(w, params:NondimensionalBeamParameters):
    """Returns a discretized version
    of the dispersion relation on a frequency grid (w is an array)"""

    g = params.g

    W = w * w

    #DEBUG: check that nothing weird has happened due to
    #NumPy's detestable automatic reshaping.
    assert np.shape(W) == np.shape(w)


    #     Kp=( (1+g)*W + sqrt(W).*sqrt((1-2*g).*W + g^2*(4+W)) )/(2*g);
    # Km=( (1+g)*W - sqrt(W).*sqrt((1-2*g).*W + g^2*(4+W)) )/(2*g);

    E = (sqrt(W)*(sqrt((1-2*g)*W + (g**2)*(4+W))))
    Kp = ( ((1 + g)*W) + E )/(2*g)
    Km = ( ((1 + g)*W) - E )/(2*g)

    #DEBUG: check dimensions
    assert np.shape(W) == np.shape(Kp)
    assert np.shape(W) == np.shape(Km)

    return (sqrt(Kp), sqrt(Km))


def f_function(w, params:NondimensionalBeamParameters):
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




def compute_frequencies_wavenumbers(w, tolerance:float, params:NondimensionalBeamParameters):
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

        
def compute_modal_amplitutes(w, kp, km, params:NondimensionalBeamParameters):
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
    #IMP!: This calculation suffers from numerical instability due
    #to subtraction of large values 
    apc = (g*Km - W)*(-kp*np.sin(km*L) + km*np.sin(kp*L));
    aps = km*((g*Kp - W)*np.cos(km*L) - (g*Km - W)*np.cos(kp*L));
    amc = (g*Kp - W)*(kp*np.sin(km*L)- km*np.sin(kp*L));
    ams = kp*(-(g*Kp - W)*np.cos(km*L) + (g*Km - W)*np.cos(kp*L));

    assert np.shape(apc) == np.shape(w)
    assert np.shape(aps) == np.shape(w)
    assert np.shape(amc) == np.shape(w)
    assert np.shape(ams) == np.shape(w)

    return (apc,aps,amc,ams)

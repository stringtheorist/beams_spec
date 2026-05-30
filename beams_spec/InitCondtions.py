import numpy as np


from .BeamProblem import TimoshenkoAdimParams



#Initial conditions from Loic's initial example
#IMPORTANT : For the moment type annotations are terrible with NumPy so
#they will be left out in all code that handles arrays.
#Which means pretty much all the code in this project unfortunately. 
def v1_s0(x, prms:TimoshenkoAdimParams):
    return 0.0

def e1_s0(x, prms:TimoshenkoAdimParams):
    return 0.0

def w2_s0(x, prms:TimoshenkoAdimParams):
    return 0.0

def k2_s0(x, prms:TimoshenkoAdimParams):
    r = prms.L/2.0
    k_init = 1 / r
    #assert k_init.shape == x.shape
    return k_init

def phi1_s0(x, prms:TimoshenkoAdimParams):
    r = prms.L/2.0
    delta_phi1 = r*(np.cos(x/r) - 1.0)
    phi1_init = prms.phi1_00 + delta_phi1
    #assert phi1_init.shape == x.shape
    return phi1_init

def phi3_s0(x, prms:TimoshenkoAdimParams):
    r = prms.L/2.0
    delta_phi3 = r*np.sin(x/r) - 1.0
    phi3_init = prms.phi3_00 + delta_phi3
    #assert phi3_init.shape == x.shape
    return phi3_init

def theta_s0(x, prms:TimoshenkoAdimParams):
    r = prms.L/2.0
    delta_theta = x / r
    theta_init = prms.theta_00 + delta_theta
    #assert theta_init.shape == x.shape
    return theta_init



import numpy as np


from .beams_spec_structures import NondimensionalBeamParameters



#Initial conditions from Loic's initial example
#IMPORTANT : For the moment type annotations are terrible with NumPy so
#they will be left out in all code that handles arrays.
#Which means pretty much all the code in this project unfortunately. 
def v1_init(x, params:NondimensionalBeamParameters):
    return 0.0

def e1_init(x, params:NondimensionalBeamParameters):
    return 0.0

def o2_init(x, params:NondimensionalBeamParameters):
    return 0.0

def k2_init(x, params:NondimensionalBeamParameters):
    r = params.L/2.0
    k_init = 1 / r
    #assert k_init.shape == x.shape
    return k_init

def phi1_s0(x, params:NondimensionalBeamParameters):
    r = params.L/2.0
    delta_phi1 = r*(np.cos(x/r) - 1.0)
    phi1_init = params.phi1_00 + delta_phi1
    #assert phi1_init.shape == x.shape
    return phi1_init

def phi3_s0(x, params:NondimensionalBeamParameters):
    r = params.L/2.0
    delta_phi3 = r*np.sin(x/r) - 1.0
    phi3_init = params.phi3_00 + delta_phi3
    #assert phi3_init.shape == x.shape
    return phi3_init

def theta_s0(x, params:NondimensionalBeamParameters):
    r = params.L/2.0
    delta_theta = x / r
    theta_init = params.theta_00 + delta_theta
    #assert theta_init.shape == x.shape
    return theta_init



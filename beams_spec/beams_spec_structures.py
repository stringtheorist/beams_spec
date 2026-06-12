from collections.abc import Callable

class TimoshenkoAdimParams:
    def __init__(self, L:float = 50.0, g:float = 2.5, phi1_00:float = 0.0,
                 phi3_00:float = 0.0, theta_00:float = 0.0):
        self.L = L
        self.g = g
        self.phi1_00 = phi1_00
        self.phi3_00 = phi3_00
        self.theta_00 = theta_00


# class TimoshenkoBeamProblem:
#     def __init__(self,
#                  params: TimoshenkoAdimParams,
#                  v1_s0: Callable[[float, TimoshenkoAdimParams], float],   
#                  e1_s0: Callable[[float, TimoshenkoAdimParams], float],   
#                  w2_s0: Callable[[float,TimoshenkoAdimParams], float],   
#                  k2_s0: Callable[[float,TimoshenkoAdimParams], float],
#                  phi1_s0: Callable[[float,TimoshenkoAdimParams], float],
#                  phi3_s0: Callable[[float,TimoshenkoAdimParams], float],
#                  theta_s0: Callable[[float,TimoshenkoAdimParams], float],
#                  phi1_00:float = 0.0,
#                  phi3_00:float = 0.0,
#                  theta_00:float = 0.0):

#         self.params = params
#         self.phi1_00 = phi1_00    # These are scalars 
#         self.phi3_00 = phi3_00    # 
#         self.theta_00 = theta_00  #
#         self.v1_s0   = v1_s0     #
#         self.e1_s0   = e1_s0     #
#         self.w2_s0   = w2_s0     #
#         self.k2_s0   = k2_s0     # These are functions (initial conditions)
#         self.phi1_s0 = phi1_s0   #
#         self.phi3_s0 = phi3_s0   #
#         self.theta_s0= theta_s0  #    
        
        

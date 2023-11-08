from Networks.Eco import Eco
from Networks.BRN import BRN
import numpy as np 
class EcoBRN(Eco, BRN):
    '''
    A BRN with ecology dynamics.
    '''
    def __init__(self, block_sizes, degree, name='', build=True):
        BRN.__init__(self, block_sizes, degree, name, 'eco', build)
        Eco.__init__(self)
        self.gt = np.array([0.1, 5, 1, 5, 1])
    
    def dxdt(self, x, t, param, degree=None):
        '''
        Return f(x) + degree * g(x, x)
        '''
        B, K, C, D, E = param 
        return B + x * (1 - x / K) * (x/C - 1) + self.deg_unique * x * x / (D + E*x)

    def unload(self, param):
        B, K, C, D, E = param 
        H = 0.1
        return np.array([B, K, C, D, E-H, H])
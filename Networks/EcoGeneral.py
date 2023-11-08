import numpy as np
from Networks.Eco import Eco 
from Networks.GeneralNet import GeneralNet

class EcoGeneral(Eco, GeneralNet):
    '''
    A general network with ecology dynamics.
    '''
    def __init__(self, N=10, p=3, name='', topo='er', m=4, seed=42):
        GeneralNet.__init__(self, N, p, 'eco', name, topo, m, seed=seed)
        self.gt = np.array([0.1, 5, 1, 5, 0.9, 0.1])
    def gradient(self, param, y, x):
        '''
        Compute the gradient d objective / d param.

        d steady-state / d param is computed using the full dynamics:
        F = f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        A = self.A
        N = self.N 

        B, K, C, D, E, H = self.unload(param) 

        Z = np.repeat(x, N).reshape((N, N))

        G = D + E * Z + H * (Z.T)
        
        Y = A * Z / G - H * A * np.outer(x, x) / (G * G)

        Y1 = A * Z.T / G - E * A * np.outer(x, x) / (G * G)

        diag = (1 - x / K) * (x / C - 1) + x * (1 - x / C) / K + x * (1 - x / K) / C + \
                np.sum(Y1, axis=1)

        np.fill_diagonal(Y, diag)

        XT = np.zeros((len(param), N))

        XT[0] = 1 # F_B 
        XT[1] = x * (x / C - 1) * x / (K ** 2) # F_K
        XT[2] = x * (x / K - 1) * x / (C ** 2) # F_C
        XT[3] = np.sum(- A * np.outer(x, x) / (G * G), axis=1) # F_D
        XT[4] = np.sum(- A * np.outer(x * x, x) / (G * G), axis=1) # F_E
        XT[5] = np.sum(- A * np.outer(x, x * x) / (G * G), axis=1) # F_H 

        dxdparam = - ( XT @ ( np.linalg.inv(Y)).T )
        dxdparam = dxdparam[:, self.obs_idx]

        gradient = dxdparam @ (x[self.obs_idx] - y[self.obs_idx])
        
        gradient /= len(self.obs_idx)
        
        return gradient
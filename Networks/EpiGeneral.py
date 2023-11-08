import numpy as np
from Networks.Epidemic import Epidemic
from Networks.GeneralNet import GeneralNet

class EpiGeneral(Epidemic, GeneralNet):
    '''
    A general network with epidemic dynamics.
    '''
    def __init__(self, N=10, p=3, name='', topo='er', m=4, seed=42):
        GeneralNet.__init__(self, N, p, 'epi', name, topo, m, seed=seed)
        self.gt = np.array([0.5])
    def gradient(self, param, y, x):
        '''
        Compute the gradient d objective / d param.

        d steady-state / d param is computed using the full dynamics:
        F = f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        A = self.A
        N = self.N
        B = 1
        R = param[0]
        Z = np.repeat(x, N).reshape((N, N))
        Y = A * (R * (1 - Z))
        diag = - B - R * (A @ x)
        np.fill_diagonal(Y, diag)
        XT = np.zeros((len(param), N))
        #XT[0] = - x
        XT[0] = (1 - x) * (A @ x)
        
        dxdparam = - ( XT @ ( np.linalg.inv(Y)).T )
        dxdparam = dxdparam[:, self.obs_idx]

        gradient = dxdparam @ (x[self.obs_idx] - y[self.obs_idx])
        
        gradient /= len(self.obs_idx)
        
        return gradient
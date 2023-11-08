import numpy as np
from Networks.Gene import Gene 
from Networks.GeneralNet import GeneralNet

class GeneGeneral(Gene, GeneralNet):
    '''
    A general network with gene regulatory dynamics.
    '''
    def __init__(self, N=10, p=3, name='', topo='er', m=4, seed=42):
        GeneralNet.__init__(self, N, p, 'gene', name, topo,m, seed=seed)
        self.gt = np.array([1,1,2])

    def gradient(self, param, y, x):
        '''
        Compute the gradient d objective / d param.

        d steady-state / d param is computed using the full dynamics:
        F = f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        ''' 
         
        A = self.A
        N = self.N
        B, f, h = param  

        g2 = h * (x ** (h-1)) / ((x ** h + 1) ** 2)
         
        Y = np.repeat(g2, N).reshape(N,N)
        Y = A * (Y.T)
        diag = - B * f * (x ** (f - 1))
        np.fill_diagonal(Y, diag)
        Z = np.repeat(x, N).reshape(N,N) 
        
        XT = np.zeros((len(param), N))
        XT[0] = - x ** f 
        XT[1] = - B * np.log(x) * (x ** f) 
        XT[2] = np.sum(A * (Z.T ** h) * np.log(Z.T) / (Z.T ** h + 1) ** 2, axis=1) 
        
        dxdparam = - ( XT @ ( np.linalg.inv(Y)).T ) 
        dxdparam = dxdparam[:, self.obs_idx]

        gradient = dxdparam @ (x[self.obs_idx] - y[self.obs_idx])
        
        gradient /= len(self.obs_idx)
        
        return gradient 
    
     
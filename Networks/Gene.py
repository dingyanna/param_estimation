import numpy as np
from Networks.Net import Net

class Gene(Net):
    def f(self, x):
        B,f,h = self.gt
        return - B * (x ** f)
    def g(self, x, y ):
        B,f,h = self.gt
        return (y ** h) / (y ** h + 1)
 
    def dxdt_mfa(self, x, t, param, degree):
        '''
        Return the N-D mean-field formula f(x, param) + degree * g(x, xeff, param).
        '''
        xeff = self.xeff
        B, f, h = param
        xh = xeff ** h 
        xf = x ** f 
        return - B * xf + degree * (xh / (xh + 1))
    
    def dxdt_xeff(self, x, t, param, beta=None):
        '''
        Return the 1-D mean-field formula f(x, param) + beta * g(x, x, param).
        '''
        if beta is None:
            beta = self.beta
        B, f, h = param
        xf = x ** f  
        xh = x ** h 

        dxdt = - B * xf + beta * (xh / (xh + 1))
        return dxdt  
    
    def dxdt(self, x, t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        if A is None:
            A = self.A
        B, f, h = param
        xf = x ** f 
        xh = x ** h
        dxdt = - B * xf + A @ (xh / (xh + 1))
        return dxdt  
 
    def gradient_block(self, param, y, x):
        '''
        Compute the gradient d objective / d param, which requires the sensitivity d steady-state / d param.

        d steady-state / d param is computed using the formula:
        f(x, param) + beta g (x, x, param) = 0
        
        Assumption:
        the given network is a BRN and thus beta is equal to degree.
        '''
        degree = self.deg_unique
        B, f, h = self.unload(param)

        F_B = - x ** f 
        F_f = - B * (x ** f) * np.log(x)
        F_h = degree * (x ** h) * np.log(x) / (x ** h + 1) ** 2
        F_x = - B * f * (x ** (f - 1)) + degree * (h * x ** (h-1)) / (x ** h + 1) ** 2

        dF_dB = (x - y) @ (- F_B / F_x)
        dF_df = (x - y) @ (- F_f / F_x)
        dF_dh = (x - y) @ (- F_h / F_x)

        gradient = np.array([dF_dB, dF_df, dF_dh])
        gradient /= len(degree)
        return gradient 
    
    def gradient_mfa(self, param, y1, x1):
        '''
        Compute the gradient d objective / d param.

        d steady-state / d param is computed using the formula:
        f(x, param) + degree g (x, xeff, param) = 0
        '''
        xeff = self.xeff
        degree = self.degree[self.obs_idx]
        y = y1[self.obs_idx]
        x = x1[self.obs_idx] 

        B, f, h = param
        
        F_B = - x ** f 
        F_f = - B * (x ** f) * np.log(x)
        F_h = degree * (xeff ** h) * np.log(xeff) / (xeff ** h + 1) ** 2
        F_x = - B * f * (x ** (f - 1))

        dF_dB = (x - y) @ (- F_B / F_x)
        dF_df = (x - y) @ (- F_f / F_x)
        dF_dh = (x - y) @ (- F_h / F_x)

        gradient = np.array([dF_dB, dF_df, dF_dh])
        gradient /= len(degree)
        return gradient
    
     
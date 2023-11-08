import numpy as np
from Networks.Net import Net
import warnings
warnings.filterwarnings('default')
from scipy.integrate import odeint

class Eco(Net):
    '''
    Create a set of methods for Ecology dynamics.
    ''' 
    
    def gradient_block(self, param, y, x):
        '''
        Compute the gradient d objective / d param, which requires the sensitivity d steady-state / d param.

        d steady-state / d param is computed using the formula:
        f(x, param) + beta g (x, x, param) = 0
        
        Assumption:
        the given network is a BRN and thus beta is equal to degree.
        '''
        degree = self.degree
        B, K, C, D, E = param 
        F_B = 1 
        F_K = x * (x / K ** 2) * (x / C - 1)
        F_C = x * (1 - x / K) * (- x / (C ** 2))
        
        F_D = - degree * x * x / ((D + E * x) ** 2)
        F_E = - degree * x * x * x / ((D + E * x) ** 2)
        F_x = 2 * x * (1 / C + 1 / K) - 3 * x * x / (K * C) - 1 +\
            degree * (2 * x / (D + E * x) - x * x * E / ((D + E * x) ** 2))
        
        x_B = - F_B / F_x 
        x_K = - F_K / F_x    
        x_C = - F_C / F_x
        x_D = - F_D / F_x
        x_E = - F_E / F_x

        dF_dB = (x - y) @ x_B 
        dF_dK = (x - y) @ x_K
        dF_dC = (x - y) @ x_C
        dF_dD = (x - y) @ x_D
        dF_dE = (x - y) @ x_E
        gradient = np.array([dF_dB, dF_dK, dF_dC, dF_dD, dF_dE])
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

        B, K, C, D, E, H = param 
        F_B = 1
        F_K = x * (x / K ** 2) * (x / C - 1)
        F_C = x * (1 - x / K) * (- x / (C ** 2))
        
        F_D = - degree * x * xeff / ((D + E * x + H * xeff) ** 2)
        F_E = - degree * x * x * xeff / ((D + E * x + H * xeff) ** 2)
        F_H = - degree * x * xeff * xeff / ((D + E * x + H * xeff) ** 2)
        F_x = 2 * x * (1 / C + 1 / K) - 3 * x * x / (K * C) - 1 +\
            degree * (xeff / (D + E * x + H * xeff) - \
                     x * xeff * E / ((D + E * x + H * xeff) ** 2))
        x_B = - F_B / F_x 
        x_K = - F_K / F_x    
        x_C = - F_C / F_x
        x_D = - F_D / F_x
        x_E = - F_E / F_x
        x_H = - F_H / F_x

        dF_dB = (x - y) @ x_B 
        dF_dK = (x - y) @ x_K
        dF_dC = (x - y) @ x_C
        dF_dD = (x - y) @ x_D
        dF_dE = (x - y) @ x_E
        dF_dH = (x - y) @ x_H
        gradient = np.array([dF_dB, dF_dK, dF_dC, dF_dD, dF_dE, dF_dH]) 
        gradient /= len(degree)
        
        return gradient

    def dxdt_mfa(self, x,t, param, degree):
        '''
        Return the N-D mean-field formula f(x, param) + degree * g(x, xeff, param).
        '''  
        xeff = self.xeff
        B,K,C,D,E,H = param
        return B + x * (1 - x/K) * (x/C - 1) + degree * (x*xeff) / (D + E*x + H*xeff)
    
     
    def dxdt_xeff(self, x,t, param, beta=None):
        '''
        Return the 1-D mean-field formula f(x, param) + beta * g(x, x, param).
        '''
        B,K,C,D,E,H = param
        return B + x * (1 - x/K) * (x/C - 1) + beta * (x ** 2) / (D + (E+H)*x) 
            
    def get_param_err(self, param, gt):
        '''
        Return relative parameter error
        '''
        if len(self.gt) == 5:
            if len(param) == 6:
                B,K,C,D,E,H = param 
                param = np.array([B,K,C,D,E+H])
        else: 
            if len(param) == 5:
                B,K,C,D,Ep = param 
                H = 0.1
                param = np.array([B,K,C,D,Ep-H,H])
        
        # K and C are interchangeable.
        param_temp = param.copy()
        param_temp[1] = param[2]
        param_temp[2] = param[1]
        param_err = min(np.mean(np.abs((param - gt) / gt)), np.mean(np.abs((param_temp - gt) / gt))) 
        if np.mean(np.abs((param - gt) / gt)) < np.mean(np.abs((param_temp - gt) / gt)):
            return param_err
        else:
            return param_err 

    def dxdt(self, x, t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        if A is None:
            A = self.A
        N = len(A)
        B,K,C,D,E,H = param
        X = np.repeat(x, N).reshape(N,N)
        M = (np.outer(x,x)) / (D + E * X + H * X.T)
        dxdt = B+x*(1-x/K)*(x/C-1)+ (A * M).sum(1)
        return dxdt

     
import numpy as np
from Networks.Net import Net

class Epidemic(Net):
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
        degree = self.deg_unique
        R = param[0]
        B = 1 
        F_R = degree * (1 - x) * x 
        F_x = - B + degree * R * (1 - 2 * x) 
        x_R = - F_x / F_R  
        derr_dR = (x - y) @ x_R 
        
        gradient = np.array([derr_dR])
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
        R = param[0]
        B = 1
        
        #F_B = - x
        F_R = degree * (1 - x) * xeff
        F_R[degree == 0] = 1 # to avoid division by zero
        
        F_x = - B - degree * R * xeff
         
        x_R = - F_x / F_R 
        x_R[degree == 0] = 0
 
        derr_dR = (x - y) @ x_R 
        
        gradient = np.array([derr_dR])
        gradient /= len(degree)
        return gradient

    def dxdt_mfa(self, x,t, param, degree):
        '''
        Return the N-D mean-field formula f(x, param) + degree * g(x, xeff, param).
        '''
        xeff = self.xeff
        R = param[0]
        B = 1
        return - B * x + degree * R * (1 - x) * xeff
    
    def dxdt_xeff(self, x,t, param, beta=None):
        '''
        Return the 1-D mean-field formula f(x, param) + beta * g(x, x, param).
        '''
        if beta is None:
            beta = self.beta
        R = param[0]
        B = 1 
        return - B * x + beta * R * (1 - x) * x


    def dxdt(self, x,t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        if A is None:
            A = self.A
        R = param[0]
        B = 1
        dxdt = - B * x + R * (1 - x) * (A @ x)
        return dxdt

     
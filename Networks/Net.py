import numpy as np
from scipy.integrate import odeint
import scipy.stats as stats 
import random 
 
class Net(object): 

    def unload(self, param):
        '''
        Unload param, add extra parameters if needed.
        '''
        return param

    def obj(self, param, y):
        '''
        Return the objective function value and the predicted state at param.
        Assume there are adversarial nodes.
        '''
        xhat = self.Dmap(param, y) 
        err = np.mean(np.square(xhat[self.obs_idx] - y[self.obs_idx]))
        return 0.5 * err, xhat 

    def get_param_err(self, param, gt):
        '''
        Return relative parameter error
        ''' 
        return np.mean(np.abs((param - gt) / gt))
    
    def get_steady_state(self, x0, t, param, A=None, h=0.001):
        '''
        Return steady states with initial condition x0, evolution time span t, and parameter param.
        '''
        if A is None:
            A = self.A
        if self.dyn == 'eco' and len(param) == 5: 
            print('mismatch: len(param) == 4 => extend param')
            B,K,C,D,Ep = param
            H = 0.1 
            param = np.array([B,K,C,D,Ep-H,H])  
        state = self.solve_ode(self.dxdt, x0, t, param, degree=A) 
        return state

    def apply_noise(self, ntype, noise_level, true_x):
        '''
        Add noise to true steady states true_x with noise type ntype, noise level noise_level.

        ntype: 
        0 - Lognormal
        1 - Gaussian with varying standard deviation
        2 - Adversarial noise
        3 - Gaussian i.i.d.
        4 - non-equilibrium
        ''' 
        if noise_level == 0:
            return true_x, 0
        N = self.N
        y = true_x
        if ntype == 0: # Lognormal
            print('Noise Type: Lognormal')
            lognormal = np.exp(np.random.normal(- (noise_level ** 2) / 2, noise_level, true_x.shape))
            y = true_x * lognormal
        elif ntype == 1: # Gaussian
            print('\nNoise Type: Gaussian\n')
            eps = stats.truncnorm.rvs(-1 / noise_level, np.inf / noise_level, loc=0, scale=noise_level, size=true_x.shape)
            y = true_x * (1 + eps)
        elif ntype == 2: # adversial
            print('Noise Type: adversarial')
            n_adv = int(N*noise_level)
            noise_level = 0.15
            eps = stats.truncnorm.rvs(-1 / noise_level, np.inf / noise_level, loc=0, scale=noise_level, size=true_x.shape)
            y = true_x * (1 + eps)
            print('Number of adversarial nodes', n_adv)
            inds = random.sample(range(N), n_adv)
            y[inds] = 0
            return y, inds
        elif ntype == 3: # Gaussian, i.i.d.
            print('\nNoise Type: Gaussian i.i.d\n')
            eps = np.random.normal(0, np.mean(true_x)*noise_level, true_x.shape)
            y = true_x + eps
            y[y<0] = 0
            print('<x> - <y>', np.mean(true_x) - np.mean(y)) 
        return y, 0 

    def dxdt_mfa_traj(self, x,t, param):
        '''
        Return the N-D mean-field formula f(x, param) + degree * g(x, xeff, param).
        '''
        self.xeff = np.mean(x) + t * self.dxdt_xeff(np.mean(x), t, param, self.beta)
        return self.dxdt_mfa(x, t, param, self.degree) 
    
    def mfa(self, param, y):
        '''
        Implement D^mfa.
        '''
        self.xeff = self.get_xeff(param, y, self.beta)[0]
        t_final = 10
        nsteps = 10
        t = np.linspace(0,t_final,t_final*nsteps)
        x_unique = self.solve_ode(self.dxdt_mfa, np.full(len(self.deg_unique), self.xeff), t, param, self.deg_unique)
        xhat = np.zeros(self.N)
        for i in range(len(self.deg_unique)):
            xhat[self.degree == self.deg_unique[i]] = x_unique[i]
        return xhat

    def block(self, param, y):
        '''
        Solve f(x, param) + degree * g(x, x, param) = 0 for x.
        '''
        self.xeff = self.get_xeff(param, y, self.beta)
        t = np.linspace(0,5,500)
        x_unique = self.solve_ode(self.dxdt_xeff, np.full(len(self.deg_unique), self.xeff), t, param, self.deg_unique)
        xhat = np.zeros(self.N)
        for i in range(len(self.deg_unique)):
            xhat[self.degree == self.deg_unique[i]] = x_unique[i]
        return xhat
    
    def get_xeff(self, param, y, beta):
        '''
        Solve f(x, param) + beta * g(x, x, param) = 0 for x.
        '''
        t_final = 5
        nsteps = 1
        t = np.linspace(0,t_final,t_final*nsteps)
        xeff = self.solve_ode(self.dxdt_xeff, np.array([np.mean(y)]), t, param, beta)
        return xeff
    
    def block_to_all(self, xbar):
        '''
        Copy block state to individual nodes according to degrees.
        '''
        x = np.zeros(self.N)
        for i in range(len(self.deg_unique)):
            x[self.degree == self.deg_unique[i]] = xbar[i]
        return x
    
    def all_to_block(self, x):
        '''
        Take the average of states with the same degree.
        '''
        xbar = np.zeros(len(self.deg_unique))
        for i in range(len(self.deg_unique)):
            xbar[i] = np.mean(x[self.degree == self.deg_unique[i]])
        return xbar 
    
    def L(self, x):
        return np.sum(self.A @ x) / np.sum(self.A) 
    
    def solve_ode(self, dxdt, x0, t, param, degree=None):
        '''
        Iteratively calls odeint to ensure correctness of steady states, i.e., dxdt close to zero.
        '''
        res = odeint(dxdt, y0=x0, t=t, args=(self.unload(param), degree))[-1]
        max_iter = 0
        itr = 0
        while np.mean(np.abs(dxdt(res, t, self.unload(param), degree))) > 1e-8:  
            if itr == max_iter:
                break
            res = odeint(dxdt, y0=res, t=t, args=(self.unload(param), degree))[-1]
            itr += 1 
        return res
    
     
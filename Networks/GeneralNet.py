import networkx as nx
import numpy as np
from scipy.optimize import root
from Networks.Net import Net
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

class GeneralNet(Net):
    '''
    A general network.
    '''
    def __init__(self, N, p, dyn, name, topo='er', m=4, directed=False, seed=42):
        Net.__init__(self)
        self.N = N
        self.name = name
        self.topo = topo
        self.dyn = dyn
        if topo == 'er':
            G = nx.fast_gnp_random_graph(N, p, seed=seed)
        else:
            G = nx.barabasi_albert_graph(N, m, seed=seed)
        self.G = G
        self.A = nx.to_numpy_array(G, nodelist=range(N))
        self.degree = self.A.sum(axis=0) # in degree
        self.deg_unique, self.block_sizes = np.unique(self.degree, return_counts=True)
        self.beta = np.sum(self.A @ self.degree) / np.sum(self.A)
        self.H = np.std(self.degree) ** 2 / np.mean(self.degree)
        self.sampled = range(N)
        self.Dmap = self.mfa_plus
         
        self.directed = directed
        self.int_step_size = 0.001
    
    def setTopology(self, A):
        self.A = A
        self.degree = self.A.sum(axis=1)
        self.out_degree = self.A.sum(0)
        binA = np.zeros((len(A),len(A)))
        binA[A != 0] = 1
        self.degree_k = binA.sum(axis=1) # to distinguish between degree and weighted degree
        self.deg_unique, self.block_sizes = np.unique(self.degree_k, return_counts=True)
        self.beta = np.sum(self.A @ self.degree) / np.sum(self.A)
        self.N = A.shape[0]
        self.H = np.std(self.degree) ** 2 / np.mean(self.degree)
        self.obs_idx = list(range(self.N))
    
    def mfa_plus(self, param, y):
        '''
        Implement D^mfa+
        '''
        xhat = self.mfa(param, y) 
        t_final = 5
        nsteps = 1
        t = np.linspace(0,t_final,t_final*nsteps)
        xhat = self.solve_ode(self.dxdt, xhat, t, param) 
        return xhat
    
    def mfa_plus_partial(self, param, y):
        '''
        Implement D^mfa+, assuming partial topology is given.
        '''
        t = np.linspace(0,5,500)
        self.xeff = self.get_xeff(param, y, self.beta_partial)
        self.degree_partial = self.A_partial.sum(1) + self.missing_degree
        unique_deg = np.unique(self.degree_partial)
        x_unique = self.solve_ode(self.dxdt_mfa, np.full(len(unique_deg), self.xeff), t, param, unique_deg)
        xhat = np.zeros(self.Np)
        for i in range(len(self.deg_unique_partial)):
            xhat[self.degree_partial == self.deg_unique_partial[i]] = x_unique[i]
        xhat = self.solve_ode(self.dxdt_partial, xhat, t, param)
        return xhat

    def ode(self, param, y):
        '''
        Implement D^full
        '''
        t_final = 10
        nsteps = 1
        t = np.linspace(0,t_final,int(t_final*nsteps))
        xhat = self.solve_ode(self.dxdt, y, t, param) 
        return xhat
    
    def ode_partial(self, param, y):
        t = np.linspace(0,50,5000)
        self.xeff = self.get_xeff(param, y, self.beta_partial)
        state = odeint(self.dxdt_partial, y0=y, t=t, args=(self.unload(param),), atol=1e-15)[-1]
        state = root(self.dxdt_partial, state, args=(0, self.unload(param),), tol=1e-14).x
        return state

    def print_stats(self, y, x, logger):
        '''
        Print statistics about the network.
        '''
        logger.info(f'==> Simulating   {self.name} Network')
        logger.info(f'[size]           {self.N}')
        logger.info(f'[state]          {np.mean(x)}')
        logger.info(f'[obs err]        {np.mean(np.square(y - x))}')
        logger.info(f'[degree]         {np.mean(self.degree)}')
        logger.info(f'[heterogeneity]  {self.H}')
        logger.info(f'[noise]          {np.round(100*np.mean(np.abs( (y-x)/x )), 5)}%\n')
        logger.info(f'[directed]       {self.directed}\n')

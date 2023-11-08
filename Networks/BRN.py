import numpy as np
from Networks.Net import Net
import networkx as nx
from scipy.integrate import odeint


class BRN(Net):
    '''
    A block regular network.
    '''
    def __init__(self, block_sizes, degree, name, dyn, build=True):
        self.block_sizes = np.array(block_sizes, dtype=int) # block size, e.g., [30, 40, 50]
        self.deg_unique = degree 
        self.name = name
        self.dyn = dyn
        self.N = np.sum(self.block_sizes)
        if build:
            # create the corresponding list of regular networks
            Gs = []
            print('==> Constructing BRN')
            for i in range(len(degree)): 
                G = nx.random_regular_graph(int(degree[i]),int(block_sizes[i]))
                Gs.append(G)
            G = nx.disjoint_union_all(Gs)
            self.A = nx.to_numpy_array(G, nodelist=range(len(G)))
            self.degree = self.A.sum(0)
            self.beta = np.mean(self.degree * self.degree) / np.mean(self.degree)
        self.degree = self.block_to_all(self.deg_unique)
        self.beta = np.mean(self.degree * self.degree) / np.mean(self.degree)
        assert(len(self.block_sizes) == len(self.deg_unique))
        self.topo = 'brn'
        self.Dmap = self.block
    
    def setTopology(self, deg_unique, bsize):       
        self.deg_unique = deg_unique
        self.block_sizes = bsize  
        self.N = np.sum(self.block_sizes)
        self.degree = self.block_to_all(self.deg_unique)
        self.beta = np.mean(self.degree * self.degree) / np.mean(self.degree)
        self.obs_idx = list(range(self.N))

    def block_to_all(self, xbar):
        '''
        Copy block state to individual nodes according to degrees.
        '''
        x = np.zeros(self.N)
        for i in range(len(xbar)):
            lb = np.sum(self.block_sizes[:i])
            ub = np.sum(self.block_sizes[:i+1])
            x[lb:ub] = xbar[i].copy()
        return x
    
    def all_to_block(self, x):
        '''
        Take the average of states with the same degree.
        '''
        xbar = np.zeros(len(self.block_sizes))
        for i in range(len(self.block_sizes)):
            lb = np.sum(self.block_sizes[:i])
            ub = np.sum(self.block_sizes[:i+1])
            xbar[i] = x[lb:ub].mean()
        return xbar

    def get_steady_state(self, x0, t, param):
        '''
        Return steady states with initial condition x0, evolution time span t, and parameter param.
        ''' 
        self.xeff = self.get_xeff(param, x0, self.beta)
        x_unique = odeint(self.dxdt_xeff, y0=np.full(len(self.deg_unique), self.xeff), t=t, args=(self.unload(param), self.deg_unique), atol=1e-15)[-1]
        state = self.block_to_all(x_unique)
        return state
    
    def print_stats(self, y, x, logger):
        '''
        Print statistics about the network.
        '''
        logger.info(f'==>  {self.name} Block-Regular Network')
        logger.info(f'[size]             {self.N}')
        logger.info(f'[degree]           {self.deg_unique}')
        logger.info(f'[block sizes]      {self.block_sizes}') 
        logger.info(f'[avg state]        {np.mean(x)}') 
        logger.info(f'[obs error]        {np.mean(np.square(y - x))}')
        logger.info(f'[noise]            {np.round(np.mean(np.abs((y - x) / x)) * 100, 5)}%')
    
from Networks.Gene import Gene 
from Networks.BRN import BRN
import numpy as np
class GeneBRN(Gene, BRN):
    '''
    A BRN with gene regulatory dynamics.
    '''
    def __init__(self, block_sizes, deg_unique, name='', build=True):
        BRN.__init__(self, block_sizes, deg_unique, name, 'gene', build)

        self.gt = np.array([1,1,2])
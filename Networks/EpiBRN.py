from Networks.Epidemic import Epidemic
from Networks.BRN import BRN
import numpy as np 
class EpiBRN(Epidemic, BRN):
    '''
    A BRN with gene regulatory dynamics.
    '''
    def __init__(self, block_sizes, deg_unique, name='', build=True):
        BRN.__init__(self, block_sizes, deg_unique, name, 'epi', build)
        self.gt = np.array([0.5])
     
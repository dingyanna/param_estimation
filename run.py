from experiment import *
import argparse
parser = argparse.ArgumentParser('Dynamics Inference')
parser.add_argument('--experiment', type=str, default='efficiency', help='perturb, topo, adv_recover, noise_level, network_size, adv, plot_loss, compute_ss, brn_vs_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dynamics', type=str, default='eco', help="eco, epi, gene")
parser.add_argument('--topology', type=str, default='er', help="er, brn, sf, real")
parser.add_argument('--data', type=str, default='', help="net6, net8")
parser.add_argument('--ub', type=float, default=10)
parser.add_argument('--x0', type=float, default=6.)
# Topology related
parser.add_argument('--n', type=int, default=200) # Total number of nodes
parser.add_argument('--k', type=int, default=12) # average degree (parameter for ER network)
parser.add_argument('--m', type=int, default=4) # parameter for scale free network
parser.add_argument('--w', type=float, default=1) # uniform interaction strength
parser.add_argument('--nu', type=float, default=0) # power law interaction strength
parser.add_argument('--gamma', type=float, default=0) # power law interaction strength

parser.add_argument('--bsize', type=int, default=100) # block size
parser.add_argument('--block_number', type=int, default=-1) # block number
parser.add_argument('--perturb_ratio', type=float, default=0) # topology perturb ratio
parser.add_argument('--perturb_type', type=int, default=0, help="0: rewire, 1: remove, 2: addition") # type of perturbation
parser.add_argument('--build_brn', type=bool, default=False) # type of perturbation


# optimization
parser.add_argument('--ss_solver', type=str, default='mfap', help="mfa, block_mfa, mfap, ode")
parser.add_argument('--line_search', type=int, default=0, help="0: brute-force, 1: secant method, 2: newton ralphson")
parser.add_argument('--tol', type=float, default=1e-8)
parser.add_argument('--max_iter', type=int, default=1000, help="maximal number of iteration of CGD")
parser.add_argument('--prob_theta0', type=str, default="u", help="the sampling method for the initial parameter guess; n - normal, u - uniform")
 
parser.add_argument('--noise_level', type=float, default=0.13)
parser.add_argument('--noise_type', type=int, default=1)


if __name__ == '__main__':
    args = parser.parse_args()
    topo = args.topology
    dyn = args.dynamics
    exp = args.experiment 
    if topo == 'real' and exp in ['nobserve', 'nblock_bsize']:
        print('Cannot take a real network when topology is varied in the experiment.')
        exit(1) 
    if exp == 'efficiency': 
        efficiency(args)
    elif exp == 'perturb': 
        perturb(args)  
    elif exp == 'topo':
        topo_increasing_n(args)
    elif exp == 'adv_recover':
        adv_recover(args)  
    elif exp == "noise_level":
        noise_level(args)
    elif exp == "network_size":
        network_size(args) 
    elif exp == "plot_loss":
        plot_landscape(args) 
    elif exp == "compute_ss":
        compute_steady_state(args) 
    elif exp == "brn_vs_true":
        brn_vs_true(args)
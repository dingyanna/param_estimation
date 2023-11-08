import numpy as np
from CGD import CGD
from Networks.EcoGeneral import EcoGeneral
from Networks.GeneGeneral import GeneGeneral
from Networks.EpiGeneral import EpiGeneral
from Networks.EcoBRN import EcoBRN
from Networks.GeneBRN import GeneBRN
from Networks.EpiBRN import EpiBRN
 
import random
import time
from utils import *  
import os 
from scipy.integrate import odeint
import pandas as pd
import argparse   
import datetime
import calendar
eps_state = 1e-7

def create_net(args):
    if args.topology != 'brn':
        n = args.n
        avg_d = args.k
        if args.dynamics == 'eco':
            net = EcoGeneral(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
        elif args.dynamics == 'gene':
            net = GeneGeneral(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
        else:
            net = EpiGeneral(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
        if args.topology == 'real':
            topo = load_real_net(f'./data/{args.data}.edges.csv')
            net.setTopology(topo)
    else:   
        bsize = args.bsize
        deg_unique = np.array([7, 13, 17, 23, 29])
        if args.block_number > 0: # default is -1
            # sample degree
            random.seed(0)
            deg_unique = np.array(random.sample(range(5,25), args.block_number))
        block_sizes = np.ones(len(deg_unique)) * bsize
        if args.dynamics == 'eco':
            net = EcoBRN(block_sizes, deg_unique, build=args.build_brn)
        elif args.dynamics == 'gene':
            net = GeneBRN(block_sizes, deg_unique, build=args.build_brn)
        elif args.dynamics == 'epi':
            net = EpiBRN(block_sizes, deg_unique, build=args.build_brn)
    gt = read_param(args)
    net.gt = gt
    net.obs_idx = list(range(net.N))
    return net

def mape(xhat, true_x, eps_state=1e-7):
    return np.mean(np.abs((true_x[np.abs(true_x) > eps_state] - xhat[np.abs(true_x) > eps_state]) / true_x[np.abs(true_x) > eps_state])) * 100

 
def learn_dynamics(args, net, logger, init_param, y, true_x, x0, t):    
    gt = net.gt
    observation = y
    
    obs_err = mape(y, true_x)
    # set gradient and steady-state solver according to steady-state solvers
    method = args.ss_solver 
    if method == 'ode' or method == "full":
        net.Dmap = net.ode
        net.gradient = net.gradient
    elif method =='mfap':
        net.Dmap = net.mfa_plus
        net.gradient = net.gradient
    elif method == 'mfa' or method == "mfa+":
        net.Dmap = net.mfa
        net.gradient = net.gradient_mfa
    else:
        net.Dmap = net.block
        net.gradient = net.gradient_block
 
    bound = [0, args.ub] 
    tol = args.tol 
    net.print_stats(y, true_x, logger) 
    inc1 = 2 
    inc2 = 2
    cgd = CGD(net.obj, net.gradient, net.get_param_err, bound, observation, tol=tol, max_iter=args.max_iter, logger=logger, normalize_direction=True, inc1=inc1, inc2=inc2)
 
    t1 = time.time()
    param, err, param_err, state_err = cgd.run(init_param, gt, true_x)
    t2 = time.time()
    rt1 = t2 - t1
    serr_improvement = ( (obs_err - state_err) / state_err ) 

    return state_err, param_err, serr_improvement, rt1, param 

def topo_increasing_n(args):
    np.random.seed(args.seed)
    args1 = argparse.Namespace(**vars(args))
    args1.topology = 'brn'
    args1.ss_solver = 'block'

    net = create_net(args)
    brn = create_net(args1) 

    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"

    if args.topology == 'real':
        trail = f"{args.dynamics}_{args.data}"
    else:
        trail = f"{args.dynamics}_{args.topology}"
    
    log_path = f"./results/{date}/{args.experiment}/{trail}/result_{args.seed}_nlevel{args.noise_level}_p0{args.prob_theta0}.log"
    if not os.path.exists(f"./results/{date}/{args.experiment}/{trail}"):
        os.makedirs(f"./results/{date}/{args.experiment}/{trail}")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(str(args))

    param = sample_param(net.gt, args.ub, args.dynamics, args.prob_theta0)
    if args.dynamics == "eco":
        param_brn = np.array([param[0], param[1], param[2], param[3], param[4]+param[5]])
    else:
        param_brn = param 
 
    Ns = np.linspace(100, 500, 11) 
    deg_unique = np.array([7, 13, 17, 23, 29])
    ntype = args.noise_type
    noise_level = args.noise_level
    for i in range(len(Ns)):
        n = int(Ns[i])
        # fix average degree
        p = 13 / n 
        G = nx.fast_gnp_random_graph(n, p, seed=42)
        A = nx.to_numpy_array(G, nodelist=range(len(G)))
        net.setTopology(A)
        true_x, x0, t = get_steady_state(args, net, net.gt)
        y, _ = net.apply_noise(ntype, noise_level, true_x) 
        learn_dynamics(args, net, logger, param, y, true_x, x0, t)
        
        block_sizes = np.full(len(deg_unique), int(n/len(deg_unique)))
        brn.setTopology(deg_unique, block_sizes)
        true_x, x0, t = get_steady_state(args1, brn, brn.gt)
        y, _ = net.apply_noise(ntype, noise_level, true_x) 
        learn_dynamics(args1, brn, logger, param_brn, y, true_x, x0, t)
    
        # fix probability of edge creation
        p = 0.12
        G = nx.fast_gnp_random_graph(n, p, seed=42)
        A = nx.to_numpy_array(G, nodelist=range(len(G)))
        net.setTopology(A)
        true_x, x0, t = get_steady_state(args, net, net.gt) 
        y, _ = net.apply_noise(ntype, noise_level, true_x) 
        learn_dynamics(args, net, logger, param, y, true_x, x0, t)

        block_sizes1 = np.full(int(n/20), 20)
        deg_unique1 = np.array(random.sample(range(5,185), int(n/20)))
        brn.setTopology(deg_unique1, block_sizes1)
        true_x, x0, t = get_steady_state(args1, brn, brn.gt)
        y, _ = net.apply_noise(ntype, noise_level, true_x) 
        learn_dynamics(args1, brn, logger, param_brn, y, true_x, x0, t)



def sample_param(gt, ub, dyn, prob_theta0="u"):

    if prob_theta0 == "u":
        print("sampling uniformly")
        if dyn != "eco":
            return np.random.uniform(0.09, ub, len(gt))
        else:
            return np.random.uniform(0.01, ub, len(gt))
    print("sampling normally")
    param = np.random.normal(gt, 0.2*gt)
    for i in range(len(param)):
        if param[i] < 0.09:
            param[i] = 0.09 # impose a lower bound for all parameters
        if dyn == "gene" and param[i] > ub: # impose an upper bound for gene parameters
            param[i] = ub
    return param
def get_steady_state(args, net, gt):
    if args.dynamics == 'epi':
        x0 = np.full(len(net.degree), .5)
        t = np.linspace(0, 10,1000)
    else:
        x0 = np.full(len(net.degree), 6.)
        t = np.linspace(0, 10,1000)
    true_x = net.get_steady_state(x0, t, gt)
    return true_x, x0, t



def noise_level(args):
    
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"

    if args.topology == 'real':
        trail = f"{args.dynamics}_{args.data}"
    elif args.topology == 'er':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.k}_{args.nu}"
    elif args.topology == 'sf':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.gamma}_{args.nu}"
    log_path = f"./results/{date}/{args.experiment}/{trail}/result_{args.seed}.log"
    if not os.path.exists(f"./results/{date}/{args.experiment}/{trail}"):
        os.makedirs(f"./results/{date}/{args.experiment}/{trail}")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(str(args))
    # Create network
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    net = create_net(args)
    
    
    # Load ground truth parameters
    gt = read_param(args)
    net.gt = gt 
    

    initial_param = sample_param(gt, args.ub, args.dynamics, args.prob_theta0)

    true_x, x0, t = get_steady_state(args, net, net.gt)
    n_level = np.linspace(0.01, 0.4, 20)
    
    state_err = [ ]
    state_err_exact = [ ]
    runtime = [ ]
    param_err = [ ]
    improvement = [ ]
    obs_err = [ ]
    ode_system = ["mfa"]
    if args.dynamics == "eco":
        param_label = ["B", "K", "C", "D", "E", "H"]
    elif args.dynamics == "gene":
        param_label = ["B", "f", "h"]
    else:
        param_label = ["B"]
    all_params = []
    for i in range(len(param_label)):
        all_params.append([])
    for i in range(len(ode_system)):
        state_err.append([])
        state_err_exact.append([])
        runtime.append([])
        param_err.append([])
        improvement.append([])
        obs_err.append([])
        for j in range(len(param_label)):
            all_params[j].append([])

    ntype = args.noise_type 

    for noise_level in n_level:
        
        y, _ = net.apply_noise(ntype, noise_level, true_x)
        for i in range(len(ode_system)):
            ode = ode_system[i]
            logger.info(f"\nODE System: {ode}\n")
            args.ss_solver = ode 
            se, pe, se_imp, rt, param = learn_dynamics(args, net, logger, initial_param, y, true_x, x0, t)
            xhat, x0, t = get_steady_state(args, net, param)
            se1 = mape(xhat, true_x)
            state_err[i].append(se)
            state_err_exact[i].append(se1)
            runtime[i].append(rt)
            param_err[i].append(pe)
            improvement[i].append(se_imp)
            oe = mape(y, true_x)
            obs_err[i].append(oe)

            for j in range(len(param)):
                all_params[j][i].append(param[j])
    
    for i in range(len(ode_system)):
        res = {
            "noise_level": n_level,
            "runtime": runtime[i],
            "state err": state_err[i],
            "param err": param_err[i],
            "improvement": improvement[i],
            "state err exact": state_err_exact[i],
            "obs err": obs_err[i]
        }
        for j in range(len(param_label)):
            pl = param_label[j]
            res[pl] = all_params[j][i]
        pd.DataFrame(res).to_csv(f'./results/{date}/{args.experiment}/{trail}/result{args.seed}_{ode_system[i]}.csv', index=None )

def network_size(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"

    if args.topology == 'real':
        trail = f"{args.dynamics}_{args.data}"
    elif args.topology == 'er':
        trail = f"{args.dynamics}_{args.topology}"
    elif args.topology == 'sf':
        trail = f"{args.dynamics}_{args.topology}"
    log_path = f"./results/{date}/{args.experiment}/{trail}/result_{args.seed}.log"
    if not os.path.exists(f"./results/{date}/{args.experiment}/{trail}"):
        os.makedirs(f"./results/{date}/{args.experiment}/{trail}")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(str(args))
    # Create network
    net = create_net(args)
    N = np.linspace(50, 1000, 20)

    initial_param = sample_param(net.gt, args.ub, args.dynamics)

    prob_edge = [0.05, 0.5, 0.9]
    gammas = [2.1, 3, 4]

    state_err = [ ]
    state_err_exact = [ ]
    runtime = [ ]
    param_err = [ ]
    improvement = [ ]
    imp_exact = []
    obs_err = [ ]
    if args.dynamics == "eco":
        param_label = ["B", "K", "C", "D", "E", "H"]
    elif args.dynamics == "gene":
        param_label = ["B", "f", "h"]
    else:
        param_label = ["beta"]
    all_params = []
    for i in range(len(param_label)):
        all_params.append([])
    for i in range(len(prob_edge)):
        state_err.append([])
        state_err_exact.append([])
        runtime.append([])
        param_err.append([])
        improvement.append([])
        imp_exact.append([])
        obs_err.append([])
        for j in range(len(param_label)):
            all_params[j].append([])
     
    for i in range(len(prob_edge)):
        for n in N:
            if args.topology == "er":
                G = nx.fast_gnp_random_graph(int(n), prob_edge[i], seed=args.seed)
                A = nx.to_numpy_array(G, range(len(G)))
            elif args.topology == "sf":
                deg = [int(random.paretovariate(gammas[i])) for j in range(int(n))]
                if np.sum(deg) % 2 == 1:
                    deg[random.sample(range(int(n)), 1)[0]] += 1
                G = nx.configuration_model(deg)
                G = nx.Graph(G)
                A = nx.to_numpy_array(G, range(len(G)))
            net.setTopology(A)
            true_x, x0, t = get_steady_state(args, net, net.gt)
            ntype = args.noise_type
            noise_level = args.noise_level
            y, _ = net.apply_noise(ntype, noise_level, true_x)

            se, pe, se_imp, rt, param = learn_dynamics(args, net, logger, initial_param, y, true_x, x0, t)
            xhat, x0, t = get_steady_state(args, net, param) 
            se1 = mape(xhat, true_x)
            print("state error exact", se1)
            
            state_err[i].append(se)
            state_err_exact[i].append(se1)
            runtime[i].append(rt)
            param_err[i].append(pe)
            improvement[i].append(se_imp)
            
            oe = mape(y, true_x) # observed error
            obs_err[i].append(oe)
            imp_exact[i].append((oe - se1) / se1)
            
            for j in range(len(param)):
                all_params[j][i].append(param[j])

    for i in range(len(prob_edge)):
        print("\n storing result\n")
        if args.topology == "er":
            end = f"prob_edge{prob_edge[i]}"
        else:
            end = f"gamma{gammas[i]}"
        
        res = {
            "N": [int(n) for n in N],
            "runtime": runtime[i],
            "state err": state_err[i],
            "param err": param_err[i],
            "improvement": improvement[i],
            "improvement exact": imp_exact[i],
            "state err exact": state_err_exact[i],
            "obs err": obs_err[i]
        }
        for j in range(len(param_label)):
            pl = param_label[j]
            res[pl] = all_params[j][i]
            
        pd.DataFrame(res).to_csv(f'./results/{date}/{args.experiment}/{trail}/result{args.seed}_{end}.csv', index=None )
   

def efficiency(args):
    '''
    Learn dynamics of the given network.
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"

    if args.topology == 'real':
        trail = f"{args.dynamics}_{args.data}"
    elif args.topology == 'er':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.k}_{args.nu}"
    elif args.topology == 'sf':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.gamma}_{args.nu}"
    log_path = f"./results/{date}/{args.experiment}/{trail}/result_{args.seed}_nlevel{args.noise_level}_p0{args.prob_theta0}.log"
    if not os.path.exists(f"./results/{date}/{args.experiment}/{trail}"):
        os.makedirs(f"./results/{date}/{args.experiment}/{trail}")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(str(args))
    # Create network
    net = create_net(args) 
    param = sample_param(net.gt, args.ub, args.dynamics, args.prob_theta0)
    
    true_x, x0, t = get_steady_state(args, net, net.gt)
    logger.info(f'|dxdt| {np.mean(np.abs(net.dxdt(true_x, 0, net.gt)))}')
     
    ntype = args.noise_type
    noise_level = args.noise_level
    y, _ = net.apply_noise(ntype, noise_level, true_x) 
    ode_system = ["full", "mfa", "mfap"] 
    state_err = []
    state_err_exact = []
    runtime = []
    param_err = []
    improvement = []
    initial_param = sample_param(net.gt, args.ub, args.dynamics, args.prob_theta0)
    for i in range(len(ode_system)):
        ode = ode_system[i]
        logger.info(f"\nODE System: {ode}\n")
        args.ss_solver = ode
        se, pe, se_imp, rt, param = learn_dynamics(args, net, logger, initial_param, y, true_x, x0, t)
       
        state_err.append(se) 
        xhat, x0, t = get_steady_state(args, net, param)
        se_exact = mape(xhat, true_x)
        state_err_exact.append(se_exact)

        runtime.append(rt)
        param_err.append(pe)
        improvement.append(se_imp) 
    res = {
        "ode": ode_system,
        "runtime": runtime,
        "state err": state_err, # |(xhat - y)/y|, where xhat is computed using the user-defined system (full, mfa, or mfa plus)
        "state err exact": state_err_exact, # |(xhat - y)/y|, where xhat is computed using the learned parameter for the full ODE system
        "param err": param_err,
        "improvement": improvement
    }
     
    pd.DataFrame(res).to_csv(f'./results/{date}/{args.experiment}/{trail}/result{args.seed}_nlevel{args.noise_level}_p0{args.prob_theta0}.csv', index=None )
  
def perturb(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"

    if args.topology == 'real':
        trail = f"{args.dynamics}_{args.data}"
    elif args.topology == 'er':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.k}_{args.nu}"
    elif args.topology == 'sf':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.gamma}_{args.nu}"
    log_path = f"./results/{date}/{args.experiment}/{trail}/result_{args.seed}.log"
    if not os.path.exists(f"./results/{date}/{args.experiment}/{trail}"):
        os.makedirs(f"./results/{date}/{args.experiment}/{trail}")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(str(args))
    # Create network
    net = create_net(args)
    
    true_x, x0, t = get_steady_state(args, net, net.gt)
    
    ntype = args.noise_type
    noise_level = args.noise_level
    y, _ = net.apply_noise(ntype, noise_level, true_x)
    
    perturb_types = [0, 2, 4]
    perturb_ratios = [0.1, 0.2, 0.3, 0.4]

    ptype_names = {
        0: "Random Rewiring",
        1: "Node Addition",
        2: "Node Removal",
        3: "Greedy Rewiring",
        4: "Edge Removal"
    }

    state_err = []
    state_err_exact = []
    traj_err = []
    ptype = []
    pratio = []
    
    if args.dynamics == "eco":
        param_label = ["B", "K", "C", "D", "E", "H"]
    elif args.dynamics == "gene":
        param_label = ["B", "f", "h"]
    else:
        param_label = ["beta"]
    all_params = []
    for i in range(len(param_label)):
        all_params.append([])

    # theparameters are obtained via mfa method 
    if args.data == "net6":
        param = np.array([1.5159324670010468, 4.069138486138794, 3.897222484468571, 5.3666214003458945, 2.758577078054419, 1.3772848546282732])
    elif args.data == "net8":
        param = np.array([1.0012194861453557e-05, 2.8366505330572043, 2.799661504685604, 5.3324871009933945, 1.5913979346147575, 0.051035283505790965])
    elif args.data == "tya":
        param = np.array([0.9687629722634417, 1.0446006568786814, 2.798985895966437])
    elif args.data == "mec":
        param = np.array([1.0810055643616125, 0.9833610169808821, 2.999999749845122])
    elif args.data == "infect-dublin": 
        param = np.array([0.422110553])  
    else:  
        param = np.array([0.432160804]) 

    true_A = net.A.copy()
    ode = "mfa"
    logger.info(f"\nODE System: {ode}\n")
    args.ss_solver = ode

    for perturb_type in perturb_types:
        for perturb_ratio in perturb_ratios: 
            new_A = perturb_net(true_A, perturb_ratio, perturb_type)[0]
            net.setTopology(new_A)

            true_x, x0, t = get_steady_state(args, net, net.gt)
            xhat, x0, t = get_steady_state(args, net, param)

            true_traj = odeint(net.dxdt, y0=x0, t=t, args=(net.unload(net.gt), )) 
            pred_traj = odeint(net.dxdt, y0=x0, t=t, args=(net.unload(param), )) 
            
            traj_err.append(mape(pred_traj, true_traj))
            state_err_exact.append(mape(xhat, true_x, eps_state=1e-7))
            xhat_mfa = net.mfa(param, xhat)

            state_err.append(mape(xhat_mfa, true_x, eps_state=1e-7))
            ptype.append(ptype_names[perturb_type])
            pratio.append(perturb_ratio)
            
            for i in range(len(param_label)):
                all_params[i].append(param[i]) 
            pd.DataFrame({
                f"true_x_{perturb_type}_{perturb_ratio}": true_x,
                f"xhat_{perturb_type}_{perturb_ratio}": xhat,
                f"xhat_mfa_{perturb_type}_{perturb_ratio}": xhat_mfa 
            }).to_csv(f'./results/{date}/{args.experiment}/{trail}/type{perturb_type}_ratio{perturb_ratio}_{ode}_seed{args.seed}.csv', index=None )

    res = {
        f"state err exact {trail}": state_err_exact,
        f"state err {trail}": state_err,
        f"trajectory err {trail}": traj_err,
        f"type {trail}": ptype,
        f"ratio {trail}": pratio
    }
    for i in range(len(param_label)):
        res[param_label[i]] = all_params[i]
    
    pd.DataFrame(res).to_csv(f'./results/{date}/{args.experiment}/{trail}/result_ode{ode}_seed{args.seed}.csv', index=None )
    
    
def plot_landscape(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"

    if args.topology == 'real':
        trail = f"{args.dynamics}_{args.data}"
    elif args.topology == 'er':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.k}_{args.nu}"
    elif args.topology == 'sf':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.gamma}_{args.nu}"
    
    if not os.path.exists(f"./results/{date}/{args.experiment}"):
        os.makedirs(f"./results/{date}/{args.experiment}")

    net = create_net(args)
    true_x, x0, t = get_steady_state(args, net, net.gt)
    ntype = args.noise_type
    noise_level = args.noise_level
    y, _ = net.apply_noise(ntype, noise_level, true_x)
    
    alpha1 = np.linspace(0.1, 10, 500)
    alpha2 = np.linspace(0.1, 10, 500) 
    loss_noiseless = []
    loss = []
    A1 = []
    A2 = []
    xs = []
    system = args.ss_solver
    for a1 in alpha1:
        for a2 in alpha2:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            print(a1, a2)
            A1.append(a1)
            A2.append(a2) 
            param = net.gt.copy()
            param[1] = a1 
            param[2] = a2  
            if system == "full":
                xhat = net.ode(param, y)
                dxdt_ = np.mean(np.abs(net.dxdt(xhat, 0, param, )))
            else:
                xhat = net.mfa(param, y)
                dxdt_ = np.mean(np.abs(net.dxdt_mfa(xhat, 0, param, net.degree))) 
            if dxdt_ > 1e-5:
                print("not converge")
                loss.append(1500)
                xs.append(np.nan)
              
            else:
                error = np.mean(np.square(xhat - y))
                error_noiseless = np.mean(np.square(xhat - true_x))
                if error > 1500:
                    loss.append(1500)
                else:
                    loss.append(error)
                if error_noiseless > 1500:
                    loss_noiseless.append(1500)
                else:
                    loss_noiseless.append(error_noiseless)
    pd.DataFrame({
        "K": A1,
        "C": A2,
        f"{trail}_loss_{system}_nlevel0.13": loss 
    }).to_csv(f"./results/{date}/{args.experiment}/{trail}_loss_{system}_nlevel0.13_seed{args.seed}.csv", index=None)
    pd.DataFrame({
        "K": A1,
        "C": A2,
        f"{trail}_loss_{system}_nlevel0": loss_noiseless
    }).to_csv(f"./results/{date}/{args.experiment}/{trail}_loss_{system}_nlevel0_seed{args.seed}.csv", index=None)
     
def adv_recover(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"

    if args.topology == 'real':
        trail = f"{args.dynamics}_{args.data}"
    elif args.topology == 'er':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.k}_{args.nu}"
    elif args.topology == 'sf':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.gamma}_{args.nu}"
    log_path = f"./results/{date}/{args.experiment}/{trail}/{args.ss_solver}_result_{args.seed}.log"
    if not os.path.exists(f"./results/{date}/{args.experiment}/{trail}"):
        os.makedirs(f"./results/{date}/{args.experiment}/{trail}")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(str(args))
    net = create_net(args)
     
    logger.info('\n Re-learn using covered states\n')
    true_x, x0, t = get_steady_state(args, net, net.gt)
    
    ntype = args.noise_type
    noise_level = args.noise_level
    y, idx = net.apply_noise(ntype, noise_level, true_x)

    percent_err = np.mean(np.abs((y-true_x) / true_x)) 
    logger.info(f'Observed percent error {percent_err}')
 
    param = sample_param(net.gt, args.ub, args.dynamics)
    se, pe, se_imp, rt, learned_param = learn_dynamics(args, net, logger, param, y, true_x, x0, t)
    np.savetxt(f'./results/{date}/{args.experiment}/{trail}/{args.ss_solver}_1stparam{args.seed}.txt', learned_param)
    xhat, x0, t = get_steady_state(args, net, learned_param)
    pd.DataFrame({
        f"true_{trail}": true_x,
        f"observed_{trail}": y,
        f"xhat_{trail}": xhat
    }).to_csv(f'./results/{date}/{args.experiment}/{trail}/{args.ss_solver}_1st_result{args.seed}.csv', index=None )

     
    adv_idx = np.where(np.abs((xhat - y) / (y+1e-8)) > 1)[0]
    logger.info(f'adv_idx {[str(i) for i in adv_idx]}')
    logger.info(f'idx {[str(i) for i in idx]}')  
    rest_idx = []
    for i in range(net.N):
        if i not in adv_idx:
            rest_idx.append(i) 
    net.obs_idx = rest_idx
    observation = y.copy()
    observation[adv_idx] = xhat[adv_idx]
    args.ss_solver = "mfap"
    se, pe, se_imp, rt, param = learn_dynamics(args, net, logger, learned_param, observation, true_x, x0, t)

    xhat, x0, t = get_steady_state(args, net, param)
    pd.DataFrame({
        f"xhat_{trail}": xhat
    }).to_csv(f'./results/{date}/{args.experiment}/{trail}/{args.ss_solver}_result{args.seed}.csv', index=None )
    np.savetxt(f'./results/{date}/{args.experiment}/{trail}/{args.ss_solver}_param{args.seed}.txt', param)
    return 

def compute_steady_state(args):
    '''
    Learn dynamics of the given network.
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"

    if args.topology == 'real':
        trail = f"{args.dynamics}_{args.data}"
    elif args.topology == 'er':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.k}_{args.nu}"
    elif args.topology == 'sf':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.gamma}_{args.nu}"
    log_path = f"./results/{date}/{args.experiment}/{trail}/result_{args.seed}_{args.ss_solver}.log"
    if not os.path.exists(f"./results/{date}/{args.experiment}/{trail}"):
        os.makedirs(f"./results/{date}/{args.experiment}/{trail}")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(str(args))
    # Create network
    args.k = int(0.2 * args.n)
    net = create_net(args)
    logger.info(f'number of unique degrees {len(net.deg_unique)}')
     
    if args.dynamics == "epi":
        x0 = np.full(len(net.degree), .5)
    else:
        x0 = np.full(len(net.degree), 6.) 
    t = np.linspace(0,10,1000)
    t1 = time.time()
    if args.ss_solver == "full":
        true_x = net.solve_ode(net.dxdt, x0, t, net.unload(net.gt))
        logger.info(f"<|dxdt|> {np.mean(np.abs(net.dxdt(true_x, 0, net.gt)))}")

    else:
        true_x = net.mfa(net.gt, x0)
        logger.info(f"<|dxdt|> {np.mean(np.abs(net.dxdt_mfa(true_x, 0, net.gt, net.degree)))}")

    t2 = time.time() 
    logger.info(f"ODE System {args.ss_solver}")
    logger.info(f"Runtime: {t2-t1}")
    logger.info(f"Steady State {np.mean(true_x)}")
    
 
def brn_vs_true(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    today = datetime.datetime.now()
    date = f"{calendar.month_name[today.month][:3]}{today.day}"

    if args.topology == 'real':
        trail = f"{args.dynamics}_{args.data}"
    elif args.topology == 'er':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.k}_{args.nu}"
    elif args.topology == 'sf':
        trail = f"{args.dynamics}_{args.topology}_{args.n}_{args.gamma}_{args.nu}"
    log_path = f"./results/{date}/{args.experiment}/{trail}.log"
    if not os.path.exists(f"./results/{date}/{args.experiment}"):
        os.makedirs(f"./results/{date}/{args.experiment}")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(str(args))
    # Create network
    net = create_net(args)
    
    param = sample_param(net.gt, args.ub, args.dynamics, args.prob_theta0)
    
    true_x, x0, t = get_steady_state(args, net, net.gt)

    param = net.gt 
    res = odeint(net.dxdt, y0=x0, t=t, args=(net.unload(param),  ))[-1]
    max_iter = 100
    itr = 0
    while np.mean(np.abs(net.dxdt(res, t, net.unload(param)  ))) > 1e-13: 
        if itr == max_iter:
            break
        res = odeint(net.dxdt, y0=res, t=t, args=(net.unload(param),  ))[-1]
        itr += 1
    logger.info(f'|dxdt| {np.mean(np.abs(net.dxdt(res, 0, net.gt)))}')

    true_x = res.copy()
    pd.DataFrame({
        f"degree_{trail}": net.degree, 
        f"state_{trail}": true_x
    }).to_csv(f"./results/{date}/{args.experiment}/{trail}.csv", index=None)

    degree = np.linspace(min(net.degree), max(net.degree))
    state = []
    for beta in degree:
        res = odeint(net.dxdt_xeff, y0=100, t=t, args=(param, beta))[-1]
        while np.mean(np.abs(net.dxdt_xeff(res, t, net.unload(param), beta  ))) > 1e-13: 
            if itr == max_iter:
                break
            res = odeint(net.dxdt_xeff, y0=res, t=t, args=(net.unload(param), beta ))[-1]
            itr += 1    
        state.append(res[0])
    pd.DataFrame({
        f"brn_degree_{trail}": degree,
        f"brn_state_{trail}": state
    }).to_csv(f"./results/{date}/{args.experiment}/{trail}_brn.csv", index=None)
 
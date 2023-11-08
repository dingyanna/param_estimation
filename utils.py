import numpy as np
import networkx as nx
import logging
import random  
 
def load_real_net(data):
    '''
    Read adjacency matrix from data files.

    Assumption:
    Each line in data file takes the form 'i,j,{'weight': x}', 
    where (i,j) is an edge and x is the weight of the edge
    '''
    print(data) 
    dlm = ','
    edges = np.genfromtxt(data, delimiter=dlm)
 
    if edges.shape[1] == 3:
        edges = edges[:, :2]
    G = nx.Graph()
    G.add_edges_from(edges)
    A = nx.to_numpy_array(G, nodelist=np.unique(edges))
    print('[number of edges]', len(edges))
    print('[number of nodes]', len(G))
    return A

def read_param(args): 
    if args.dynamics == 'gene': 
        gt = np.array([1,1,2])
    elif args.dynamics == 'eco' and args.topology != 'brn': 
        gt = np.array([0.1, 5, 1, 5, 0.9, 0.1])
    elif args.dynamics == 'eco' and args.topology == 'brn':  
        gt = np.array([0.1,5,1,5,1])
    else: 
        gt = np.array([0.5])
    gt = np.array(gt, dtype=np.float64)
    return gt 

def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger

def greedy_rewire(G, k):
    np.random.seed(0)
    N = len(G)
    count = 0
    inds = list(range(N))
    while count < k:
        node = random.sample(range(N), 1)[0]
        node_js = [j for j in G.neighbors(node) if G.degree(j) > 1]
        if len(node_js) == 0:
            continue
        node_j = random.sample(node_js, 1)[0]
        probs = []
        neighbor = []
        for j in G.neighbors(node_j):
            if G.has_edge(node, j):
                continue
            neighbor.append(j)
            probs.append(G.degree(j))
        if len(probs) == 0:
            #print("haven't found any accepted nodes")
            continue 
        probs = np.divide(probs, np.sum(probs))
        m = random.choices(neighbor, weights=probs, k=1)[0]
        G.remove_edge(node, node_j)
        G.add_edge(node, m)
        count += 1
        
    return G, inds

def perturb_net(A, perturb_ratio, perturb_type, logger=None): 
    
    np.random.seed(0)
    random.seed(0)
    edges = []
    non_edges = []
    N = len(A)
    for i in range(N):
        for j in range(i+1, N):
            if A[i,j] != 0:
                edges.append((i,j))
            else:
                non_edges.append((i,j))
    G = nx.Graph()
    G.add_edges_from(edges)
    if perturb_type == 0: # rewire
        k = int(perturb_ratio * len(edges))
        to_remove = random.sample(range(len(edges)), k)
        to_add = random.sample(range(len(non_edges)), k)
        for i in range(k):
            G.remove_edge(edges[to_remove[i]][0], edges[to_remove[i]][1])
            G.add_edge(non_edges[to_add[i]][0], non_edges[to_add[i]][1])
        inds = list(range(N))
    elif perturb_type == 1: # node addition
        k = int(perturb_ratio * N)
        m = 4
        to_add = []
        for i in range(N,N+k):
            to_be_attached = random.sample(range(N), m)
            for j in range(m):
                to_add.append((i, to_be_attached[j]))
        G.add_edges_from(to_add)
        inds = None
    elif perturb_type == 2: # node removal 
        k = int(perturb_ratio * N)
        to_remove = random.sample(range(N), k)
        G.remove_nodes_from(to_remove)
        inds = []
        for i in range(N):
            if i not in to_remove:
                inds.append(i)  
    elif perturb_type == 3: # greedy rewiring 
        k = int(perturb_ratio * len(edges))
        G, inds = greedy_rewire(G, k)  
    elif perturb_type == 4: # edge removal
        k = int(perturb_ratio * len(edges))
        to_remove = random.sample(range(len(edges)), k)
        for i in to_remove:
            G.remove_edge(edges[i][0], edges[i][1])
        inds = list(range(N))            
    A = nx.to_numpy_array(G, nodelist=list(G.nodes)) 
    if logger is not None:
        logger.info('\nAfter Perturbation')
        logger.info(f'[# nodes] {len(A)}')
        logger.info(f'[# edges] {A.sum() / 2} \n')
    else:
        print('\nAfter Perturbation')
        print('[# nodes]', len(A))
        print('[# edges]', A.sum() / 2, '\n')
    return A, inds
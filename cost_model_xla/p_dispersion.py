import numpy as np
import random
import cvxopt
import os
from scipy.sparse import csr_matrix, eye as speye, vstack
from tqdm import tqdm, trange

from multiprocessing import Pool

MUL_DELTA = 10

# reference: https://stackoverflow.com/a/35566620
def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = cvxopt.spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def p_dispersion_lp(G, k, eps=0.454):
    num_nodes = G.shape[0]
    diameter = np.max(G)
    min_dist = np.min(G[np.nonzero(G)])
    delta = diameter / MUL_DELTA
    dists = [min_dist + delta * i for i in range(MUL_DELTA)]
    print("N = {}".format(num_nodes))
    variables = [[None]*len(dists)] * num_nodes
    print("Creating vector c.")
    c = cvxopt.matrix([-r for r in dists] * num_nodes)
    print("Building 1st constraint matrix.")
    G1_vals = []
    G1_is = []
    G1_js = []
    for i in range(num_nodes):
        for j in range(len(dists)):
            G1_vals.append(1)
            G1_is.append(0)
            G1_js.append(i*len(dists) + j)
    G1 = csr_matrix((G1_vals, (G1_is, G1_js)))
    G1_h = np.array([k])
    print("2nd constraint.")
    G2_vals = []
    G2_is = []
    G2_js = []
    for i in trange(num_nodes):
        for j in trange(len(dists)):
            for u in trange(num_nodes):
                dist_u_i = G[i,u]
                r = dists[j]
                if dist_u_i < r/2:
                    G2_vals.append(1)
                    G2_is.append(u)
                    G2_js.append(i*len(dists) + j)
    G2 = csr_matrix((G2_vals, (G2_is, G2_js)))
    G2_h = np.ones((num_nodes,1))
    print("X range constraint.")
    G3 = speye(num_nodes*len(dists))
    G3_h = np.ones((num_nodes*len(dists), 1))
    G4 = -speye(num_nodes*len(dists))
    G4_h = np.zeros((num_nodes*len(dists), 1))
    
    G_concated = vstack([G1, G2, G3, G4])
    h_concated = np.vstack((G1_h, G2_h, G3_h, G4_h))

    G_cvx = scipy_sparse_to_spmatrix(G_concated)
    h_cvx = cvxopt.matrix(h_concated)
    print("Start solving...")
    sol = cvxopt.solvers.lp(c, G_cvx, h_cvx, solver="glpk")
    print("Solution obtained.")
    solution = np.array(sol["x"]).reshape((num_nodes, len(dists)))

    # rounding
    print("Start rounding.")
    while True:
        S = set()
        for i in trange(num_nodes):
            for j in range(len(dists)):
                x_i_r = solution[i,j]
                r = dists[j]
                add_prob = (1-eps)*(1-np.e**(-x_i_r))
                if random.random() < add_prob:
                    should_break = False
                    for (i_, r_) in S:
                        if r < r_ and G[i][i_] < r_/2:
                            should_break = True
                            break
                    if should_break:
                        continue
                    S.add((i, r))
        if len(S) <= k:
            # break
            indices = list(set([i for (i, r) in S]))
            yield indices

def sum_min_distance(G, A):
    dist = 0
    min_j = {}
    for i in A:
        min_dist = float("inf")
        min_j_for_i = -1
        for j in A:
            if j == i:
                continue
            dist = G[i,j]
            if dist < min_dist:
                min_dist = dist
                min_j_for_i = j
        dist += min_dist
        min_j[i] = min_j_for_i
    return dist, min_j

def sum_min_distance_edit(G, last_dist, min_j, A, idx_rm, idx_add):
    new_min_j = min_j.copy()
    last_dist -= G[idx_rm, min_j[idx_rm]]
    new_min_j.pop(idx_rm)
    for index in A:
        if index != idx_rm and min_j[index] == idx_rm:
            last_dist -= G[index, idx_rm]
            min_j_for_index = -1
            min_dist_for_index = float("inf")
            for A_index in A:
                if A_index != idx_rm and A_index != index:
                    if G[index, A_index] < min_dist_for_index:
                        min_dist_for_index = G[index, A_index]
                        min_j_for_index = A_index
            new_min_j[index] = min_j_for_index
            last_dist += min_dist_for_index
    min_dist_for_add = float("inf")
    min_j_for_add = -1
    for index in A:
        if index != idx_rm:
            orig_min_dist = G[index, new_min_j[index]]
            if G[index, idx_add] < orig_min_dist:
                new_min_j[index] = idx_add
                last_dist -= orig_min_dist
                last_dist += G[index, idx_add]
            if G[index, idx_add] < min_dist_for_add:
                min_dist_for_add = G[index, idx_add]
                min_j_for_add = index
    last_dist += min_dist_for_add
    new_min_j[idx_add] = min_j_for_add
    return last_dist, new_min_j


def p_dispersion_local_search(G, k, sample_ratio = 1, patience = 3, l=None, tqdm_position=0):
    num_nodes = G.shape[0]
    A = set(random.sample(list(range(num_nodes)), k=k))
    if l is None:
        l = int(np.ceil(k * np.log(k)))
    last_dist, min_j = sum_min_distance(G, A)
    max_dist = last_dist
    sample_k = int(np.ceil(sample_ratio * num_nodes))
    # print("Using {} samples in each iteration, ratio: {}".format(sample_k, sample_ratio))
    opt_counter = 0
    tqdm_iterator = trange(l, position=tqdm_position, desc="worker {}: ".format(tqdm_position), leave=False)
    for i in tqdm_iterator:
        max_new_min_j = None
        max_idx_rm = -1
        max_idx_add = -1
        for out_index in random.sample(range(num_nodes), sample_k):
            if out_index not in A:
                for a_index in A:
                    new_dist, new_min_j = sum_min_distance_edit(G, last_dist, min_j, A, a_index, out_index)
                    if new_dist > max_dist:
                        max_dist = new_dist
                        max_idx_rm = a_index
                        max_idx_add = out_index
                        max_new_min_j = new_min_j
        if max_idx_rm == -1:
            opt_counter += 1
            if opt_counter >= patience:
                break
        else:
            A.remove(max_idx_rm)
            A.add(max_idx_add)
            last_dist = max_dist
            min_j = max_new_min_j
    tqdm_iterator.close()
    return list(A)

def worker_func(arg):
    max_dist = -float("inf")
    max_min_j = None
    max_a_index = -1
    max_out_index = -1
    for (G, last_dist, last_min_j, A, a_index, out_index) in arg:
        new_dist, new_min_j = sum_min_distance_edit(G, last_dist, last_min_j, A, a_index, out_index)
        if new_dist > max_dist:
            max_dist = new_dist
            max_min_j = new_min_j
            max_a_index = a_index
            max_out_index = out_index
    return (max_dist, max_min_j, max_a_index, max_out_index)

def parallel_p_dispersion_local_search(G, k, sample_ratio = 1, patience = 3, l=None):
    num_nodes = G.shape[0]
    A = set(random.sample(list(range(num_nodes)), k=k))
    if l is None:
        l = int(np.ceil(k * np.log(k)))
    last_dist, min_j = sum_min_distance(G, A)
    max_dist = last_dist
    sample_k = int(np.ceil(sample_ratio * num_nodes))
    # print("Using {} samples in each iteration, ratio: {}".format(sample_k, sample_ratio))
    opt_counter = 0
    for i in trange(l, desc="iter: ", leave=True):
        max_new_min_j = None
        max_idx_rm = -1
        max_idx_add = -1
        map_args = []
        for out_index in random.sample(range(num_nodes), sample_k):
            if out_index not in A:
                for a_index in A:
                    map_args.append( (
                        G, last_dist, min_j, A, a_index, out_index
                    ) )
        num_cores = min(os.cpu_count(), len(map_args))
        grouped_map_args = []
        chunk_size = int(np.ceil(len(map_args) / num_cores))
        for i in range(num_cores):
            actual_chunk_size = min(chunk_size, len(map_args)-i*chunk_size)
            grouped_map_args.append(map_args[i*chunk_size:i*chunk_size+actual_chunk_size])
        with Pool(num_cores) as p:
            distances = list(tqdm(p.imap_unordered(worker_func, grouped_map_args), total=len(grouped_map_args), desc="inner: ", leave=False))
        for (new_dist, new_min_j, a_index, out_index) in distances:
            if new_dist > max_dist:
                max_dist = new_dist
                max_idx_rm = a_index
                max_idx_add = out_index
                max_new_min_j = new_min_j
        if max_idx_rm == -1:
            opt_counter += 1
            if opt_counter >= patience:
                break
        else:
            A.remove(max_idx_rm)
            A.add(max_idx_add)
            last_dist = max_dist
            min_j = max_new_min_j
    return list(A)
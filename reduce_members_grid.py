import numpy as np
import time
from numba import njit

from numba.core import types
#from numba.typed import Dict
from numba.typed import List

import math
import helper_sca2gri
import ctypes

# from numba import config

# config.DISABLE_JIT = True


@njit
def comp_upper_bound_members_per_cell(n_points_per_cell, neighborhood_onion_x, neighborhood_onion_y, grid_dim):    
    cell_neighborhood_step=np.zeros(grid_dim)
    cell_neighborhood_step[:]=np.inf
    
    upper_bound_members_per_cell=np.zeros(grid_dim, dtype=np.int64)

    consider_for_next_round=List(np.arange(grid_dim[0]*grid_dim[1], dtype=np.int64))

    n_neighborhood_layers=len(neighborhood_onion_x)
    
    for neighborhood_step in np.arange(n_neighborhood_layers, dtype=np.int64):

        considered_grid_cells=consider_for_next_round
        consider_for_next_round=List.empty_list(types.int64)
        
        for grid_cell_1 in considered_grid_cells:

            n_members=n_points_per_cell[grid_cell_1]
            
            
            if n_members==0:
                continue

            grid_cell=helper_sca2gri.i2ii(grid_cell_1, grid_dim)
            n_members_available=n_members-upper_bound_members_per_cell[grid_cell]
            
            
            unattended_neighbors=[]

            for neighbor_idx in range(len(neighborhood_onion_x[neighborhood_step])):
                neighbor=(neighborhood_onion_x[neighborhood_step][neighbor_idx],\
                          neighborhood_onion_y[neighborhood_step][neighbor_idx])
                neighbor_cell=(grid_cell[0]+neighbor[0], grid_cell[1]+neighbor[1])

                if min(neighbor_cell)<0 or neighbor_cell[0]>=grid_dim[0] or neighbor_cell[1]>=grid_dim[1]:
                    continue

                # as we progress in lockstep, there is no way we can reach a previously discovered cell more quickly   
                assert cell_neighborhood_step[neighbor_cell]==np.inf or not neighborhood_step<cell_neighborhood_step[neighbor_cell]

                if neighborhood_step<=cell_neighborhood_step[neighbor_cell]:
                    unattended_neighbors.append(neighbor_cell)

            
            # check if we have enough potential to cover them all            
            if n_members_available>=len(unattended_neighbors):

                for e in unattended_neighbors:
                    cell_neighborhood_step[e]=neighborhood_step
                upper_bound_members_per_cell[grid_cell[0], grid_cell[1]]+=len(unattended_neighbors)

                # only consider cell for next round if the is potential left after this one
                if n_members_available>len(unattended_neighbors):
                    consider_for_next_round.append(helper_sca2gri.ii2i(grid_cell, grid_dim))
                
            # we do not have enough potential to cover them all,
            # so keep all the points we have in the cell
            else:
                upper_bound_members_per_cell[grid_cell]=n_members

    return upper_bound_members_per_cell

@njit
def _dist_func(a, cell_size):
    #return np.linalg.norm(a*np.array(cell_size))
    return ((a[0]*cell_size[0])**2+(a[1]*cell_size[1])**2)**0.5

@njit
def comp_neighborhood_onion_d_limit(cell_size, tau):
    if tau <0:
        d_limit=np.inf
    else:
        d_limit=tau+0.5*_dist_func((1,1), cell_size)

    return d_limit


@njit
def comp_neighborhood_onion(grid_dim, cell_size, tau):

    d_limit=comp_neighborhood_onion_d_limit(cell_size, tau)

    neighborhood_x=List.empty_list(types.int64)
    neighborhood_y=List.empty_list(types.int64)
    neighborhood_dist=List.empty_list(types.float64)
    
    for n_1 in np.arange(grid_dim[0]*grid_dim[1], dtype=np.int64):
        n=helper_sca2gri.i2ii(n_1, grid_dim)
        d=_dist_func(n, cell_size)        
                
        if d<=d_limit:
            neighborhood_x.append(n[0])
            neighborhood_y.append(n[1])
            neighborhood_dist.append(d)
            if n[0]>0:
                neighborhood_x.append(-n[0])
                neighborhood_y.append(n[1])
                neighborhood_dist.append(d)
            if n[1]>0:
                neighborhood_x.append(n[0])
                neighborhood_y.append(-n[1])
                neighborhood_dist.append(d)
            if n[0]>0 and n[1]>0:
                neighborhood_x.append(-n[0])
                neighborhood_y.append(-n[1])
                neighborhood_dist.append(d)
    

    neighborhood_onion_x=List()
    neighborhood_onion_y=List()
    
    latest_dist=-1.
    
    a = np.empty(len(neighborhood_dist), dtype=type(neighborhood_dist[0]))
    for i,v in enumerate(neighborhood_dist):
        a[i] = v

    idcs_sorted=np.argsort(a)
    
    for idx in idcs_sorted:
        
        d=neighborhood_dist[idx]
        if d != latest_dist:            
            
            latest_dist=d
            neighborhood_onion_x.append(List.empty_list(types.int64))
            neighborhood_onion_y.append(List.empty_list(types.int64))
        

        neighborhood_onion_x[-1].append(neighborhood_x[idx])
        neighborhood_onion_y[-1].append(neighborhood_y[idx])

    return neighborhood_onion_x, neighborhood_onion_y


@njit
def random_sample_set(n,k):

    seen = {0}
    seen.clear()
    index = np.empty(k, dtype=np.int64)
    for i in range(k):
        j = np.random.randint(0, n)
        while j in seen:
            j = np.random.randint(0, n)
        seen.add(j)
        index[i] = j
    return index

'''
pick the members to keep according to the upper bound
'''
@njit
def select_embedding_idcs(n_points_per_cell, points_per_cell, upper_bound_members_per_cell, grid_dim):

    max_allowed_collision_prob=0.01

    np.random.seed(0)

    n_cells=grid_dim[0]*grid_dim[1]

    n_occupied_cells=0
    embedding_idcs=List.empty_list(types.int64)
    for grid_cell_1 in range(n_cells):

        grid_cell=helper_sca2gri.i2ii(grid_cell_1, grid_dim)
        
        n_members_keep=upper_bound_members_per_cell[grid_cell]        

        if n_members_keep>0:
            n_occupied_cells+=1
        else:
            continue


        # simple: just pick the first ones        
        # embedding_idcs+=members[:n_members_keep]

        # random selection
        n_candidates=np.int64(min(n_points_per_cell[grid_cell_1], n_cells))
        if probability_of_duplicate_approx(n_candidates, n_members_keep)\
           < max_allowed_collision_prob:
            selected_idcs=random_sample_set(n_candidates, n_members_keep)

            for i in selected_idcs:
                embedding_idcs.append(points_per_cell[n_cells*grid_cell_1+i])
            
        else:
            candidate_idcs=np.arange(n_candidates, dtype=np.int64)

            selected_idcs=np.random.choice(candidate_idcs, size=n_members_keep, replace=False)

            for i in selected_idcs:
                embedding_idcs.append(points_per_cell[n_cells*grid_cell_1+i])

                

    embedding_idcs=sorted(embedding_idcs)

    return embedding_idcs, n_occupied_cells


@njit
def get_embedding_idcs(n_points_per_cell, points_per_cell, upper_bound_members_per_cell, grid_dim):

    n_cells=grid_dim[0]*grid_dim[1]

    n_occupied_cells=0

    embedding_idcs=np.empty(np.sum(upper_bound_members_per_cell), dtype=np.int64)

    embedding_idcs_off=0
    
    for grid_cell_1 in range(n_cells):
        
        grid_cell=helper_sca2gri.i2ii(grid_cell_1, grid_dim)

        n_members_keep=upper_bound_members_per_cell[grid_cell]        

        if n_members_keep>0:
            n_occupied_cells+=1
        else:
            continue

        embedding_idcs[embedding_idcs_off:embedding_idcs_off+n_members_keep]=\
            points_per_cell[n_cells*grid_cell_1:n_cells*grid_cell_1+n_members_keep]
        
        embedding_idcs_off+=n_members_keep


    assert embedding_idcs_off==len(embedding_idcs)

    return embedding_idcs, n_occupied_cells


def comp(rg, embedding, cell_size, embedding_box, grid_dim, tau):

    timings={}

    start_time=time.time()    

    n_points_per_cell, points_per_cell=rg(cell_size, embedding_box, grid_dim)
    timings['reduce_grid']=time.time()-start_time

    start_time=time.time()

    if tau is None:
        tau=-1.
        

    neighborhood_onion_x, neighborhood_onion_y=comp_neighborhood_onion(grid_dim, cell_size, tau)        
        
    timings['reduce_onion']=time.time()-start_time

    start_time=time.time()
    upper_bound_members_per_cell=comp_upper_bound_members_per_cell(n_points_per_cell, neighborhood_onion_x, neighborhood_onion_y, grid_dim)
    timings['reduce_bound']=time.time()-start_time
    

    embedding_idcs, n_occupied_cells=\
        get_embedding_idcs(n_points_per_cell, points_per_cell, upper_bound_members_per_cell, grid_dim)
    

    return embedding_idcs, n_occupied_cells, (n_points_per_cell, points_per_cell), upper_bound_members_per_cell, timings

    
import numpy as np

if __name__ == '__main__':

    o_x, o_y=comp_neighborhood_onion((20,20), (1., 0.5), 1.)
    print(o_x)
    print(o_y)

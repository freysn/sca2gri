from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances

import sys
import os

from pathlib import Path
HOME=str(Path.home())

import time

import numpy as np
import helper_sca2gri

def assign_cost(embedding, embedding_idcs, grid_cell_centers, tau, n_occupied_cells, upper_bound_members_per_cell, grid_dim, int_fac):

    timings={}

    start_time=time.time()

    m=len(embedding_idcs)
    n=len(grid_cell_centers)

    cost_virtual=None
    if tau is not None:
        tau_int=tau*int_fac

        '''
        conservative choice of n_total that potentially permits an assignment with solely
        virtual assignments
        '''
        n_total=m+n

        '''
        occupied cells can be subtracted as for them it can be assumed that there is a
        real assignment (to their respective cells)
        '''
        n_total-=n_occupied_cells

        cost_virtual=tau_int+1
        
    else:
        tau_int=np.iinfo(np.int64).max
        n_total=max(m,n)
        cost_virtual=tau_int


    cost=np.full((n_total, n_total), cost_virtual, dtype=int)

    cost_real=pairwise_distances(embedding[embedding_idcs], Y=grid_cell_centers, metric='euclidean')

    cost[:len(embedding_idcs), :len(grid_cell_centers)]=cost_real*int_fac


    if tau is not None:
        idcs_off=0
        for grid_cell_1 in range(n):

            grid_cell=helper_sca2gri.i2ii(grid_cell_1, grid_dim)
            n_members_keep=upper_bound_members_per_cell[grid_cell]            

            if n_members_keep==0:
                continue

            idcs=embedding_idcs[idcs_off:idcs_off+n_members_keep]
            
            cost[idcs_off:idcs_off+n_members_keep, grid_cell_1]=np.minimum(cost[idcs_off:idcs_off+n_members_keep, grid_cell_1], tau_int)
            idcs_off+=n_members_keep

        assert idcs_off==len(embedding_idcs), f'{idcs_off} vs {len(embedding_idcs)}'


    return cost

def assign_LA(cost, embedding_idcs, n, *, int_fac=100000, do_total_cost=False):

    timings={}
    start_time_linear_sum_assignment=time.time()

    row_ind, col_ind = linear_sum_assignment(cost)

    timings['assign_linas']=time.time()-start_time_linear_sum_assignment

    m=len(embedding_idcs)

    real_assignments=[(embedding_idcs[i],j) for i,j in enumerate(col_ind[:m]) if j<n]

    total_cost=None

    if do_total_cost:
        total_cost=0

        for i,j in enumerate(col_ind[:m]):
            if j<n:
                total_cost+=cost[i,j]

        total_cost/=int_fac

    return real_assignments, total_cost, timings

def comp(embedding, embedding_idcs, grid_cell_centers, tau, n_occupied_cells, upper_bound_members_per_cell, grid_dim):

    int_fac=100000        

    timings={}
    start_time_assign=time.time()
    
    
    cost=assign_cost(embedding, embedding_idcs, grid_cell_centers, tau, n_occupied_cells, upper_bound_members_per_cell, grid_dim, int_fac)

    timings['assign_cost']=time.time()-start_time_assign
    
    real_assignments, _, timings_assign_LA=assign_LA(cost, embedding_idcs, len(grid_cell_centers), int_fac=int_fac)
    
    timings = timings | timings_assign_LA
    
    embedding_idcs, grid_idcs=zip(*real_assignments)    

    return embedding_idcs, grid_cell_centers[np.array(grid_idcs, dtype=int)], timings

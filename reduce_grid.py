import numpy as np
import time
from numba import njit

from numba.core import types
from numba.typed import List

import math
import helper_sca2gri

import platform

if platform.system()=="Darwin":
    import reduce_grid_metal

# from numba import config

# config.DISABLE_JIT = True

@njit
def grid_cpu(embedding, cell_size, embedding_box, grid_dim):

    grid_idcs=((embedding-np.array(embedding_box[0:2]))/np.array(cell_size))


    n_cells=grid_dim[0]*grid_dim[1]
    # ctypes.c_uint32
    n_points_per_cell=np.zeros(n_cells, dtype=np.uint32)
    # ctypes.c_uint32
    points_per_cell=np.zeros(n_cells*n_cells, dtype=np.uint32)

    for i, grid_idx_f in enumerate(grid_idcs):

        grid_idx=(int(math.floor(grid_idx_f[0])),int(math.floor(grid_idx_f[1])))
        
        if min(grid_idx)>0 and grid_idx[0]<grid_dim[0] and grid_idx[1]<grid_dim[1]:
            
            grid_idx_1=helper_sca2gri.ii2i(grid_idx, grid_dim)

            local_idx=n_points_per_cell[grid_idx_1]


            if local_idx<n_cells:
                points_per_cell[n_cells*grid_idx_1+local_idx]=i

            n_points_per_cell[grid_idx_1]+=1

    
    return n_points_per_cell, points_per_cell


class ReduceGrid:

    def __init__(self, embedding, *, do_metal=True, n_cells_max=128*256):

        self.do_metal=do_metal
        if not self.do_metal:
            self.embedding=embedding
        else:
            self.instance=reduce_grid_metal.load()

            self.n_points=len(embedding)

            self.n_cells_max=n_cells_max        

            self.buf_embedding=reduce_grid_metal.alloc_set_embedding(self.instance, self.n_points, embedding.flatten())

            self.buf_embedding_box_off__cell_size, self.buf_grid_dim=reduce_grid_metal.alloc_params_float_int(self.instance)

            self.buf_n_points_per_cell, self.buf_points_per_cell=reduce_grid_metal.alloc_grid(self.instance, n_cells_max)

    def __call__(self, cell_size, embedding_box, grid_dim):
        if self.do_metal:
            return self.grid_metal(cell_size, embedding_box, grid_dim)
        else:
            return grid_cpu(self.embedding, cell_size, embedding_box, grid_dim)
        
    def grid_metal(self, cell_size, embedding_box, grid_dim):

        gpu_start = time.time()
        assert grid_dim[0]*grid_dim[1] <= self.n_cells_max, f'grid_dim[0]={grid_dim[0]}*grid_dim[1]={grid_dim[1]}<= self.n_cells_max={self.n_cells_max}'
        reduce_grid_metal.set_embedding_box__cell_size(self.buf_embedding_box_off__cell_size, embedding_box, cell_size)
        reduce_grid_metal.set_grid_dim(self.buf_grid_dim, grid_dim)

        
        # just for testing, can actually leave uninitialized
        # buf_points_per_cell_INIT=2**32-1
        # mmetal.upload(buf_points_per_cell, buf_points_per_cell['n']*[buf_points_per_cell_INIT])

        

        reduce_grid_metal.run(self.instance, self.n_points,\
                              self.buf_n_points_per_cell,\
                              self.buf_points_per_cell,\
                              self.buf_embedding,\
                              self.buf_embedding_box_off__cell_size,\
                              self.buf_grid_dim)

        #gpu_end = time.time()

        #print(f'METAL runtime {gpu_end-gpu_start} {self.buf_embedding['n']}')
        
        n_points_per_cell=reduce_grid_metal.mmetal.download(self.buf_n_points_per_cell)
        points_per_cell=reduce_grid_metal.mmetal.download(self.buf_points_per_cell)

        #gpu_end = time.time()
        #print(f'METAL runtime download {gpu_end-gpu_start} {self.buf_embedding['n']}')

        np_n_points_per_cell=np.frombuffer(n_points_per_cell, dtype=np.uint32)
        np_points_per_cell=np.frombuffer(points_per_cell, dtype=np.uint32)
            
        # gpu_end = time.time()
        # print(f'METAL runtime convert {gpu_end-gpu_start} {self.buf_embedding['n']}')
        
        return np_n_points_per_cell, np_points_per_cell
            

        
        

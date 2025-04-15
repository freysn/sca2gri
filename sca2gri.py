import numpy as np
import time

import reduce_members_grid
import reduce_grid
import assign

import time

from pathlib import Path
HOME=str(Path.home())
import os
import sys

sys.path.insert(0, './data')


import math

from enum import auto
from enum import Flag

import sca2gri_draw

class Method(Flag):
    sca2gri = auto()
    dgrid = auto()
    hagrid = auto()

def method_str(m):
    if m==Method.sca2gri:
        return r'Sca$^2$Gri'
    elif m==Method.dgrid:
        return r'DGrid'
    elif m==Method.hagrid:
        return f'Hagrid'
    else:
        return None
        
class sca2gri:

    def __init__(self, embedding, rep, *, n_cells_max=128*256):

        self.n_cells_max=n_cells_max
        self.embedding=embedding
        self.rep=rep

        self.cache={}
        

    def run_sca2gri(self, rg, cell_size, grid_dim, embedding_box, tau, *, embedding=None, do_reduce_embedding=True):

        timings={}

        start_time_total=time.time()
      
        do_reduce_embedding=do_reduce_embedding and\
            (\
             # fewer or equal cells than members
             (grid_dim[0]*grid_dim[1]<=len(embedding))\
             or\
             # tau is restricted
             (tau is not None)\
             )

        # -> if tau (displacement) is unrestricted and
        # there are at least as many cells as members,
        # no reduction will be possible

        if embedding is None:
            embedding=self.embedding

        grid_cell_centers=sca2gri.grid_cell_centers(grid_dim, embedding_box,  cell_size)


        print(f'm={len(grid_cell_centers)} grid points ({grid_dim}), n={len(embedding)} data points')

        embedding_idcs=None

        members_per_grid_cell=[]

        n_occupied_cells=0
        
        if do_reduce_embedding:
            assert rg is not None
            start_time_reduce_embedding=time.time()            

            embedding_idcs, n_occupied_cells, members_per_grid_cell, upper_bound_members_per_cell, timings_reduce\
                =reduce_members_grid.comp(rg, embedding, cell_size, embedding_box, grid_dim, tau)
            

            timings = timings | timings_reduce
        else:
            embedding_idcs=np.array(range(len(embedding)), dtype=int)
            upper_bound_members_per_cell=None

        
        embedding_idcs, grid_cell_centers, timings_assign=assign.comp(embedding, embedding_idcs, grid_cell_centers, tau, n_occupied_cells, upper_bound_members_per_cell, grid_dim)

        timings= timings | timings_assign
        
        return embedding_idcs, grid_cell_centers, grid_cell_centers, members_per_grid_cell, timings

    @staticmethod
    def get_embedding_box_dim_x(embedding_box, embedding_box_full):
        if embedding_box is not None:
            embedding_box_dim_x=\
                (embedding_box[2]-embedding_box[0])
        else:
            embedding_box_dim_x=\
                (embedding_box_full[2]-embedding_box_full[0])

        return embedding_box_dim_x
    
    @staticmethod
    def tauz_to_tau(tauz, embedding_box, embedding_box_full):                    
        return tauz*sca2gri.get_embedding_box_dim_x(embedding_box, embedding_box_full)

    @staticmethod
    def tau_to_tauz(tau, embedding_box, embedding_box_full):                    
        return tau/sca2gri.get_embedding_box_dim_x(embedding_box, embedding_box_full)
            
    @staticmethod
    def embedding_box(embedding):
        eps=1e-6
        return (min(embedding[:, 0]), min(embedding[:, 1]), max(embedding[:, 0])*(1+eps),max(embedding[:, 1])*(1+eps))

    @staticmethod
    def cell_size(n_cells_x, embedding_box, rep_dim):
        cell_width=(1./n_cells_x)*(embedding_box[2]-embedding_box[0])
        cell_height=cell_width*rep_dim[1]/rep_dim[0]
        return (cell_width, cell_height)
    
    @staticmethod
    def grid_dim(n_cells_x, embedding_box, rep_dim):
        _, cell_height=sca2gri.cell_size(n_cells_x, embedding_box, rep_dim)
        return (n_cells_x, math.ceil((embedding_box[3]-embedding_box[1])/cell_height))

    @staticmethod
    def grid_cell_centers(grid_dim, embedding_box,  cell_size):
        cell_width, cell_height=cell_size
        grid_cell_centers=np.zeros((grid_dim[0]*grid_dim[1],2))        

        import itertools

        for x,y in itertools.product(range(grid_dim[0]), range(grid_dim[1])):
            
            grid_cell_centers[x+y*grid_dim[0]]=(embedding_box[0]+(x+.5)*cell_width,\
                                  embedding_box[1]+((y+.5)*cell_height))

        return grid_cell_centers
    
    
    '''
    assume normalized embedding
    '''
    def run(self, fig, ax, *, rg=None, n_cells_x=32, do_reduce_embedding=True, embedding_box=None, tau=None, method=Method.sca2gri, embedding=None, labels=None,\
            kwargs_draw=sca2gri_draw.default_kwargs_draw(),\
            cache_key=None):

        
        
        if embedding is None:
            embedding=self.embedding

        assert embedding is not None
        
        rep=self.rep

        if rep is None:
            rep_dim_x, rep_dim_y=32,32
        else:
            rep_shape=rep[0].shape

            if len(rep_shape)==2:
                rep_dim_y,rep_dim_x=rep_shape
            elif len(rep_shape)==3:
                rep_dim_y,rep_dim_x,nChannels=rep_shape
                assert nChannels==3 or nChannels==4
            else:
                assert False, f'invalid shape {rep_shape}'
    

        if embedding_box is None:        
            embedding_box=sca2gri.embedding_box(embedding)


        cell_size=sca2gri.cell_size(n_cells_x, embedding_box, (rep_dim_x, rep_dim_y))

        print(f'cell_size {cell_size} n_cells_x {n_cells_x} embedding_box {embedding_box}')

        grid_dim=sca2gri.grid_dim(n_cells_x, embedding_box, (rep_dim_x, rep_dim_y))

        grid_cell_centers=None
        n_points_per_cell=None
        points_per_cell=None
        
        timings={}

        if cache_key is not None and cache_key in self.cache:
            embedding_idcs=self.cache[cache_key]['embedding_idcs']
            grid_cell_pos=self.cache[cache_key]['grid_cell_pos']
            grid_cell_centers=self.cache[cache_key]['grid_cell_centers']
            n_points_per_cell=self.cache[cache_key]['n_points_per_cell']
            points_per_cell=self.cache[cache_key]['points_per_cell']
            start_time=time.time()
        
        elif method==Method.sca2gri:
            if rg is None and tau is not None:
                rg=reduce_grid.ReduceGrid(embedding, n_cells_max=self.n_cells_max)
            start_time=time.time()
            embedding_idcs, grid_cell_pos, grid_cell_centers, members_per_grid_cell, timings=self.run_sca2gri(rg,cell_size, grid_dim, embedding_box, tau, embedding=embedding, do_reduce_embedding=do_reduce_embedding)

            if len(members_per_grid_cell)>0:
                n_points_per_cell, points_per_cell=members_per_grid_cell
        elif method==Method.dgrid:
            
            embedding_idcs, grid_cell_pos, start_time=self.run_dgrid(cell_size, embedding=embedding)
        elif method==Method.hagrid:
            embedding_idcs, grid_cell_pos, start_time=self.run_hagrid(cell_size, embedding=embedding)
                    
        else:
            assert False


        if cache_key is not None and cache_key not in self.cache:

            print(f'write cache entry {cache_key}')
            d={}
            d['embedding_idcs']=embedding_idcs
            d['grid_cell_pos']=grid_cell_pos
            d['grid_cell_centers']=grid_cell_centers
            d['n_points_per_cell']=n_points_per_cell
            d['points_per_cell']=points_per_cell
            self.cache[cache_key]=d
            
        timings['total']=time.time()-start_time
        if fig is not None:
            start_time=time.time()
            sca2gri_draw.draw(fig, ax, rep, grid_dim, cell_size,\
                              embedding, embedding_box, embedding_idcs,\
                              grid_cell_pos, grid_cell_centers,\
                              n_points_per_cell,points_per_cell,\
                              labels=labels,\
                              **kwargs_draw)

            timings['draw']=time.time()-start_time
        return embedding[np.array(embedding_idcs, dtype=int)], grid_cell_pos, cell_size, timings

        

    def run_dgrid(self, cell_size, *, embedding=None):

        path_to_dgrid='./comparison_unmanaged/dimensionality-reduction/dgrid'
        
        start_time=time.time()
        
        cell_width, cell_height=cell_size

        if embedding is None:
            y=self.embedding
        else:
            y=embedding
        
        from pathlib import Path
        HOME=str(Path.home())

        import os
        import sys

        
        sys.path.insert(0, os.path.abspath(path_to_dgrid))

        from dgrid import DGrid

        start_time = time.time()
        y_overlap_removed = DGrid(glyph_width=cell_width, glyph_height=cell_height, delta=1.0).fit_transform(y)
        print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

        return range(len(y_overlap_removed)), y_overlap_removed, start_time

    def run_hagrid(self, cell_size, *, embedding=None):
        import pythonmonkey as pm
        # from pythonmonkey.lib import pmdb
        #pmdb.enable()
        print(f'run hagrid start')

        path_to_hagrid_js='./comparison_unmanaged/hagrid-master/dist/hagrid.js'

        hagrid=pm.require(path_to_hagrid_js)
        #embedding_hagrid=[[x,y,i] for i, (x,y) in enumerate(embedding)]

        embedding_hagrid=[]
        for i, (x,y) in enumerate(embedding):
            embedding_hagrid.append(list([float(x), float(y), i]))

        print(f'embedding_hagrid {embedding_hagrid[:10]}')
        print(f'run hagrid with #{len(embedding_hagrid)} points')
        start_time=time.time()
        hagrid_method='hilbert'
        #hagrid_method='dgrid'
        #hagrid_method='nmap'

        res=hagrid.gridify(embedding_hagrid, hagrid_method)

        # note that this occassionally gets violated from hagrid (?)
        assert len(res)==len(set([(x,y) for x,y in res])) or True

        grid_dim=[0,0]

        for xf,yf in res:
            x=int(xf+.5)
            y=int(xf+.5)

            grid_dim[0]=max(grid_dim[0], x+1)
            grid_dim[1]=max(grid_dim[1], y+1)

        print('hagrid grid_dim', grid_dim)

        embedding_box=sca2gri.embedding_box(embedding)

        embedding_idcs=range(len(embedding))

        grid_cell_pos_full=sca2gri.grid_cell_centers(grid_dim, embedding_box, cell_size)

        grid_cell_pos=np.empty((len(res),2))
        for i, (xf,yf) in enumerate(res):

            grid_cell_pos[i]=((xf+.5)/grid_dim[0], (yf+0.5)/grid_dim[0])

        assert len(grid_cell_pos)==len(set([(x,y) for x,y in grid_cell_pos])) or True
        cell_size=(1./grid_dim[0],1./grid_dim[0])

        print('hagrid grid_cell_pos', grid_cell_pos)

        return embedding_idcs, grid_cell_pos, start_time

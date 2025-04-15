
import mmetal
import numpy as np
import time
import sys
import ctypes


def load():
    
    with open('reduce_grid_kernel.metal') as f:
        shader_str=f.read()

    instance=mmetal.load(shader_str, 'reduce_grid')

    return instance

def alloc_set_embedding(instance, n_points, embedding):

    assert len(embedding)==n_points*2

    buf_embedding = mmetal.create_buffer(instance, n_points*2, ctype=ctypes.c_float)
    mmetal.upload(buf_embedding, embedding)

    return buf_embedding


def alloc_params_float_int(instance):
    buf_embedding_box_off__cell_size = mmetal.create_buffer(instance, 4, ctype=ctypes.c_float)
    buf_grid_dim = mmetal.create_buffer(instance, 2, ctypes.c_uint32)

    return buf_embedding_box_off__cell_size, buf_grid_dim


def set_embedding_box__cell_size(buf_embedding_box_off__cell_size, embedding_box, cell_size):
    mmetal.upload(buf_embedding_box_off__cell_size, (embedding_box[0],embedding_box[1],
                                                  cell_size[0], cell_size[1]))

def alloc_grid(instance, n_cells_max):
    buf_n_points_per_cell = mmetal.create_buffer(instance, n_cells_max, ctypes.c_uint32)
    buf_points_per_cell = mmetal.create_buffer(instance, n_cells_max*n_cells_max, ctypes.c_uint32)

    return buf_n_points_per_cell, buf_points_per_cell

def set_grid_dim(buf_grid_dim, grid_dim):
    mmetal.upload(buf_grid_dim, grid_dim)

def run(instance, n_points, buf_n_points_per_cell,\
                             buf_points_per_cell,\
                             buf_embedding,\
                             buf_embedding_box_off__cell_size,\
                             buf_grid_dim):

    mmetal.upload(buf_n_points_per_cell, buf_n_points_per_cell['n']*[0])

    mmetal.run(instance, n_points, [buf_n_points_per_cell,\
                                    buf_points_per_cell,\
                                    buf_embedding,\
                                    buf_embedding_box_off__cell_size,\
                                    buf_grid_dim])
    
if __name__ == '__main__':

    n_points = 1000*10

    grid_dim=(32,32)
    n_cells=grid_dim[0]*grid_dim[1]

    embedding_box=(0., 0., 1., 1.)

    cell_size=(embedding_box[2]/grid_dim[0],\
               embedding_box[3]/grid_dim[1])

    print(f'cell_size {cell_size}')

    embedding=np.random.rand(n_points*2).astype(float)

    instance=load()
    
    
    # populate it with random samples from a uniform distribution over [0, 1)
    buf_embedding=alloc_set_embedding(instance, n_points, embedding)

    
    buf_embedding_box_off__cell_size, buf_grid_dim=alloc_params_float_int(instance)
    buf_n_points_per_cell, buf_points_per_cell=alloc_grid(instance, n_cells)

    
    
    set_embedding_box__cell_size(buf_embedding_box_off__cell_size, embedding_box, cell_size)
    set_grid_dim(buf_grid_dim, grid_dim)
    
    
    
    
    
    for _ in range(10):

        # just for testing, can actually leave uninitialized
        buf_points_per_cell_INIT=2**32-1
        mmetal.upload(buf_points_per_cell, buf_points_per_cell['n']*[buf_points_per_cell_INIT])

        gpu_start = time.time()

        run(instance, n_points,\
            buf_n_points_per_cell,\
            buf_points_per_cell,\
            buf_embedding,\
            buf_embedding_box_off__cell_size,\
            buf_grid_dim)

        gpu_end = time.time()

        print(f'runtime {gpu_end-gpu_start}')


        # buf_n_points_per_cell.release()
        # buf_points_per_cell.release()
        # buf_embedding.release()
        # buf_embedding_box_off__cell_size.release()
        # buf_grid_dim.release()


        #
        # run some sanity checks
        #

        buf_n_points_per_cell_h=mmetal.download(buf_n_points_per_cell)
        buf_points_per_cell_h=mmetal.download(buf_points_per_cell)
        
        assert n_points==np.sum(buf_n_points_per_cell_h)
        #, f'{n_points} vs {np.sum(buf_n_points_per_cell['buf'])}'
    
    
        for cell_id in range(n_cells):
            n_points=buf_n_points_per_cell_h[cell_id]

            cell_idx=(cell_id % grid_dim[0], cell_id // grid_dim[0])


            pos=(-1, -1)

            if n_points>0:
                point_id=buf_points_per_cell_h[cell_id*n_cells]

                pos=(embedding[2*point_id], embedding[2*point_id+1])


            if False:
                print(f'cell {cell_idx}: pos {pos}')

            for i in range(cell_id*n_cells, cell_id*n_cells+n_points):
                assert buf_points_per_cell_h[i] != buf_points_per_cell_INIT

            for i in range(cell_id*n_cells+n_points, (cell_id+1)*n_cells):
                assert buf_points_per_cell_h[i] == buf_points_per_cell_INIT

    
    sys.exit(0)
    # -------------


    buffer1['buf'][:] = [i for i in range(buffer_size)]

    for off in range(9, 15):
        buffer2['buf'][0]=off
        np_start = time.time()
        out_np = [np.sqrt(i)+off for i in buffer1['buf']]
        np_end = time.time()

        gpu_start = time.time()
        instance.run_function(buffer_size, [buffer1, buffer2, buffer_atomic])
        #print(f'count {buffer_atomic['buf']}')
        gpu_end = time.time()

        print("Sqrt test - CPU time: ", np_end - np_start)
        print("Sqrt test - GPU time: ", gpu_end - gpu_start)

        assert(np.allclose(buffer1['buf'], out_np, atol=1e-5))



    buffer1.release()

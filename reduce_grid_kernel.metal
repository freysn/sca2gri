#include <metal_stdlib>

using namespace metal;

kernel void reduce_grid(device atomic_uint* buf_n_points_per_cell,			
			device uint32_t* buf_points_per_cell,
			const device float* buf_embedding,
			const device float* buf_embedding_box_off__cell_size,
			const device uint32_t* buf_grid_dim,
			uint id [[thread_position_in_grid]])
{

  const float2 embedding(buf_embedding[2*id],buf_embedding[2*id+1]);

  const float2 embedding_box_off(buf_embedding_box_off__cell_size[0],
			     buf_embedding_box_off__cell_size[1]);

  const float2 cell_size(buf_embedding_box_off__cell_size[2],
			 buf_embedding_box_off__cell_size[3]);

  
  const uint2 grid_dim(buf_grid_dim[0],buf_grid_dim[1]);
  

  const float2 grid_idx_f=(embedding-embedding_box_off)/cell_size;

  const int grid_idx_x=floor(grid_idx_f.x);
  const int grid_idx_y=floor(grid_idx_f.y);

  if(min(grid_idx_x, grid_idx_y)>=0 && grid_idx_x<grid_dim.x && grid_idx_y<grid_dim.y)
    {
      
      const uint32_t grid_id=grid_idx_x+grid_dim.x*grid_idx_y;
         
      const uint32_t points_per_cell_idx
	=atomic_fetch_add_explicit(&buf_n_points_per_cell[grid_id],
				   1, memory_order_relaxed);

      const uint32_t n_cells=grid_dim.x*grid_dim.y;
      const uint32_t max_n_points_per_cells=n_cells;
      if(points_per_cell_idx<max_n_points_per_cells)
	buf_points_per_cell[grid_id*n_cells+points_per_cell_idx]=id;
    }
}

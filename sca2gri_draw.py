import matplotlib.pyplot as plt

import matplotlib
import helper_sca2gri

import numpy as np

from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                  TextArea, OffsetBox)


def draw_tiles(fig, ax, rep, grid_dim, cell_size, embedding_box, embedding_idcs, grid_cell_pos, *, patch_frame_edgecolor=None, rep_zoom_fac=0.8):


    cell_width, cell_height=cell_size
    
    zoom_width=None

    for i_embedding, (x,y) in zip(embedding_idcs, grid_cell_pos):

        img=rep[i_embedding]
        
        assert np.isclose(img.shape[1]/img.shape[0], cell_width/cell_height), f'np.isclose({img.shape[1]}/{img.shape[0]}={img.shape[1]/img.shape[0]}, {cell_width}/{cell_height}={cell_width/cell_height})'

        zoom_base=0.2
        
        cmap=None

        #for gray color map instead of default
        #cmap='gray'
        
        imagebox = OffsetImage(img, zoom=zoom_base, zorder=10, cmap=cmap)


        

        ecol=None
        if patch_frame_edgecolor is not None:
            assert i_embedding in patch_frame_edgecolor
            ecol=patch_frame_edgecolor[i_embedding]
        
        ab = AnnotationBbox(imagebox, (x,y), xycoords='data', frameon=(ecol is not None),pad=0)

        if patch_frame_edgecolor is not None:
            ab.patch.set_linewidth(1.5)
            ab.patch.set_edgecolor(ecol)

        ax.add_artist(ab)

        if zoom_width is None:
            extent_display=ab.get_window_extent()        

            width_display=extent_display.x1-extent_display.x0
            height_display=extent_display.y1-extent_display.y0

            assert np.isclose(width_display/height_display, cell_width/cell_height), f'np.isclose({width_display}/{height_display}={width_display/height_display}, {cell_width}/{cell_height}={cell_width/cell_height})'

            inv = ax.transData.inverted()


            from_data, to_data=inv.transform([(extent_display.x0,extent_display.y0),(extent_display.x1,extent_display.y1)])

            width_data=to_data[0]-from_data[0]
            height_data=to_data[1]-from_data[1]

            assert np.isclose(width_data/height_data, cell_width/cell_height, rtol=0.05), f'np.isclose({width_data}/{height_data}={width_data/height_data}, {cell_width}/{cell_height}={cell_width/cell_height})'

            zoom_width=cell_width/width_data
            zoom_height=cell_height/height_data

            assert np.isclose(zoom_width/zoom_height, 1., rtol=0.05)

        ab.offsetbox.set_zoom(rep_zoom_fac*zoom_width*zoom_base)


def draw(fig, ax, rep, grid_dim, cell_size, embedding, embedding_box, embedding_idcs, grid_cell_pos, grid_cell_centers, n_points_per_cell, points_per_cell, rep_zoom_fac=0.8, labels=None, do_show_grid_cell_centers=False, do_show_scatterplot=False, do_show_displacement_indicator=True, do_show_rep_frame_n_members=False, do_show_embedding_box=False, displacement_indicator_markersize=3, displacement_indicator_lw=0.5, displacement_indicator_zorder=20, embedding_box_col='gray', embedding_box_ls='-', zoom_boxes=[], do_draw_tiles=True):


    cell_width, cell_height=cell_size

    max_n_members_per_grid_cell=-1


    n_cells=grid_dim[0]*grid_dim[1]

    if n_points_per_cell is not None:
        max_n_members_per_grid_cell=max(n_points_per_cell)

    assert ax is not None
    '''
    plotting
    '''

    # Set equal scaling (i.e., make circles circular) by changing the axis limits.
    ax.set_aspect('equal', adjustable='datalim')

    cmap = matplotlib.colormaps['viridis']


    if grid_cell_centers is not None and do_show_grid_cell_centers:
        ax.scatter(grid_cell_centers[:, 0], grid_cell_centers[:, 1], color=(0.6, 0.6, 0.6))

    if do_show_scatterplot:

        

        # with this, all indices are selected regardless of area
        idcs=range(len(embedding))
        #idcs=np.where((embedding[:,0]>=embedding_box[0]) & (embedding[:,1]>=embedding_box[1]) & (embedding[:,0]<=embedding_box[2]) & (embedding[:,1]<=embedding_box[3]))

        print(f'there are {len(idcs)} scatterplot points')

        col=(0.6, 0.6, 0.6, 0.2)

        print(f'len(col) {len(col)} for {len(idcs)} idcs')

        ax.scatter(embedding[idcs, 0], embedding[idcs, 1], color=col, s=0.1)

    x_values=[]
    y_values=[]

    if do_show_displacement_indicator:
        for i_embedding, pos in zip(embedding_idcs, grid_cell_pos):
            x2,y2=pos
            x1,y1=embedding[i_embedding]

            col='black'
            zorder=displacement_indicator_zorder

            # if not do_show_displacement_indicator:
            #     zorder=0
            #     col='white'

            alpha=0.5
            ax.plot((x1, x2), (y1, y2), zorder=zorder, lw=displacement_indicator_lw, color=col, dash_capstyle='round', alpha=alpha)
            ax.plot(x1, y1, zorder=zorder, color=col, marker='o', markersize=displacement_indicator_markersize, alpha=alpha)

    if do_show_embedding_box:
        ax.add_patch(matplotlib.patches.Rectangle((embedding_box[0], embedding_box[1]), embedding_box[2]-embedding_box[0], embedding_box[3]-embedding_box[1], linewidth=4, edgecolor=embedding_box_col, facecolor='none', zorder=0, ls=embedding_box_ls))

    ax.set_xlim([embedding_box[0], embedding_box[2]+cell_size[0]])
    ax.set_ylim([embedding_box[1], embedding_box[3]+cell_size[1]])

    for zoom_box, zoom_box_col, zoom_box_ls in zoom_boxes:
        rect = matplotlib.patches.Rectangle((zoom_box[0], zoom_box[1]), zoom_box[2]-zoom_box[0], zoom_box[3]-zoom_box[1], linewidth=4, edgecolor=zoom_box_col, facecolor='none', zorder=100, linestyle=zoom_box_ls)

        ax.add_patch(rect)



    # this is required to have correct dimensions
    fig.canvas.draw()

    if do_show_rep_frame_n_members:
        patch_frame_edgecolor={}
    else:
        patch_frame_edgecolor=None

    if rep is not None:
        n_rep=rep
    else:
        n_rep={}
        if labels is not None:
            max_label=max(labels)

    if (patch_frame_edgecolor is not None) or (rep is None):


        for i_embedding, (x,y) in zip(embedding_idcs, grid_cell_pos):
            x_idx=int((x-embedding_box[0])/cell_width)
            y_idx=int((y-embedding_box[1])/cell_height)

            assert x_idx<grid_dim[0], f'CHECK {x} {y} | {x/cell_width} {grid_dim} {max(embedding[:,0])} {cell_width} {cell_height}'
            assert y_idx<grid_dim[1]

            members=[]
            n_members=0
            
            if n_points_per_cell is not None:            
                
                grid_idx=helper_sca2gri.ii2i((x_idx, y_idx), grid_dim)
                n_members=n_points_per_cell[grid_idx]

                off=n_cells*grid_idx
                
                members=points_per_cell[off:off+min(n_members, n_cells)]

            if rep is None:

                if labels is None:
                    img=np.ones((8,8))*.5
                else:
                    img=np.ones((8,8,4))

                    if max_label==0:
                        col=cmap(.5)
                    else:
                        col=cmap(labels[i_embedding]/max_label)
                        img[:,:,0]=col[0]
                        img[:,:,1]=col[1]
                        img[:,:,2]=col[2]

                    op=0.7

                    if max_n_members_per_grid_cell>1:
                        base_opacity=0.2
                        op=base_opacity+(1.-base_opacity)*(n_members/max_n_members_per_grid_cell)

                    img[:,:,3]=op
                n_rep[i_embedding]=img

            if do_show_rep_frame_n_members:                

                n_members_shown=len(set(embedding_idcs).intersection(set(members)))
                n_members_omitted=n_members-n_members_shown

                if n_members_omitted==0:
                    col=None
                else:
                    col=cmap(n_members_omitted/max_n_members_per_grid_cell)

                patch_frame_edgecolor[i_embedding]=col
            

    if do_draw_tiles:
        draw_tiles(fig, ax, n_rep, grid_dim, cell_size, embedding_box, embedding_idcs, grid_cell_pos, rep_zoom_fac=rep_zoom_fac, patch_frame_edgecolor=patch_frame_edgecolor)




def default_kwargs_draw():
    return {'rep_zoom_fac' : 0.8,
            'do_show_grid_cell_centers' : False,\
            'do_show_scatterplot': False,\
            'do_show_displacement_indicator': True,\
            'do_show_rep_frame_n_members': False,\
            'do_show_embedding_box': False,\
            'displacement_indicator_markersize': 3,\
            'displacement_indicator_lw': 0.5,\
            'displacement_indicator_zorder': 20,
            'embedding_box_col':'gray',\
            'embedding_box_ls' : '-',\
            'zoom_boxes' : [],\
            'do_draw_tiles' : True}

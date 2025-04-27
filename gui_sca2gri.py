import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, Button, Slider, TextBox

from matplotlib.widgets import CheckButtons

import sca2gri
import time
import reduce_grid

import argparse
import shutil
from pathlib import Path

import helper_sca2gri as helper

import sca2gri_draw

import copy

import sys

class GUI_sca2gri:

    def __init__(self, args):

        self.history_index=-1
        
        self.fname_pdf_counter=None

        self.embedding_box=None
        
        
        if args.do_pdf:

            self.fname_pdf_counter=0
            
            self.fname_pdf_dir=f'/tmp/gui_sca2gri/{args.caseName}'
            helper.create_empty_dir(self.fname_pdf_dir)

        self.log=[]
        self.log_fname=None

        self.draw_args=sca2gri_draw.default_kwargs_draw()
        
        self.draw_args['do_show_scatterplot']=True
        self.draw_args['do_draw_tiles']=True
        self.draw_args['do_show_displacement_indicator']=True
        self.draw_args['rep_zoom_fac']=0.95

        self.do_write_log=False

        self.do_cache=args.do_cache

        if args.do_log:
            self.log=[{'caseName' : args.caseName}]
            path=f'{helper.home()}/tmp/gui_sca2gri'
            helper.create_dir_if_not_exists(path)
            self.log_fname=f'{path}/{args.caseName}_log.pkl'
            helper.pkl_dump(self.log,self.log_fname)
            self.do_write_log=True
            
        self.init_fig_ax()

        # embedding=helper.pkl_load(args.emb)

        # if embedding is None:
        #     print(f'embedding could not be loaded from {args.embedding}')
        #     sys.exit(0)

        # if args.rep.endswith('.pkl'):
        #     rep=helper.pkl_load(args.rep)

        #     if rep is None:
        #         print(f'rep could not be loaded from {args.rep}')
        #         sys.exit(0)
        # else:
        #     import rep_imgs
        #     rep=rep_imgs.rep_imgs(args.rep)

        #     if len(rep)==0:
        #         print(f'rep could not be loaded from {args.rep}')
        #         sys.exit(0)

        import sca2gri_load

        embedding, rep=sca2gri_load.load_emb_rep(args.emb, args.rep)
        
        self.sca2gri=sca2gri.sca2gri(embedding, rep)

        self.sca2gri_rg=reduce_grid.ReduceGrid(self.sca2gri.embedding, do_metal=not args.no_metal)

        self.embedding_box_full=sca2gri.sca2gri.embedding_box(self.sca2gri.embedding)
        
        self.n_cells_x=12
        self.tau=0.1
        
        if True:            
            self.update()
            #self.run_sca2gri()
        else:
            self.scatter = self.ax.scatter(x, y)


        

        #(left, bottom, width, height)
        ax_check = plt.axes([0.35, 0.85, .3, 0.08])  # Position of checkboxes

        self.checkbox_name2label={'scatterplot' : 'do_show_scatterplot',\
                                  'tiles' : 'do_draw_tiles',\
                                  'displacement indicator' : 'do_show_displacement_indicator'}
                        
        checkbox = CheckButtons(ax_check,\
                                ['scatterplot',\
                                 'tiles',\
                                 'displacement indicator'],\
                                [self.draw_args['do_show_scatterplot'],\
                                 self.draw_args['do_draw_tiles'],\
                                 self.draw_args['do_show_displacement_indicator']])

        checkbox.on_clicked(self.toggle_checkboxes)

            
        # Store the original limits to reset later
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        
        self.init_rect_selector()
        
        # Add a reset button
        reset_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
        self.reset_button = Button(reset_ax, 'Reset Zoom')
        self.reset_button.on_clicked(self.reset_zoom)

        # Add a back button
        back_ax = plt.axes([0.75, 0.01, 0.02, 0.05])
        self.back_button = Button(back_ax, '<')
        self.back_button.on_clicked(self.go_back)

        # Add a forward button
        forward_ax = plt.axes([0.77, 0.01, 0.02, 0.05])
        self.forward_button = Button(forward_ax, '>')
        self.forward_button.on_clicked(self.go_forward)
        


        textbox_width=0.05
        textbox_alignment='center'
        
        # Add a TextBox for point size
        textbox_ax_size = plt.axes([0.2, 0.01, textbox_width, 0.05])
        self.textbox_size = TextBox(textbox_ax_size, r'$\tau_z\,=\,$ ', initial=str(self.tau), textalignment=textbox_alignment)
        self.textbox_size.on_submit(self.update_tau)

        # Add a TextBox for point size
        textbox_ax_n_cells = plt.axes([0.6, 0.01, textbox_width, 0.05])
        self.textbox_n_cells = TextBox(textbox_ax_n_cells, r'$g_x\,=\,$ ', initial=str(self.n_cells_x), textalignment=textbox_alignment)
        self.textbox_n_cells.on_submit(self.update_n_cells_x)

        self.update()
        
        plt.show()

    def toggle_checkboxes(self, checkbox_name):

        label=self.checkbox_name2label[checkbox_name]
        self.draw_args[label] = not self.draw_args[label]
        
        self.update()
        self.ax.figure.canvas.draw_idle()

    def update(self, *, kwargs=None):
                        
        
        self.run_sca2gri(kwargs=kwargs)


        if self.do_write_log:
            helper.pkl_dump(self.log,self.log_fname)

        if self.fname_pdf_counter is not None:
            fname=f'{self.fname_pdf_dir}/{str(self.fname_pdf_counter).zfill(2)}.pdf'
            print(f'create {fname}')
            self.run_sca2gri(fname_pdf=fname, force_redraw=True)
            self.fname_pdf_counter+=1
        else:
            self.ax.figure.canvas.draw_idle()
        
    
    def run_sca2gri(self, fname_pdf=None, kwargs=None,\
                    force_redraw=False):

        kwargs_in_none=(kwargs is None)
        
        if kwargs_in_none:
            kwargs={'n_cells_x' : self.n_cells_x,\
                    'tau' : sca2gri.sca2gri.tauz_to_tau(self.tau, self.embedding_box, self.embedding_box_full),\
                    'embedding_box' : self.embedding_box,
                    'kwargs_draw' : self.draw_args}

        if self.do_cache:
            kwargs_hash=helper.hash_dict(kwargs)
        else:
            kwargs_hash=None
        
        if len(self.log) == 0 or self.log[self.history_index] != kwargs or force_redraw:
            

            
            fig=self.fig
            ax=self.ax

            if fname_pdf is not None:
                fig, ax = plt.subplots()
                plt.title(f'zoom={self.embedding_box}, tau={self.tau}, n_cells_x={self.n_cells_x}')
            else:
                ax.clear()
                self.init_rect_selector()

                    
            start_time=time.time()
            _, _, _, timings=self.sca2gri.run(fig, ax, rg=self.sca2gri_rg, **kwargs, cache_key=kwargs_hash)
            
            print(f'####################sca2gri took {time.time()-start_time}s | {timings}')

            if fname_pdf is not None:
                plt.savefig(fname_pdf, dpi=1000)

            
        else:
            print(f'####################omitted updated as kwargs are unchanged')

            

        if kwargs_in_none:
            self.log.append(copy.deepcopy(kwargs))
        
        


    def init_fig_ax(self):
        self.fig, self.ax = plt.subplots(figsize=(10.24, 7.68))


    def init_rect_selector(self):
        # Connect the RectangleSelector widget
        #                                               useblit=True,
        self.rect_selector = RectangleSelector(self.ax, self.on_select,
                                               button=[1],  # Left mouse button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True,
                                               props={'facecolor': 'red', 'edgecolor':'black', 'alpha': 0.2, 'zorder': 10000, 'fill' : True})
    


        

    def on_select(self, eclick, erelease):
        """
        Callback function when a region is selected.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Ensure coordinates are sorted (bottom-left to top-right)
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        self.zoom(x1, x2, y1, y2)
        
        self.rect_selector.set_visible(False)

    def zoom(self, x1, x2, y1, y2):
        """
        Zoom into the selected area.
        """
        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y1, y2)

        embedding_box=(x1,y1, x2, y2)

        self.embedding_box=embedding_box
        self.update()
                

    def reset_zoom(self, event):
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)

        self.embedding_box=None

        self.update()

        self.ax.figure.canvas.draw_idle()


    def go_back(self, event):

        new_history_index=max(self.history_index-1, -len(self.log))
        

        print(f'history_index {new_history_index} [len(self.log) {len(self.log)}]')
        
        self.update(kwargs=self.log[new_history_index])

        self.history_index=new_history_index

    def go_forward(self, event):

        new_history_index=min(self.history_index+1, -1)

        print(f'history_index {self.history_index} [len(self.log) {len(self.log)}]')
        
        self.update(kwargs=self.log[new_history_index])

        self.history_index=new_history_index

    def update_tau(self, val):

        self.tau=float(val)
        
        self.update()        
        
        self.ax.figure.canvas.draw_idle()

    def update_n_cells_x(self, val):

        self.n_cells_x=int(val)
        
        self.update()        
        
        self.ax.figure.canvas.draw_idle()

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rep', default='case_data/dropglyph_rep.zip')
    
    parser.add_argument('--emb', default='case_data/dropglyph_emb.pkl')

    parser.add_argument('--no_metal', action="store_true")

    parser.add_argument('--caseName', default='dropglyph')

    parser.add_argument('--do_pdf', action="store_true")
    parser.add_argument('--do_log', action="store_true")

    parser.add_argument('--do_cache', action="store_true")
    
    args = parser.parse_args()

    GUI_sca2gri(args)

from numba import njit

@njit
def ii2i(grid_cell, grid_dim):
    return grid_cell[0]+grid_cell[1]*grid_dim[0]

@njit
def i2ii(grid_cell_1, grid_dim):

    y=grid_cell_1//grid_dim[0]
    x=grid_cell_1-grid_dim[0]*y
    return (x,y)


import pickle
import numpy as np

def pkl_load(fname):

    o=None
    try:
        f=open(fname, 'rb')
        o=pickle.load(f)
    except:
        o=None
        
    return o

def pkl_dump(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

import shutil
from pathlib import Path

def create_empty_dir(fname_pdf_dir):
    shutil.rmtree(fname_pdf_dir, ignore_errors=True)        
    Path(fname_pdf_dir).mkdir(parents=True)


def create_dir_if_not_exists(fname_pdf_dir):
    Path(fname_pdf_dir).mkdir(parents=True, exist_ok=True)



import subprocess

def pdfcrop(fname):
    subprocess.run(["pdfcrop", fname, fname])

def pdf2png(fname_pdf, fname_png, *, density=400):
    subprocess.run(['convert', '-verbose', '-density', str(density), '-trim', fname_pdf, '-flatten', fname_png])

def pngcrop(fname_png):
    subprocess.run(['convert', fname_png, '-trim', fname_png])

def home():
    from pathlib import Path
    return str(Path.home())

def hex_to_binary_array(hex_string, *, n=64):
    # Convert hex string to binary string
    binary_string = bin(int(hex_string, 16))[2:]  # Convert to binary and remove "0b" prefix
                    
    # Ensure the length is a multiple of 8 for byte alignment
    #padded_binary_string = binary_string.zfill(8 * ((len(binary_string) + 7) // 8))
    padded_binary_string = binary_string.zfill(n)
    
    # Convert the binary string to a list of integers
    binary_array = np.array([int(bit) for bit in padded_binary_string], dtype=np.uint8)
                    
    return binary_array



from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))




def hash_dict(d):
    """Recursively hash a dictionary that may contain dictionaries or lists."""
    if not isinstance(d, dict):
        raise TypeError("Input must be a dictionary.")
    
    # To handle hashable and non-hashable items in a consistent manner
    def hash_value(value):
        if isinstance(value, dict):
            return hash_dict(value)  # Recursively hash dictionaries
        elif isinstance(value, list):
            return hash(frozenset(hash_value(item) for item in value))  # Recursively hash list elements
        else:
            return value  # Return hashable items directly (like strings, numbers)
        
    # Convert the dictionary into a frozenset of (key, hashed_value) pairs to ensure immutability
    return hash(frozenset((key, hash_value(value)) for key, value in d.items()))

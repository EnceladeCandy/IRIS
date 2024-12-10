import os
import sys
sys.path.append("../src")
import torch 
from astropy.visualization import ImageNormalize, AsinhStretch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
import mpol.constants as const
from glob import glob
import json
from utils import create_dir

from preprocessing.gridding import grid, pillbox_window, sinc_window, bin_data

from functools import partial

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args):
    npz_dir = args.npz_dir 
    output_dir = args.output_dir # must be a folder
    
    # To grid multiple datasets across nodes (npz_dir must be the path to a folder)
    if N_WORKERS > 1: 
        paths = glob(os.path.join(npz_dir, "*.npz"))
        assert len(paths) > 1
        print(f"ARRAY JOB, THIS_WORKER={THIS_WORKER}")
        path = paths[THIS_WORKER]
        basename = os.path.basename(path)
        target_name, _= os.path.splitext(basename)
        dirname = os.path.dirname(path)
        # Importing target parameters 
        target_params_dir = "../dsharp_params.json" # a file with the pixel scale info for each target. 
        with open(target_params_dir, 'r') as file:
            dict_disks_info = json.load(file) 
        if args.target_name: # meant to be used only for the AS205_binary. Otherwise the right file is picked up 
            target_name = args.target_name
        disk_info = dict_disks_info[target_name]
        pixel_scale = disk_info['pixel_scale'] # arcsec
    
    # If gridding only one dataset (npz_dir must be the path to an npz file)
    elif N_WORKERS == 1: 
        print("Gridding one dataset...")
        paths = [args.npz_dir]
        path = paths[0]
        basename = os.path.basename(path)
        target_name, _= os.path.splitext(basename)
        target_params_dir ="../dsharp_params.json" # I spent some time creating this file, please use it. 
        
        with open(target_params_dir, 'r') as file:
            dict_disks_info = json.load(file) 
        if args.target_name: # meant to be used only for the AS205_binary. Otherwise the right file is picked up 
            target_name = args.target_name
        disk_info = dict_disks_info[target_name]
        pixel_scale = disk_info['pixel_scale'] # arcsec

    
    print(f"Gridding visibilities for {target_name} with pixel size = {pixel_scale:.2g} arcsec...")
    # Importing processed DSHARP data.
    data = np.load(path)
    u = data["uu"]
    v = data["vv"]
    vis = data["data"]
    weight = data["weight"]

    # Hermitian augmentation:
    uu = np.concatenate([u, -u])
    vv = np.concatenate([v, -v])
    vis_re = np.concatenate([vis.real, vis.real])
    vis_imag = np.concatenate([vis.imag, -vis.imag])
    weight_ = np.concatenate([weight, weight])

    print(f"The measurement set contains {len(uu)} data points")
    npix = args.npix
    img_size = args.img_size
    u_edges, v_edges = grid(pixel_scale = pixel_scale, img_size = npix)
    print(uu.min(), u_edges.min())
    print(uu.max(),  u_edges.max())
    print(vv.min(),  v_edges.min())
    print(vv.max(),  v_edges.max())

    # Removed, we just don't model this data
    # assert u_edges.min() < uu.min()
    # assert u_edges.max() > uu.max()
    # assert v_edges.min() < vv.min()
    # assert v_edges.max() > vv.max()
    delta_u = u_edges[1] - u_edges[0]
    truncation_radius = delta_u

    if args.window_function == "sinc": 
        window_fn = partial(sinc_window, pixel_size=delta_u)
    
    elif args.window_function == "pillbox": 
        window_fn = partial(pillbox_window, pixel_size=delta_u)
    else:
        raise ValueError("The window function specified is not implemented yet or does not exist ! Choose between 'sinc' and 'pillbox'.")

        # Real part mean and count
    if not args.debug_mode:
        params = (uu, vv, vis_re, weight_, (u_edges, v_edges), window_fn, truncation_radius)
        vis_bin_re = bin_data(*params, statistics_fn="mean", verbose=1)
        std_bin_re = bin_data(*params, statistics_fn="std", verbose=2)

        # Image part mean
        params = (uu, vv, vis_imag, weight_, (u_edges, v_edges), window_fn, truncation_radius)
        vis_bin_imag = bin_data(*params, statistics_fn="mean", verbose=1)
        std_bin_imag = bin_data(*params, statistics_fn="std", verbose=2)

        # Count: 
        counts = bin_data(*params, statistics_fn="count", verbose=1)
    
    if args.experiment_name: 
        suffix = f"_{args.experiment_name}"

    else: 
        suffix = ""
    create_dir(output_dir)
    save_dir = os.path.join(output_dir, f"{target_name}_{npix}_gridded_{args.window_function}" + suffix + ".npz")
    if not args.debug_mode:
        np.savez(
            save_dir, 
            vis_bin_re = vis_bin_re,
            vis_bin_imag = vis_bin_imag,
            std_bin_re = std_bin_re,
            std_bin_imag = std_bin_imag,
            counts = counts
        )

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Sampling parameters
    parser.add_argument("--npz_dir",            required = True,  type = str,                       help = "Path to the processed .npz measurement set")
    parser.add_argument("--target_name",        required = False, type = str,     default = None,   help = "Target name. If not specified, the code automatically uses the format 'diskname_continuum' from the gridded file.")
    parser.add_argument("--output_dir",         required = True,  type = str,                       help = "Directory where to save the gridded visibilities (specify a folder not any type of file)")
    parser.add_argument("--npix",               required = False, type = int,    default = 4096,    help = "Total number of pixels of the padded image.")
    parser.add_argument("--window_function",    required = False, type = str,    default = "sinc",  help = "Either 'sinc' or 'pillbox'")
    parser.add_argument("--img_size",           required = False, type = int,    default = 256,     help = "Number of pixels of the image (must be the same dimension as the dimensions of the score model)")
    parser.add_argument("--debug_mode",         required = False, type = bool,   default = False,   help = "To debug the code, skip the gridding.")
    parser.add_argument("--experiment_name",    required = False, type = str,    default = "",      help = "Name of the experiment, comes after the target name.")

    args = parser.parse_args()
    main(args) 

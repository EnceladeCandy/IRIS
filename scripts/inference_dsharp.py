import os
import sys
import json
import torch 
from torch.func import vmap
import numpy as np
import math
from glob import glob 
from score_models import ScoreModel
sys.path.append("../src")
from inference.posterior_sampling import score_likelihood, euler_sampler, pc_sampler
from inference.fourier import complex_to_real, ftshift
from inference.forward_model import model, link_function, gaussian_pb
import astropy.units as units
from utils import create_dir
from visualization import compute_dirty_image, compute_uv_window

device = "cuda" if torch.cuda.is_available() else "cpu"

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args):
    score_model = ScoreModel(checkpoints_directory=args.score_dir)
    gridded_npz_dir = args.gridded_npz_dir 
    output_dir = args.output_dir # must be a folder

    # To grid multiple datasets across nodes (npz_dir must be the path to a folder)
    if N_WORKERS > 1 and args.grid: 
        paths = glob(os.path.join(gridded_npz_dir, "*.npz"))
        assert len(paths) > 1
        print(f"Generating posterior samples for each npz file in the directory.")
        gpus_per_target = args.gpus_per_target
        path = paths[(THIS_WORKER) // gpus_per_target]

    # If gridding only one dataset (npz_dir must be the path to an npz file corresponding to the gridded visibilities)
    else: 
        path = gridded_npz_dir
    
    basename = os.path.basename(path)
    target_name, _= os.path.splitext(basename)
    parts = target_name.rsplit(f"_{args.npix}_gridded_sinc", 1)
    target_name = parts[0] if len(parts) > 1 else target_name
    dirname = os.path.dirname(path)
    # Importing target parameters 
    target_params_dir = "../dsharp_params.json" # a file with the pixel scale info for each target. 
    with open(target_params_dir, 'r') as file:
        dict_disks_info = json.load(file) 
    disk_info = dict_disks_info[target_name]
    pixel_scale = disk_info['pixel_scale']  
    
    
    # Importing gridded visibilities + noise estimates.  
    print("Importing gridded visibilities...")
    data_gridded = np.load(path) 
    vis_bin_re = torch.tensor(data_gridded["vis_bin_re"]).to(device)
    vis_bin_imag = torch.tensor(data_gridded["vis_bin_imag"]).to(device)
    std_bin_re = torch.tensor(data_gridded["std_bin_re"]).to(device)
    std_bin_imag = torch.tensor(data_gridded["std_bin_imag"]).to(device)
    counts = torch.tensor(data_gridded["counts"]).to(device)

    img_size = args.img_size # Number of rows/cols of the target image (in our case 256*256)
    npix = args.npix # Number of rows/cols of the gridded visibilities 
    
    S = counts>0 # sampling function

    vis_gridded = vis_bin_re + 1j * vis_bin_imag
    std_gridded = std_bin_re + 1j * std_bin_imag 
    vis_sampled = vis_gridded[S]
    std_sampled = std_gridded[S]

    dirty_image = compute_dirty_image(vis_gridded, norm = "backward")

    # Complex tensors into their vectorized real representations 
    y = complex_to_real(vis_sampled) 
    sigma_y = complex_to_real(std_sampled) 

    # Centering the image on the protoplanetary disk
    max_index = dirty_image.argmax().item()
    pixel_center_h = max_index // npix + disk_info["offset_h"]
    pixel_center_w = max_index % npix + disk_info["offset_w"]

    pad_b = (pixel_center_h - img_size // 2)
    pad_t = npix - (pixel_center_h + img_size // 2)
    pad_l = (pixel_center_w - img_size // 2)
    pad_r = npix - (pixel_center_w + img_size // 2)

    pad = (pad_l, pad_r, pad_b, pad_t) # torch convention (pad left, pad right, pad bottom, pad top)
    if "probes" in args.score_dir:
        print(f"Running inference with probes {img_size}*{img_size} for target {target_name}...") 
        if args.s: 
            B, C = dirty_image.max().item() / args.s, 0
        else: 
            B, C = dirty_image.max().item() / disk_info['probes_norm'], 0 
            
        dataset = "probes"

        if "vp" in args.score_dir:
            sde = "vp"
        else:
            sde = "ve"
        
    elif "skirt" in args.score_dir: 
        print(f"Running inference with skirt {img_size}*{img_size} for target {target_name}...")
        if args.s: 
            print('Scale factor was specified')
            B, C =  dirty_image.max().item() / args.s,  0
        else: 
            print('Taking scale factor from config file')
            B, C = dirty_image.max().item() / disk_info['skirt_norm'], 0 
        # B, C = dirty_image.max().item() / disk_info['skirt_norm'], 0 # SKIRT has more dynamic range
        dataset = "skirt"
        sde = "vp"

    padding_mode = args.padding_mode # either 'noise' or 'zero' 

    if padding_mode == "zero": 
        print("Running inference with zero-padding")

    elif padding_mode == "noise": 
        print("Running inference with noise padding")
    
    else:
        raise Warning("Padding specified does not exist... Running inference with default zero padding")
        padding_mode = 'zero'

    target_info_dir = os.path.join(args.npz_dir, target_name.split("_")[0] + "_continuum.npz")
    target_info = np.load(target_info_dir)
    freq_per_spw = target_info['freq_per_spw'] # in Hertz or s^-1

    if args.include_pb: 
        print("Primary beam included")
        pb, fwhm = gaussian_pb(diameter = 12, freq = freq_per_spw.mean(), shape = (npix, npix), pixel_scale = pixel_scale, device = device)
    else: 
        print("Primary beam not included")
        pb = torch.ones(size = (npix, npix), device = device)
    model_params = (S, pb, B, C, pad, padding_mode)

    # Generating posterior samples
    num_samples = args.num_samples
    batch_size = args.batch_size
    assert num_samples >= batch_size, "The number of samples should be superior to the batch size..."
    total_samples = np.empty(shape = [num_samples, 1, img_size, img_size], dtype = np.float32) 
    reconstructions = np.empty(shape = [num_samples, len(y)], dtype = np.float32)

    # Iterating the posterior sampling procedure.
    n_iterations = math.ceil(num_samples/batch_size)
    for i in range(n_iterations):
        if (i + 1) == n_iterations:
            n_samples_iteration = num_samples - i * batch_size
        else:
            n_samples_iteration = batch_size
        if args.sampler == "euler": 
            print("Sampling with the Euler-Maruyama sampler...")
            samples = euler_sampler(
            y = y,
            sigma_y = sigma_y, 
            forward_model = model, 
            score_model = score_model,
            score_likelihood =  score_likelihood, 
            model_parameters = model_params,
            num_samples = n_samples_iteration,
            num_steps = args.predictor,  
            tweedie = args.tweedie, # Experimental 
            keep_chain = False, 
            debug_mode = args.debug_mode, 
            img_size = (args.img_size, args.img_size)
        )

        elif args.sampler == "pc":
            print("Sampling with the Predictor-Corrector sampler...")
            sampling_params = [args.predictor, args.corrector, args.snr] 
            samples = pc_sampler(
            y = y,
            sigma_y = sigma_y, 
            forward_model = model, 
            score_model = score_model,
            score_likelihood = score_likelihood, 
            model_parameters = model_params,
            num_samples = n_samples_iteration,
            pc_params = sampling_params,  
            tweedie = args.tweedie, 
            keep_chain = False, 
            debug_mode = args.debug_mode, 
            img_size = (256, 256)
        )

        else: 
            raise ValueError("Sampler specified does not exist; choose between 'pc' and 'euler'.") 
        
        
        total_samples[i * batch_size: i* batch_size + n_samples_iteration] = link_function(samples, B, C).cpu().numpy().astype(np.float32)
        reconstructions[i * batch_size: i*batch_size + n_samples_iteration] = vmap(lambda x: model(None, x, None, model_params))(samples).cpu().numpy().astype(np.float32) 

        # if args.debug_mode:
        #      break
        
    # Creating experiment's directory
    print("Creating folder for the experiment in the results directory...")
    path_target = os.path.join(args.output_dir, target_name)
    create_dir(path_target)

    print("Creating folder for score model used...")
    path_dataset = os.path.join(path_target, f"{sde}_" + dataset)
    
    # Creating sampler directory
    print("Creating folder for the sampler used in the experiment's directory...")
    path_sampler = os.path.join(path_dataset, args.sampler)
    create_dir(path_sampler)

    # Creating directory according to the pc parameter being used
    print("Creating folder for the parameters used in the sampler's directory...")
    if args.sampler == "pc": 
        params_foldername = f"{args.predictor}pred_{args.corrector}corr_{args.snr}snr"
    elif args.sampler == "euler": 
        params_foldername = f"{args.predictor}steps"
    path_params = os.path.join(path_sampler, params_foldername)
    create_dir(path_params)
    


    # Converting from Jy/pixel to Jy/beam:
    print("Computing the units") 
    pb, fwhm = gaussian_pb(diameter = 12, freq = freq_per_spw.mean(), shape = (img_size, img_size), pixel_scale = pixel_scale, device = device)
    pb = pb.cpu().numpy().astype(np.float32)
    # fwhm_rad = fwhm * units.arcsec.to(units.rad)
    # pixel_area = pixel_scale * units.arcsec.to(units.radian) ** 2
    # fwhm_maj = disk_info["dirty_beam_fwhm_maj"] * 1e-3 * units.arcsec.to(units.rad)
    # fwhm_min = disk_info["dirty_beam_fwhm_min"] * 1e-3 * units.arcsec.to(units.rad)
    # beam_area = np.pi * (fwhm_maj * fwhm_min) / (4 * np.log(2))
    # samples_jy_per_beam = total_samples  * pixel_area/beam_area / pb 
    # samples_mjy_per_beam = 1e3 * samples_jy_per_beam # mJy/beam
    samples = total_samples / pb
    # Saving posterior samples
    print("Saving posterior samples...")
    # samples_jy_per_beam = total_samples
    samples_dir = os.path.join(path_params, f"{target_name}_{padding_mode}padding_{THIS_WORKER}_{args.s}.npz")
    np.savez(samples_dir, 
             samples = samples,
             observation = y.cpu().numpy().astype(np.float32),
             std_estimate = sigma_y.cpu().numpy().astype(np.float32),
             model_prediction = reconstructions,
             sampling_function = S.cpu().numpy().astype(np.float32), 
             primary_beam = pb,
             B = B, 
             C = C, 
             pad = pad, 
             padding_mode = padding_mode
            )
    print("Posterior samples saved !")
    if args.sanity_plot: 
            print('Plotting...')
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            yc = pixel_center_h 
            xc = pixel_center_w 
            Dy = yc-img_size//2
            Uy = yc+img_size//2
            Dx = xc-img_size//2
            Ux = xc+img_size//2
            
            sample = np.load(samples_dir)['samples'][-1] # just to make sure that we saved the right images.
            # norm_dirty = ImageNormalize((dirty_image/dirty_image.max()).squeeze(), vmin = 0, stretch = AsinhStretch(a=0.05))
            # norm_sample = ImageNormalize((sample/sample.max()).squeeze(), vmin = 0, stretch = AsinhStretch(a=0.05))
            
            nrows, ncols = 2, 3 
            fig, axs = plt.subplots(nrows, ncols, figsize = (ncols * 8, nrows * 8))
            # vmin = LogNorm
            # vmax = sample.max()
            ax = axs[0, 0]
            im = ax.imshow(dirty_image[Dy:Uy, Dx:Ux].cpu().numpy(), cmap = "magma", origin = "lower", norm = None)
            plt.colorbar(im, ax = ax, fraction = 0.046)
            ax.set_title("Dirty image")
            ax.axis("off")

            image = sample 
            norm = LogNorm(vmin = 1e-2, vmax = image.max(), clip = True)
            # norm = None
            ax = axs[0, 1]
            im = ax.imshow(image.squeeze(), cmap = "magma", origin = "lower", norm = norm)
            cbar = plt.colorbar(im, ax = ax, fraction = 0.046)
            ax.set_title("Posterior sample")
            ax.axis("off")
            cbar.set_label("mJy/beam")

            ax = axs[0, 2]
            im = ax.imshow(dirty_image.cpu().numpy().squeeze(), cmap = "magma", origin = "lower", norm = None)
            plt.colorbar(im, ax = ax, fraction = 0.046)
            ax.set_title("Full Dirty image")
            ax.axis('off')

            # Residuals
            nu_ref = len(y)//2 # since we are in vectorized representation. 
            residuals = ((y.cpu().numpy()-reconstructions) / sigma_y.cpu().numpy())[0].squeeze()
            residuals_grid = np.zeros(shape = (npix, npix), dtype = np.complex64)
            residuals_grid[S.cpu().numpy().astype(bool)] = (residuals[:nu_ref] + 1j * residuals[nu_ref:])

            # Dirty residuals 
            # norm = LogNorm(vmin = 1e-5, vmax = image.max(), clip = True)
            norm = None
            ax = axs[1, 0]
            image = compute_dirty_image(residuals_grid)
            im = ax.imshow(image, cmap = "bwr", origin = "lower", norm = norm, vmin = -0.5 * abs(image).max(), vmax = 0.5 * abs(image).max())
            cbar = plt.colorbar(im, ax = ax, fraction = 0.046)
            ax.set_title("Dirty residuals")

            # norm = LogNorm(vmin = 1e-5, vmax = image.max(), clip = True)
            norm = None
            D, U = compute_uv_window(residuals_grid)
            image = residuals_grid[D:U, D:U].real

            ax = axs[1, 1]
            im = ax.imshow(image, cmap = "bwr", origin = "lower", norm = norm, vmin = -1, vmax = 1)
            cbar = plt.colorbar(im, ax = ax, fraction = 0.046)
            ax.set_title("Residuals (real part)")
            
            
            image = residuals_grid[D:U, D:U].imag

            ax = axs[1, 2]
            im = ax.imshow(image, cmap = "bwr", origin = "lower", norm = norm, vmin = -1, vmax = 1)
            cbar = plt.colorbar(im, ax = ax, fraction = 0.046)
            ax.set_title("Residuals (imaginary part)")

            image_dir = os.path.join(path_params, "sanity_plot.jpeg")
            plt.savefig(image_dir, bbox_inches = "tight")     


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # Gridded npz dir + npz dir (converted MeasurementSet) + Output dir
    parser.add_argument("--gridded_npz_dir",    required = True,                        type = str,     help = "Path to the gridded visibilities.")
    parser.add_argument("--npz_dir",            required = True,                        type = str,     help = "Path to the .npz FOLDER holding information about the MeasurementSet.")
    parser.add_argument("--output_dir",         required = True,                        type = str,     help = "Path to the output folder (a few folders will be created to keep things organized).")
    
    # Grid mode 
    parser.add_argument("--grid",               required = False,   default = True,     type = bool,    help = "If True, generate posterior samples for every npz file in the directory.")
    # Data specs. \
    parser.add_argument("--img_size",           required = True,                        type = int,     help = "Image size (only supporting images with equal width and heights for now). The total number of pixels in the image should be img_size * img_size")
    parser.add_argument("--npix",               required = False,   default = 4097,     type = int,     help = "Number of cells in the Fourier grid")
    parser.add_argument("--include_pb",         required = False,   default = False,    type = bool,    help = "Whether to include or not the primary beam in the forward model. Defaults to False (= primary beam not included).")
    parser.add_argument("--s",                  required = False,   default = 0,           type = float,     help = "Image size (only supporting images with equal width and heights for now). The total number of pixels in the image should be img_size * img_size")
    parser.add_argument("--gpus_per_target",       required = False,   default = 1,        type = int,    help = "True to create a plot with posterior samples and the ground truth in the results directory (if testing the script put both debug_mode and sanity_plot to True)")

    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "euler",  type = str,     help = "Sampler used ('old_pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 10,       type = int,     help = "Total number of posterior samples to generate.")
    parser.add_argument("--batch_size",         required = False,   default = 20,       type = int,     help = "Number of posterior samples to generate per iteration (the code begins a loop if num_samples > batch_size).")
    parser.add_argument("--predictor",          required = False,   default = 4000,     type = int,     help = "Number of steps if sampler is 'euler'. Number of predictor steps if the sampler is 'pc'")
    parser.add_argument("--corrector",          required = False,   default = 1,        type = int,     help = "Number of corrector steps for the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2,     type = float,   help = "Snr parameter for PC sampling")
    parser.add_argument("--score_dir",          required = True,                        type = str,     help = "Path to the trained score model." )
    parser.add_argument("--tweedie",            required = False,   default = False,    type = bool,    help = "Sampler used ('old_pc' or 'euler')")
    parser.add_argument("--padding_mode",       required = False,   default = "zero",   type = str,     help = "Sampler used ('old_pc' or 'euler')")

    # For debugging and checking that everything works fine 
    parser.add_argument("--debug_mode",         required = False,   default = False,    type = bool,    help = "True to skip loops and debug")
    parser.add_argument("--sanity_plot",        required = False,   default = False,    type = bool,    help = "True to create a plot with posterior samples and the ground truth in the results directory (if testing the script put both debug_mode and sanity_plot to True)")

    
    args = parser.parse_args()
    main(args) 
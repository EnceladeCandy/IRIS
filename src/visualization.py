import matplotlib.pyplot as plt
from matplotlib import patches
import colorcet as cc
import numpy as np
import torch
import sys
from inference.fourier import ft, ift, ftshift, iftshift, flip
import astropy.units as units


# Units conversion: 
def arcsec_to_pixels(arcsec, pixel_scale):
        return arcsec / pixel_scale

def arcsec_to_degrees(arcsec):
    return (arcsec * units.arcsec).to(units.degree)

def compute_beam_area(bmin, bmaj):
    """
    Computes the beam area given the FWHM of the minor and major axis of the corresponding elliptical Gaussian. 

    Args:
        bmin (float): FWHM minor axis (units chosen by the user)
        bmaj (float): FWHM major axis (units chosen by the user)
    """
    return np.pi * bmin * bmaj / (4 * np.log(2))

def jy_beam_to_jy_per_arcsec2(img, bmin, bmaj): 
    """
    Computes the unit conversion from Jy/beam to Jy/arcsec^2

    Args:
        img (np.array): Image in Jy/beam to apply the unit conversion to. 
        bmin (float): FWHM of the minor axis of the beam (CLEAN/restoring beam).
        bmaj (float): FWHM of the major axis of the beam (CLEAN/restoring beam)
    """
    beam_area = compute_beam_area(bmin, bmaj)
    return img / beam_area

def jy_to_jy_per_arcsec2(img, pixel_scale): 
    """
    Computes the conversion from Jy to Jy/arcsec^2

    Args:
        img (np.array): Image in Jy 
        pixel_scale (float): Size of the pixel in arcsec
    """
    pixel_area = pixel_scale ** 2
    return img / pixel_area


# Plotting: 
def plot_chi2(ax, data, num_bins = 50, transparency = 0.5, color_bins = "midnightblue", label = None):
    """
    Plots the distribution of predicted chi squared sampled with an histogram 
    and the ground-truth chi squared distribution 
    """
    chi_samples = data[0]
    ax.hist(chi_samples, bins = num_bins, density = True, label = label, alpha = transparency, color = color_bins, histtype = "stepfield", lw = 3)
    # ax.hist(chi_samples, bins = num_bins, density = True, label = label, alpha = transparency, color = color_bins)
    ax.set(xlabel = "Degrees of freedom", ylabel = r"$\chi^2$")
    

def plot_tarp(ax, alpha, ecp, ecp_std, predictor, corrector, snr, factor_uncertainty = 3, title = True, color = "red", label = None, transparency = 0.5):
    """
    A helper function to plot a tarp test. 
    """
    print("Generating the coverage figure...")
    labels = [0., 0.2, 0.4, 0.6, 0.8, 1]
    ax.plot(alpha, ecp, color = color, label = label)
    ax.fill_between(alpha, ecp - factor_uncertainty * ecp_std, ecp + factor_uncertainty * ecp_std, alpha = transparency, color = color)
    ax.legend()
    ax.set_xlabel("Credibility Level")
    ax.set_ylabel("Expected Coverage Probability")
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.set_yticks(labels[1:])
    ax.set_yticklabels(labels[1:])
    ax.set(xlim = [0, 1], ylim = [0, 1])

    if title:
        ax.set_title(f"{int(predictor)} predictor steps, {int(corrector)} corrector steps \n snr = {snr}", y = 1.05)

def compute_median(x):
    """
    Computes the median of a batched input. 

    Args:
        x: Batched array (must be a numpy array)
    """
    return np.quantile(x, 0.5, axis = 0)

def compute_percentile_range(x):
    return np.quantile(x, 0.84, axis = 0) - np.quantile(x, 0.16, axis = 0)

def scientific_notation_latex(value):
    # Convert the number to scientific notation with two significant figures
    scientific_notation = f"{value:.3e}"
    
    # Split the scientific notation into the coefficient and exponent
    coefficient, exponent = scientific_notation.split('e')
    
    return coefficient, int(exponent.split('+')[-1])


def plot_grid_samples(images, N, title = None, cmap = "magma", norm = None, wspace = 0.1, hspace = 0.05): 
    """
    Plots a grid of images. The input

    Args:
        images: Must be of shape (B, C, H, W) (the channel can be ignored as well if necessary)
        N: Size of the grid
    """
    fig, axs = plt.subplots(N, N, figsize = (4 * N, 4 * N))
    
    k = 0
    for i, ax in enumerate(axs.flatten()):
        im = ax.imshow(images[k].squeeze(), cmap = cmap, origin = "lower", norm = norm)
        cbar = plt.colorbar(im, fraction = 0.0455)
        ax.axis("off") 
        k+=1

    if title: 
        fig.suptitle(title, y = 0.91, x = 0.5)
    plt.subplots_adjust(wspace = wspace, hspace = hspace)


# Plotting residuals
def compute_uv_window(vis_grid, edge = 50):
    """
    For plotting purposes, computes a square window where data lies in the gridded uv space (since most of the grid cells are null)

    Args:
        vis_grid (torch.tensor or np.ndarray): visibility grid
        edge: parameter to create a small space between the cell at highest frequency and the window edge
    Returns: 
    D, U = Start index and end index where most of the data is non zero in the vis_grid array

    Use: 
    When plotting next vis_grid, one can just use the array vis_grid[D:U, D:U]

    """
    npix = vis_grid.shape[-1]
    vis_grid = torch.tensor(vis_grid) # to be able to do this in numpy
    # Putting the non zero cells in a tensor
    nonzero_cells = torch.where(abs(vis_grid) > 0)

    # Getting the index of the cell the closest to the top edge
    idx_row_max = nonzero_cells[0].argmax()

    # Getting the index of the cell the closest to the right edge
    idx_col_max = nonzero_cells[1].argmax()

    # Origin at the center of the grid
    x_origin = npix // 2
    y_origin = npix // 2

    # Edges w.r.t the center of the grid
    x_edge = nonzero_cells[1][idx_col_max] - x_origin
    y_edge = nonzero_cells[0][idx_row_max] - y_origin

    # Picking the one corresponding to the biggest window to have all the visibilities.
    window = max(x_edge, y_edge) + edge
    D = npix // 2 - window
    U = npix // 2 + window

    return max(D, 0), min(U, npix) 

def compute_dirty_image(vis_gridded, norm = "backward"): 
    return flip(ftshift(ift(iftshift(vis_gridded), norm = norm)).real)


def plot_dirty_image(ax, vis_gridded, extent, norm = "backward", cmap = "magma", title = None): 
    npix = vis_gridded[-1]
    dirty_image = compute_dirty_image(vis_gridded, norm = norm)
    D =(npix - extent)//2
    U =(npix + extent)//2
    im = ax.imshow(dirty_image[D:U, D:U].cpu(), origin = "lower", cmap = cmap)  # flip needed due to CASA's convention  
    plt.colorbar(im, fraction = 0.046, label = "Jy/beam ?")
    ax.set_title(title)

def plot_uv_coverage(ax, u, v): 
    ax.scatter(u, v, s=0.5, rasterized=True, linewidths=0.0, c="r") 
    ax.set_aspect("equal")



def add_disk_annotations(ax, pixel_scale = 1, beam_params = (None, None, None), show_pixel_scale = True, show_beam = True, show_text = False,fontsize = 15, textcolor = "white"):
     # Calculate the position for the scale bar
    bar_x_start = 240
    bar_y_start = 15
    bar_width_pixels = 0.1 / pixel_scale # number of pixels for 0.1 arcsec
    bar_height_pixels = 2
    
    if show_pixel_scale:
        # Draw the scale bar
        rect = patches.Rectangle((bar_x_start - bar_width_pixels, bar_y_start), bar_width_pixels, bar_height_pixels, linewidth=1, edgecolor=textcolor, facecolor=textcolor)
        ax.add_patch(rect)
        if show_text:
            ax.text(bar_x_start - bar_width_pixels / 2, bar_y_start +15, r"$0.1''$",
                    color="white", ha='center', va='top', fontsize=fontsize-5)
        
    # Add the dirty beam ellipse shape:
    if show_beam: 
        beam_x, beam_y, position_angle = beam_params #(mas, mas, degrees)
        beam_width_pixels = arcsec_to_pixels(beam_x * 1e-3, pixel_scale = pixel_scale)
        beam_heigth_pixels = arcsec_to_pixels(beam_y * 1e-3, pixel_scale = pixel_scale)
        ellipse = patches.Ellipse((25, 25), width = beam_width_pixels, height = beam_heigth_pixels, angle = position_angle, edgecolor = textcolor, facecolor = textcolor)
        ax.add_patch(ellipse)

def plot_disk(ax, img, target_name, pixel_scale, cmap = "magma", 
              norm = None, fontsize = 15, textcolor = "white", show_beam = False, 
              beam_params = (None, None, None), show_pixel_scale = True, show_text = False, show_name = False): 
    im = ax.imshow(img, cmap = cmap, norm = norm, origin = "lower")
    if show_name:
        ax.annotate(target_name, (0.02, 0.92),  xycoords = "axes fraction", color=textcolor, fontsize=fontsize)  # Adjust fontsize if needed


    # Adding disk annotation (scale bar and Dirty beam area)
    add_disk_annotations(ax, 
                         pixel_scale = pixel_scale, 
                         beam_params = beam_params, 
                         show_pixel_scale = show_pixel_scale, 
                         show_beam = show_beam, 
                         show_text = show_text,
                         textcolor = textcolor)
    ax.axis("off")
    return im


def plot_dirty_image(ax, img, target_name, cmap = "magma", norm = None, fontsize = 15, textcolor = "white", vmin = -2, vmax = 2): 
    im = ax.imshow(img, cmap = cmap, norm = norm, origin = "lower", vmin = vmin, vmax = vmax)
    ax.annotate(target_name, (0.02, 0.92),  xycoords = "axes fraction", color=textcolor, fontsize=fontsize)  # Adjust fontsize if needed    
    ax.axis("off")
    return im

    #

def plot_residuals(ax, img, target_name, vmin = -3, vmax = 3, cmap = "bwr", norm = None, fontsize = 15, show_chi2 = False):
    im = ax.imshow(img, cmap = cmap, norm = norm, origin = "lower", vmin = vmin, vmax = vmax)
    ax.annotate(target_name, (0.02, 0.95),  xycoords = "axes fraction", color="k", fontsize=fontsize)  # Adjust fontsize if needed
    ax.axis("off")

    if show_chi2: 
        chi2 = np.sum(img ** 2)
        m = (abs(img) > 0).sum() 
        annotation_text = r"$\begin{aligned}m & = %.0f \\ \chi^2 & = %.0f\end{aligned}$" % (m, chi2)
        ax.annotate(annotation_text, (0.02, 0.85), xycoords="axes fraction", color="k", fontsize=fontsize)
    return im
    
def plot_stats(axs, median, pr, residuals, pixel_scale = 1, beam_params = None, target_name = None, dirty_res = None, show_pixel_scale = True, 
             show_dirty_res = False, show_beam = False, show_chi2 = False, show_title = False,
             show_name = False, show_text = False, med_params = ("magma", None), 
             pr_params = (cc.m_gray, None), 
             res_params = ("bwr", None),
             dirty_res_params = ("bwr", None, -2, 2),
             vmin_res = -1, vmax_res = 1, fontsize = 15, title_fontsize = 25, textcolor = "white", norm = None):
    """
    Plots a row subplot of [median, percentile range, residuals real part, residuals imaginary part]

    Args:
        ax : Must be the row of a matplotlib subplot
        posterior_samples : array with two posterior samples
        median: Median computed over a lot of posterior samples
        pr: Percentile range computed over a lot of posterior samples
        cmap (str, optional): Colormap for posterior samples and median. Defaults to "magma".
        cmap_std (str, optional): Colormap for the percentile range. Defaults to cc.m_gray.
    """
    for i, ax in enumerate(axs): 
        axs[i].axis("off")
    
    title_height = 1.05
    
    if show_name: 
        axs[0].annotate(target_name, (0.02, 0.92),  xycoords = "axes fraction", color = textcolor, fontsize=fontsize)  # Adjust fontsize if needed

    # Plotting titles
    if show_title:
        axs[0].set_title("Median\n" + r"$50$th quantile", fontsize = title_fontsize, y = title_height)
        axs[1].set_title("Percentile range\n" + r"$(84\%-16\%)$ CI", fontsize = title_fontsize, y = title_height)
        axs[2].set_title("Residuals (real part)\n" + r"$\mathrm{Re}(\tilde{\Sigma})^{-1/2}\mathrm{Re}\big(\tilde{\mathcal{V}}-\tilde{A}\mathbf{x} \big)$", fontsize = title_fontsize, y = title_height)
        axs[3].set_title("Residuals (imaginary part)\n" + r"$\mathrm{Im}(\tilde{\Sigma})^{-1/2}\mathrm{Im}\big(\tilde{\mathcal{V}}-\tilde{A}\mathbf{x} \big)$", fontsize = title_fontsize, y = title_height)
        if show_dirty_res: 
            axs[4].set_title("Dirty image residuals \n" + r"$\mathcal{F}^{-1}\big[\tilde{\Sigma}^{-1/2}\big(\tilde{\mathcal{V}}-\tilde{A}\mathbf{x}\big)\big]$", fontsize = title_fontsize, y = title_height)

    
    add_disk_annotations(axs[0], 
                         pixel_scale = pixel_scale,
                         beam_params = beam_params, 
                         show_pixel_scale = show_pixel_scale, 
                         show_beam = show_beam,
                         show_text = show_text,
                         fontsize = fontsize, 
                         textcolor = textcolor
                         )
    cmap_med, norm_med= med_params
    cmap_pr, norm_pr = pr_params
    cmap_res, norm_res = res_params
    img_med = axs[0].imshow(median,         cmap = cmap_med,  origin = "lower", norm = norm_med)
    img_pr  = axs[1].imshow(pr,             cmap = cmap_pr,   origin = "lower", norm = norm_pr)
    img_res = axs[2].imshow(residuals.real, cmap = cmap_res,  origin = "lower", norm = norm_res, vmin = vmin_res, vmax = vmax_res)
    img_res = axs[3].imshow(residuals.imag, cmap = cmap_res,  origin = "lower", norm = norm_res, vmin = vmin_res, vmax = vmax_res)

    if show_chi2: 
        chi2_real = np.sum(residuals.real ** 2)
        chi2_imag = np.sum(residuals.imag ** 2)
        m = (abs(residuals.real) > 0).sum() 

        ax = axs[2]
        m_coeff, m_exp = scientific_notation_latex(m)
        chi2_real_coeff, chi2_real_exp = scientific_notation_latex(int(chi2_real))
        annotation_text = r"$\begin{aligned}m & \sim %.3f \times 10^{%.0f} \\ \chi^2 & \sim %.3f \times 10^{%.0f} \end{aligned}$"% (float(m_coeff), float(m_exp), float(chi2_real_coeff), float(chi2_real_exp))
        ax.annotate(annotation_text, (0.02, 0.9), xycoords="axes fraction", color="k", fontsize=fontsize, bbox=dict(boxstyle="square", edgecolor='white', facecolor='white', alpha=0.5))
        
        ax = axs[3]
        chi2_imag_coeff, chi2_imag_exp = scientific_notation_latex(int(chi2_imag))
        annotation_text = r"$\begin{aligned}m & \sim %.3f \times 10^{%.0f} \\ \chi^2 & \sim %.3f \times 10^{%.0f} \end{aligned}$"% (float(m_coeff), float(m_exp), float(chi2_imag_coeff), float(chi2_imag_exp))
        ax.annotate(annotation_text, (0.02, 0.9), xycoords="axes fraction", color="k", fontsize=fontsize, bbox=dict(boxstyle="square", edgecolor='white', facecolor='white', alpha=0.5))
        

    if show_dirty_res: 
        img_size = median.shape[-1]
        npix = dirty_res.shape[-1]
        cmap_res, norm_res, vmin, vmax = dirty_res_params
        img_dirty_res = axs[4].imshow(dirty_res, cmap = "bwr", vmin = vmin, vmax = vmax, origin = "lower")
        return (img_med, img_pr, img_res, img_dirty_res)
        
    return (img_med, img_pr, img_res)


def get_dsharp_name(target):
    target_concat = target.split("_")[0]
    prefixes = ["HD", "AS", "SR", "Sz", "DoAr", "Elias", "WaOph", "WSB"]
    suffix = "Lup"


    for prefix in prefixes:
        if prefix in target_concat:
            length = len(prefix)
            name = target_concat[:length] + " " + target_concat[length:]
            return name
        elif suffix in target_concat:
            length = len(target_concat) - len(suffix)
            name = target_concat[:length] + " " + target_concat[length:]
            return name
    else: 
        name = target_concat

    return name

    
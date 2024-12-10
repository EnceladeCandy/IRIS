import torch 
import numpy as np
from torch.func import vmap, grad
import sys 
sys.path.append("../src")
from inference.posterior_sampling import sigma, mu
from inference.fourier import ft, ift, ftshift, iftshift, complex_to_real, flip

from astropy.constants import c 
import astropy.units as u

# Useful functions for the CASA primary beam.
import numpy as np
import scipy.constants
from scipy.special import j1, jn

def link_function(x, B, C): 
    """Mapping from the model space (where the score model is trained) to the image space 
    (with meaningful physical units)

    Args:
        x (array, float...): image to map 
        B (float): factor
        C (float): constant

    Returns:
        x in image space
    """
    return B*x + C

def noise_padding_dev(x, pad, sigma):
    """Padding with realizations of noise of the same temperature of the current diffusion step

    Args:
        x (torch.Tensor): ground-truth 
        pad (int): amount of pixels needed to pad x
        sigma (torch.Tensor): std of the gaussian distribution for the noise pad around the model

    Returns:
        out (torch.Tensor): noise padded version of the input
    """

    # To manage batched input
    if len(x.shape)<4:
        _, H, W = x.shape
    else: 
        _, _, H, W = x.shape
    out = torch.nn.functional.pad(x, (pad, pad, pad, pad)) 
    # Create a mask for padding region
    mask = torch.ones_like(out)
    mask[pad:pad + H, pad:pad+W] = 0.
    
    # Noise pad around the model
    z = torch.randn_like(out) * sigma
    out = out + z * mask
    return out

def zero_padding(x, pad):
    """
    Zero-pad a 2D input x using the padding given as argument. 

    Args:
        x (torch.Tensor): 2D image. 
        pad (Tuple): amount of padding in each direction around x. In PyTorch's convention, (pad_left, pad_right, pad_bot, pad_top) 

    Returns:
        torch.Tensor: zero-padded version of x. 
    """
    out = torch.nn.functional.pad(x, pad)
    return out

def gaussian_pb(diameter=12, freq=240*1e9, shape=(500, 500), pixel_scale=0.004, device='cpu'): 
    """
    Creates a Gaussian primary beam to simulate the sensitivity of the antennas

    Args:
        diameter (int, optional): Antenna diameter (meters). Defaults to 12 meters for ALMA.
        freq (float, optional): Frequency of the observation (Hz). Defaults to 240 GHZ.
        shape (tuple, optional): Image shape for the primary beam. Defaults to (500, 500).
        pixel_scale (float, optional): Pixel scale in image space (arcsec). Defaults to 0.004 arcsec.
        device (str, optional): Torch device, either 'cpu' or 'cuda' (if running on GPU). Defaults to 'cpu'.

    Returns:
        (pb, fwhm): Tuple of the primary beam and the FWHM of the antennas.
    """
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq
    fwhm = 1.02 * wavelength / diameter * (180 / torch.pi) * (3600)
    half_fov = pixel_scale * shape[0] / 2
    # Grid for PB
    x = torch.linspace(-half_fov, half_fov, shape[0], device=device)
    y = torch.linspace(-half_fov, half_fov, shape[1], device=device)
    x, y = torch.meshgrid(x, y, indexing='xy')
    # Gaussian PB parameters
    mean = torch.tensor([0.0, 0.0], device=device)  # Mean (center) of the Gaussian
    std = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0, device=device))))
    covariance_matrix = torch.tensor([[std ** 2, 0], [0, std ** 2]], device=device)  # Covariance matrix
    # 2-D Gaussian PB
    x_y = torch.stack([x.ravel(), y.ravel()], dim=1)
    inv_covariance_matrix = torch.inverse(covariance_matrix)
    diff = x_y - mean
    pb = (
        1 / (2 * torch.pi * torch.sqrt(torch.det(covariance_matrix)))
    ) * torch.exp(-0.5 * torch.sum(diff @ inv_covariance_matrix * diff, dim=1))
    pb = pb.view(x.shape)
    return (pb/pb.max(), fwhm)

def casa_airy_beam(l,m,freq_chan,dish_diameter, blockage_diameter, ipower, max_rad_1GHz, n_sample=10000, device = "cpu"):
    """
    Airy disk function for the primary beam as implemented by CASA     
    Parameters
    ----------
    l: float, radians
        Coordinate of a point on the image plane (the synthesis projected ascension and declination).
    m: float, radians
        Coordinate of a point on the image plane (the synthesis projected ascension and declination).
    freq_chan: float, Hz
        Frequency.
    dish_diameter: float, meters
        The diameter of the dish.
    blockage_diameter: float, meters
        The central blockage of the dish.
    ipower: int
        ipower = 1 single dish response.
        ipower = 2 baseline response for identical dishes.
    max_rad_1GHz: float, radians
        The max radius from which to sample scaled to 1 GHz.
        This value can be found in sirius_data.dish_models_1d.airy_disk.
        For example the Alma dish model (sirius_data.dish_models_1d.airy_disk import alma)
        is alma = {'func': 'airy', 'dish_diam': 10.7, 'blockage_diam': 0.75, 'max_rad_1GHz': 0.03113667385557884}.
    n_sample=10000
        The sampling used in CASA for PB math.
    Returns
    -------
    val : float
        The dish response.
    """
    casa_twiddle = (180*7.016*c.value)/((np.pi**2)*(10**9)*1.566*24.5) # 0.9998277835716939

    r_max = max_rad_1GHz/(freq_chan/10**9)
    # print(r_max)
    k = (2*np.pi*freq_chan)/c
    aperture = dish_diameter/2

    if n_sample is not None:
        r = np.sqrt(l**2 + m**2)
        r_inc = ((r_max)/(n_sample-1))
        r = (int(r/r_inc)*r_inc)*aperture*k #Int rounding instead of r = (int(np.floor(r/r_inc + 0.5))*r_inc)*aperture*k
        r = r*casa_twiddle
    else:
        r = np.arcsin(np.sqrt(l**2 + m**2)*k*aperture)
        
    if (r != 0):
        if blockage_diameter==0.0:
            return torch.tensor((2.0*j1(r)/r)**ipower).to(device)
        else:
            area_ratio = (dish_diameter/blockage_diameter)**2
            length_ratio = (dish_diameter/blockage_diameter)
            return torch.tensor(((area_ratio * 2.0 * j1(r)/r   - 2.0 * j1(r * length_ratio)/(r * length_ratio) )/(area_ratio - 1.0))**ipower).to(device)
    else:
        return 1

def noise_padding(x, pad, sigma):
    """
    Noise pad a 2D input x using the padding and the sigma given as argument. The sigma corresponds to the std of the perturbation's kernel 
    Gaussian noise in the padded region.

    Args:
        x (torch.Tensor): 2D input
        pad (Tuple): amount of padding in each direction around x. In PyTorch's convention, (pad_left, pad_right, pad_bot, pad_top) 
        sigma (function): Function computing the std of the perturbation kernel associated to the SDE used. Accessible through the trained score-based model. 

    Returns:
        torch.Tensor: noise-padded version of x. 
    """
    _, H, W = x.shape
    out = torch.nn.functional.pad(x, pad) 
    # Create a mask for padding region
    mask = torch.ones_like(out)
    pad_l, pad_r, pad_b, pad_t = pad
    mask[pad_t:pad_t + H, pad_l:pad_l+W] = 0.
    # Noise pad around the model
    z = torch.randn_like(out) * sigma
    out = out + z * mask
    return out


def old_forward_model(t, x, score_model, model_parameters):
    pad = model_parameters[-2]
    x_padded = noise_padding(x, pad=pad, sigma=sigma(t, score_model))
    fft_result = ft(iftshift(x_padded)).flatten() 
    visibilities_result = fft_result.real + 1j*fft_result.imag
    return torch.cat((visibilities_result.real, visibilities_result.imag))[S]


def model(t, x, score_model, model_parameters): 
    """Apply the physical model associated to a simplified version of a radiotelescope's measurement process. 
    For stability reasons and to increase the resolution in Fourier space, noise padding was used. 

    Args:
        t (torch.Tensor): temperature in the sampling procedure of diffusion models.
        x (torch.Tensor): ground-truth 
        score_model (torch.Tensor): trained score-based model (= the score of a prior)
        model_parameters (Tuple): list of parameters for the model (sampling_function, B, C)
          - index 0: sampling function (mask selecting the measured visibilities in Fourier space, must have a shape (H, W) where H and W are the height
            and width of the padded image respectively)
          - index 1 and index 2: B and C, the link_function parameters (see function link_function)

    Returns:
        y_hat (torch.Tensor): model prediction
    """
      
    
    sampling_function, pb, B, C, pad, padding_mode = model_parameters
    npix = sampling_function.shape[-1]

    # Two different modes of padding ('noise' uses the SDE perturbation kernel to pad the image with Gaussian noise)
    if padding_mode == "noise":
        padding = lambda x, pad: noise_padding(x, pad, sigma = sigma(t, score_model))
    elif padding_mode == "zero": 
        padding = zero_padding
    x = link_function(x, B=B, C=C)  # Function mapping x to a region where the SBM has been trained 
    x_padded = flip(pb * padding(x, pad)) 
    vis_full = ft(iftshift(x_padded), norm = "backward") # iftshift to place the DC component at (0,0), as expected by torch.fft.fft2
    vis_sampled = ftshift(vis_full.squeeze())[sampling_function]
    y_hat = complex_to_real(vis_sampled) # complex to vectorized real representation.
    return y_hat

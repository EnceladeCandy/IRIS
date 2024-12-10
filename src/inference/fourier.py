import torch 
import numpy as np

def complex_to_real(z):
    """
    Converts a complex tensor into a vectorized real representation where the real
    part and the imaginary part are concatenated 

    Args:
        z (torch.Tensor): complex tensor of dimension D 

    Returns:
        real tensor of dimension 2*D
    """
    return torch.cat([z.real, z.imag], axis = -1)   


def real_to_complex(y): 
    """
    Converts a real vectorized representation of a tensor into its complex equivalent (inverse function of `complex_to_real`)
    
     Args:
        y (torch.Tensor): real tensor of dimension 2 * D 

    Returns:
        complex tensor of dimension 2*D
    """
    length = len(y)
    return y[:length//2] + y[length//2:] * 1j

def ft(x, norm = "ortho"): 
    """Compute the FFT 2D for x (default to orthonormal FFT).
    Note: In torch's convention, the DC component is expected to be at (0,0) before being passed to the fft. 

    Args:
        x (array): Two-dimensionnal numpy array or torch tensor

    Returns:
        array : FFT 2D of x (computed over the last two dimensions of x, so a batched FFT is possible)
    """
    if type(x) == np.ndarray: 
        return np.fft.fft2(x, norm = norm)
    
    elif type(x) == torch.Tensor: 
        return torch.fft.fft2(x, norm = norm)
    

def ift(x, norm = "ortho"): 
    """Compute the orthonormal IFT 2D for x. 
    Note: In torch's convention, the DC component is expected to be at (N/2, N/2) (i.e. the center of the image) before being passed to the ift. 

    Args:
        x (array): Two-dimensionnal numpy array or torch tensor

    Returns:
        array : orthonormal FFT 2D of x (computed over the last two dimensions of x, so a batched FFT is possible)
    """
    if type(x) == np.ndarray: 
        return np.fft.ifft2(x, norm = norm)
    
    if type(x) == torch.Tensor: 
        return torch.fft.ifft2(x, norm = norm)
    
def ftshift(x):
    """
    Places the DC component of the input at the Nyquist Frequency (i.e. the center of the image for a square image). 
    Note: For even length inputs, fftshift and iftshift are equivalent. 
    """
    if type(x) == np.ndarray: 
        return np.fft.fftshift(x)
    
    if type(x) == torch.Tensor: 
        return torch.fft.fftshift(x)
    
def iftshift(x):
    """
    Places the DC component at the zero-component of the image. 
    """
    if type(x) == np.ndarray: 
        return np.fft.ifftshift(x)
    
    if type(x) == torch.Tensor: 
        return torch.fft.ifftshift(x)

def flip(x):
    """
    Flips the input column-wise (matching the CASA convention). 

    Args:
        x (torch.Tensor): 2D input

    Returns:
        torch.Tensor: Flipped version of x. 
    """
    if type(x) == np.ndarray: 
        return x[:, ::-1] 
    
    if type(x) == torch.Tensor: 
        return torch.flip(x, dims = [-1])
   

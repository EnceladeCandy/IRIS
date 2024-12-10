from score_models import ScoreModel
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from torch.func import vmap, grad 
from tqdm import tqdm



device = "cuda" if torch.cuda.is_available() else "cpu"

def sigma(t, score_prior): 
    return score_prior.sde.sigma(t)

def mu(t, score_prior): 
    return score_prior.sde.marginal_prob_scalars(t)[0]

def drift_fn(t, x, score_prior): 
    return score_prior.sde.drift(t, x)

def g(t, x, score_prior): 
    return score_prior.sde.diffusion(t, x)

def identity_model(t, x): 
    return x

def log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters):
    """
    Computes the log-likelihood of a gaussian distribution for the linear model corresponding to the `model` function.
    A diagonal approximation is made for the covariance matrix.
    Arguments: 
        y = processed gridded visibilities (real part and imaginary part concatenated)
        x = sky brightness 
        t = diffusion temperature
        A = linear model (sampling function and FT)  
    
    Returns: 
        log-likelihood of a gaussian distribution
    """ 
    y_hat = forward_model(t, x, score_model, model_parameters)

    # 2D DFT contribution
    Gamma_diag = torch.ones_like(y, device = y.device)/2 # Make sure to use the same notation as in the paper. 

    # Primary beam contribution:
    sampling_function, pb, B, _, _, _, = model_parameters
    npix = sampling_function.shape[-1]
    pb = model_parameters[1]
    
    # scale_factor = torch.where(t[0]>0.5, torch.tensor(1e-6).expand_as(t).to(device), torch.tensor(B).expand_as(t).to(device))
    
    # print(torch.where(pseudo_t>0.5, scale_factor ** 2))
    Gamma_diag *= torch.cat([pb[sampling_function], pb[sampling_function]]) ** 2  * B ** 2 * npix ** 2 # npix factor comes from fft backward
    var = sigma(t, score_model) **2 * Gamma_diag + mu(t, score_model)**2 * sigma_y**2 
    res = (mu(t, score_model) * y - y_hat) ** 2 / var
    log_prob = -0.5 * torch.sum(res)
    return log_prob

def score_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters):
    return vmap(grad(lambda x, t: log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters)), randomness = "different")(x, t)

def score_posterior(t, x, y, sigma_y, forward_model, score_model, score_likelihood, model_parameters, tweedie, guidance_factor = 1.):
    if tweedie: 
        sigma_t = sigma(t, score_model)[0].item() # During sampling every sample is evaluated at the same temperature so there's no issue with this
        mu_t = mu(t, score_model)[0].item()
        tweedie_x = (x + sigma_t ** 2 * score_model.score(t, x)) / mu_t
    else: 
        tweedie_x = x
    score_prior = score_model.score(t, x)
    score_lh = guidance_factor * score_likelihood(t, tweedie_x, y, sigma_y, forward_model=forward_model, score_model=score_model, model_parameters = model_parameters)
    return score_prior + score_lh

def euler_sampler(y, sigma_y, forward_model, score_model, score_likelihood, model_parameters, num_samples, num_steps,  tweedie = False, keep_chain=False, debug_mode=False, img_size = (64, 64), guidance_factor = 1.):
    """
    Discretization of the Euler-Maruyama sampler 

    Args:
        y (array): Observation
        sigma_y (float): estimated standard deviation of the gaussian noise in the observation (or ground-truth std if working with simulations)
        forward_model (function): physical model mapping a ground-truth x to a model \hat{y} 
        score_model (function): Trained score-based model playing the role of a prior 
        score_likelihood (function): see function score_likelihood 
        model_parameters (tuple): parameters of the function score_likelihood
        num_samples (int): number of samples to generate
        num_steps (int): number of steps during the sampling procedure
        tweedie (bool, optional): To enable a correction of the score of the posterior with Tweedie's formula. Defaults to False.
        keep_chain (bool, optional): To analyze possible anomalies that may occur during the sampling procedure. Defaults to False.
        debug_mode (bool, optional): Runs the loop for 20 iterations to evaluate time required for more iterations. Defaults to False.
        img_size (tuple, optional): image size of the ground-truth x. Defaults to (64, 64) (= simulation)

    Returns:
        array: posterior samples shape = (num_samples, 1, img_size, img_size) 
    """
    t = torch.ones(size = (num_samples,1)).to(device)
    sigma_max = sigma(t, score_model)[0]
    x = sigma_max * torch.randn([num_samples, 1, *img_size]).to(device)
    dt = -1/num_steps 
    
    chain = []
    with torch.no_grad(): 
        for i in (pbar := tqdm(range(num_steps - 1))):
            pbar.set_description(f"t = {t[0].item():.2f} | scale ~ {x.std():.2e} | sigma(t) = {sigma(t, score_model)[0].item():.2e} | mu(t) = {mu(t, score_model)[0].item():.2e}")
            z = torch.randn_like(x).to(device)
            gradient =  score_posterior(t, x, y, sigma_y, forward_model, score_model, score_likelihood, model_parameters, tweedie = tweedie, guidance_factor = guidance_factor)
            drift = drift_fn(t, x, score_model)
            diffusion = g(t, x, score_model)
            x_mean  = x + drift * dt - diffusion ** 2 * gradient * dt
            noise = diffusion * (-dt) ** 0.5 * z
            x = x_mean + noise
            t += dt

            if t[0].item()<1e-6:
                break

            if torch.isnan(x).any().item(): 
                print("Nans appearing")
                break
            if keep_chain: 
                chain.append(x.cpu().numpy())

            if debug_mode and i==20:
                    break

    if keep_chain: 
        return x_mean, chain
     
    else: 
        return x_mean 
      

def pc_sampler(y, sigma_y, forward_model, score_model, score_likelihood, model_parameters, num_samples, pc_params, tweedie = False, keep_chain = False, debug_mode = False, img_size = (64, 64)): 
    """
    Implementation of the predictor corrector sampler. 

    Args:
       y (array): Observation
        sigma_y (float): estimated standard deviation of the gaussian noise in the observation (or ground-truth std if working with simulations)
        forward_model (function): physical model mapping a ground-truth x to a model \hat{y} 
        score_model (function): Trained score-based model (plays the role of a prior).
        score_likelihood (function): score of the likelihood. To sample directly from the prior, put score_likelihood=None
        model_parameters (tuple): parameters of the forward model 
        num_samples (int): number of samples to generate
        pc_params (Tuple): Must respect the format (predictor, corrector, snr). 
        tweedie (bool, optional): To enable a correction of the score of the posterior with Tweedie's formula. Defaults to False.
        keep_chain (bool, optional): To analyze possible anomalies that may occur during the sampling procedure. Defaults to False.
        debug_mode (bool, optional): Runs the loop for 20 iterations to evaluate time required for more iterations. Defaults to False.
        img_size (tuple, optional): image size of the ground-truth x. Defaults to (64, 64) (= simulation)

    Returns:
        _type_: _description_
    """

    
    pred_steps, corr_steps, snr = pc_params
    t = torch.ones(size = (num_samples,1)).to(device)
    sigma_max = sigma(t, score_model)[0]
    x = sigma_max * torch.randn([num_samples, 1, *img_size]).to(device)
    dt = -1/pred_steps 

    chain = []

    if score_likelihood is None: 
        print("score_likelihood arg is None. Sampling directly from the learned prior.")
    with torch.no_grad(): 
        for i in (pbar :=tqdm(range(pred_steps-1))): 
            pbar.set_description(f"t = {t[0].item():.2f} | scale ~ {x.std():.2e} | sigma(t) = {sigma(t, score_model)[0].item():.2e} | mu(t) = {mu(t, score_model)[0].item():.2e}")
            
            # Sampling from the prior if no likelihood specified
            if score_likelihood is None:
                gradient = score_model.score(t, x)
            else:
                gradient =  score_posterior(t, x, y, sigma_y, forward_model, score_model, score_likelihood, model_parameters, tweedie = tweedie)
            
            # Corrector step: (Only if we are not at 0 temperature )
            for _ in range(corr_steps): 
                z = torch.randn_like(x)
                epsilon =  (snr * sigma(t, score_model)[0].item()) ** 2 
                x = x + epsilon * gradient + (2 * epsilon) ** 0.5 * z 

            # Predictor step: 
            z = torch.randn_like(x).to(device)
            
            if score_likelihood is None: 
                gradient = score_model.score(t, x)
            else: 
                gradient =  score_posterior(t, x, y, sigma_y, forward_model, score_model, score_likelihood, model_parameters, tweedie = tweedie)
            drift = drift_fn(t, x, score_model)
            diffusion = g(t, x, score_model)
            x_mean = x + drift * dt - diffusion**2 * gradient * dt  
            noise = diffusion * (-dt) ** 0.5 * z
            x = x_mean + noise
            t += dt

            # To avoid numerical instabilities due to vanishing variances
            if t[0].item()<1e-6:
                break

            if torch.isnan(x).any().item(): 
                print("Nans appearing, stopping sampling...")
                break

            if debug_mode and i==20:
                break

            if keep_chain: 
                chain.append(x.cpu().numpy())

        if keep_chain: 
            return x_mean, chain 
        else: 
            return x_mean  
# IRIS: A Bayesian approach for Image Reconstruction in Radio Interferometry with expressive Score-Based priors. 
<p align="center">
    <img src="assets/gif_dsharp.gif" alt="Animation" width="700">
</p>

This repository provides Python routines and tutorial notebooks used to generate the results presented in THIS PAPER. This work involves performing Bayesian inference on protoplanetary disk measurements from the DSHARP survey conducted by ALMA. The current code is somewhat experimental, so occasional bugs may be present. 


## Getting started
The packages needed for our pipeline are in the `requirements.txt` file. They can be installed by running the command
```shell
pip install -r requirements.txt
```
Due to some dependency issues, we separated the virtual environment needed for the gridding operation (requiring `casatasks` and `casatools`) and the one needed to perform the inference. 

## Data
The protoplanetary disks visibility data is freely available at the [DSHARP data release](https://almascience.eso.org/almadata/lp/DSHARP/). 
The results presented in this work are available in a [Zenodo dataset](https://zenodo.org/records/14407285).

## Tutorials
In the tutorials folder, we go through the different steps of the approach and the different things you need to know to use our code. You will find 4 tutorials: 
1) `1_gridding.ipynb`, which performs the gridding operation for the RU Lup protoplanetary disk. 
2) `2_inference_dsharp.ipynb`, which performs the inference for the RU Lup protoplanetary disk using one of our score models 
3) `3_zenodo_data.ipynb`, which explains how the zenodo datasets is related to our code. 
4) `4_inference_with_SBM_in_depth.ipynb`, where we explain our approach with more details on a toy problem.  

The notebooks can be completed independently and are designed to enable direct data downloads from Zenodo. 

## LICENSE
If you use this repository for your research, please consider citing us. 

This package is licensed under the MIT License. 



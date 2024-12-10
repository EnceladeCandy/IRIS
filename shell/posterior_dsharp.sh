#!/bin/bash
#SBATCH --tasks=1
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G         # memory per node
#SBATCH --time=00-03:00   # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=DSHARP_VP_SKIRT_first
module load python
source $HOME/diffusion/bin/activate

SCRIPT_DIR=/home/noedia/projects/def-lplevass/noedia/bayesian_imaging_radio/bayesian-imaging-radio/scripts
VP_SKIRT=/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_vp_skirt_z_256_230813225243
VE_PROBES=/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_ve_probes_z_256_230926020329
VP_PROBES=/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_vp_probes_z_256_230824141341

GRIDDED_DIR=/home/noedia/scratch/bayesian_imaging_radio/dsharp_gridded_final/HD143006_continuum_4097_gridded_sinc.npz
NPZ_DIR=/home/noedia/projects/def-lplevass/noedia/bayesian_imaging_radio/data/dsharp_npz/
RESULTS_DIR=/home/noedia/scratch/bayesian_imaging_radio/dsharp_results

python $SCRIPT_DIR/inference_dsharp.py \
    --gridded_npz_dir=$GRIDDED_DIR \
    --npz_dir=$NPZ_DIR \
    --output_dir=$RESULTS_DIR \
    --img_size=256\
    --npix=4097\
    --sampler=euler\
    --num_samples=25\
    --batch_size=8\
    --predictor=4000\
    --corrector=100\
    --snr=0.1\
    --score_dir=$VP_SKIRT \
    --sanity_plot=True\
    --debug_mode=\
    --grid=\
    --padding_mode='zero'\
    --gpus_per_target=10 \
    --s=100

    
    
    
    

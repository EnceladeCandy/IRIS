#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G        # memory per node
#SBATCH --time=00-02:59   # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Gridding_DSHARP
module load python
source $HOME/gingakei/bin/activate

TARGETS_DIR=/home/noedia/projects/def-lplevass/noedia/bayesian_imaging_radio/data/dsharp_npz
RESULTS_DIR=/home/noedia/scratch/bayesian_imaging_radio/dsharp_gridded_fixed
SCRIPT_DIR=/home/noedia/projects/def-lplevass/noedia/bayesian_imaging_radio/bayesian-imaging-radio/scripts

python $SCRIPT_DIR/generate_gridded_vis.py \
    --npz_dir=$TARGETS_DIR/Elias27_continuum.npz \
    --output_dir=$RESULTS_DIR \
    --npix=4097\
    --window_function="sinc"\
    --img_size=256\
    --target_name=AS205_binary\
    --experiment_name=binary

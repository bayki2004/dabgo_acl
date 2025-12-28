#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/kbayraktar/net_scratch/training_ml/dabgo_acl/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/kbayraktar/net_scratch/training_ml/dabgo_acl/jobs/%j.err # where to store error messages
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=tikgpu[02-06],tikgpu10
#CommentSBATCH --account=tik-highmem
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'




ETH_USERNAME=kbayraktar
PROJECT_NAME=dabgo_acl
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/training_ml/${PROJECT_NAME}
CONDA_ENVIRONMENT=base
mkdir -p ${DIRECTORY}/jobs
#TODO: change your ETH USERNAME and other stuff from above according + in the #SBATCH output and error the path needs to be double checked!

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"


[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

# Execute your code

NAMES=(
  "1987 Manchester City Council election"
)
MODES=(
  "descent"
  "ascent"
)

SOURCE="Self-Written"

for NAME in "${NAMES[@]}"; do
  for MODE in "${MODES[@]}"; do
    echo "Computing losses for ${NAME} in ${MODE} mode"

    if [ "${MODE}" == "descent" ]; then
      STEPS=100
    else
      STEPS=100
    fi

    python3 loss_computation.py \
      --model_name "${NAME}" \
      --mode "${MODE}" \
      --num_workers 4 \
      --batch_size 2 \
      --source "${SOURCE}" \
      --steps "${STEPS}" \

    echo "Losses computed for ${NAME} in ${MODE} mode"
  done
done

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0


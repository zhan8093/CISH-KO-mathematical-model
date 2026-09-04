#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=500gb
#SBATCH --constraint=milan
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhan8093@umn.edu
#SBATCH --job-name=CISH_KO
#SBATCH -A lede0024
#SBATCH -p ag2tb
#SBATCH -o julia_output_%j.txt
#SBATCH -e julia_error_%j.txt

# Load Julia module
module load julia

# Change to working directory
cd /projects/standard/lede0024/zhan8093/CISH_KO/exp7_8

# Set Julia threads to use all 128 cores
export JULIA_NUM_THREADS=50

# Print job information
echo "Job started on $(hostname) at $(date)"
echo "Running with $JULIA_NUM_THREADS Julia threads"

# Run Julia optimization
julia main_optimized.jl

echo "Job finished at $(date)"
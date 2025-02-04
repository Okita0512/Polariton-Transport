#!/bin/bash
#SBATCH -p action
#SBATCH -o output.log
#SBATCH --mem=1GB
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=Transport

time python3 temp-dep.py

#!/bin/bash -l
                                
conda activate root_env
                               
echo "Running environment at ${CONDA_PREFIX}"
cd /data/cmsdaq/DimensionBench/Arrays/

python3 runAll_Galaxy_Array.py --inputdir ArrayData

conda deactivate
cd ~/

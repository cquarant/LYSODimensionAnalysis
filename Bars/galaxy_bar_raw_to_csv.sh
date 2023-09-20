#!/bin/bash -l
                                
conda activate root_env
                               
echo "Running environment at ${CONDA_PREFIX}"
cd /data/cmsdaq/DimensionBench/Bars/

python3 runAll_Galaxy_Bar.py --inputdir BarData

conda deactivate
cd ~/

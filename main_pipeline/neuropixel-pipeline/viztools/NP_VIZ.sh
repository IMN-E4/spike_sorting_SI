#!/bin/bash

# activate environment for visualization tools
echo "Activating environment"
source /home/eduarda/.virtualenvs/viz_env/bin/activate

echo "Opening tool"
python /home/eduarda/python-related/github-repos/spike_sorting_with_samuel/main_pipeline/neuropixel-pipeline/viztools/mainwindow.py


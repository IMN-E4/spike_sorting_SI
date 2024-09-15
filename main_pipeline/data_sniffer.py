#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the script to find data in the NAS. It assumes that data is organized homogeneously.

"""

__author__ = "Eduarda Centeno"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2023/10/1"  ### Date it was created
__status__ = (
    "Production"  ### Production = still being developed. Else: Concluded/Finished.
)


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
from pathlib import Path
import glob

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
import pandas as pd

# Internal imports ### (Put here imports that are related to internal codes from the lab)


################################################################################

def find_data(rec_system, output_path, root_to_data="/nas"):
    if rec_system == "Neuropixel":
        target_path = Path(root_to_data) / "Neuropixel_Recordings"
        keyword = "*ap.meta"
        df = pd.DataFrame(
            columns=[
                "root",
                "rec_system",
                "brain_area",
                "implant_name",
                "intermediate",
                "rec_name"
            ]
        )

    elif rec_system == "Cambridge":
        target_path = Path(root_to_data) / "Cambridge_Recordings"
        keyword = "experiment*"
        df = pd.DataFrame(
            columns=[
                "root",
                "rec_system",
                "brain_area",
                "implant_name",
                "intermediate",
                "rec_name",
                "node",
                "experiment"
            ]
        )

    glob_path = target_path / f"**/{keyword}"
    all_files = glob.glob(glob_path.as_posix(), recursive=True)
    
    ## Populate dataframe
    for file in all_files:
        if rec_system == 'Neuropixel':
            split_parts = Path(file).parts[1:-1]

        elif rec_system == 'Cambridge':
            split_parts = Path(file).parts[1:]

        df.loc[len(df)] = split_parts

    df.to_csv(output_path/'from_sniffer.csv')

if __name__ == "__main__":
    find_data('Cambridge', output_path=Path('/home/eduarda/python-related/github-repos/spike_sorting_with_samuel/main_pipeline/cambridge-pipeline/viztools/'), root_to_data="/nas")
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import mne
from src.viz.visualizer import EEGVisualizer
from src.viz.visualizer import make_collage

csv_output_path = os.path.join(PROJECT_ROOT, "outputs", "eeg_culmination_csv", r"Alive_curious_still_wandering_the_edge_of_human_understand_and_nature's_mysteries_How's_the_world_at_your_end_eeg.csv")
num_rows = len(pd.read_csv(csv_output_path)) - 1
print(num_rows/256)
video_paths = EEGVisualizer(csv_output_path).visualize()
make_collage(video_paths, csv_output_path)

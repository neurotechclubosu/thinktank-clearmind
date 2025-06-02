import numpy as np
import pandas as pd
import mne
from visualizer import EEGVisualizer
from visualizer import make_collage

csv_output_path = r"eeg_culmination_csv\Alive_curious_still_wandering_the_edge_of_human_understand_and_nature's_mysteries_How's_the_world_at_your_end_eeg.csv"
num_rows = len(pd.read_csv(csv_output_path)) - 1
print(num_rows/256)
video_paths = EEGVisualizer(csv_output_path).visualize()
make_collage(video_paths, csv_output_path)

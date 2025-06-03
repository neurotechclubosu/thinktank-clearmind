import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import mne
from mne.datasets import sample
import os
import threading
import sys
import subprocess
import pyvista
from moviepy.editor import VideoFileClip, clips_array, vfx


class EEGVisualizer:
    def __init__(self, csv_path: str, subjects_dir: str = "C:/Users/anik2/mne_data/MNE-fsaverage-data/"):
        self.csv_path = csv_path
        self.subjects_dir = subjects_dir
        self.channels = [
            "Fp1", "Fp2", "F3", "F4", "T5", "T6",
            "O1", "O2", "F7", "F8", "C3", "C4",
            "T3", "T4", "P3", "P4"
        ]
        self.sfreq = 256
        self.n_jobs = -1

        # Precompute the source estimate once
        self.stc = self._compute_stc()

    def _compute_stc(self):
        df = pd.read_csv(self.csv_path)
        eeg_data = df.iloc[:, 2:].values.T
        info = mne.create_info(self.channels, sfreq=self.sfreq, ch_types=["eeg"] * len(self.channels))
        raw = mne.io.RawArray(eeg_data, info)
        raw.set_eeg_reference("average", projection=True)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)

        subject = "fsaverage"
        src = mne.setup_source_space(subject, spacing="ico4", subjects_dir=self.subjects_dir, add_dist=False)
        bem = mne.make_bem_model(subject, ico=3, subjects_dir=self.subjects_dir)
        bem_sol = mne.make_bem_solution(bem)
        trans = "fsaverage"
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol, eeg=True, n_jobs=self.n_jobs)

        if raw.n_times < 12000:
            cov = mne.compute_raw_covariance(raw, method="shrunk", n_jobs=self.n_jobs)
        else:
            cov = mne.compute_raw_covariance(raw, method="empirical", n_jobs=self.n_jobs)

        inv_op = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)
        stc = mne.minimum_norm.apply_inverse_raw(raw, inv_op, lambda2=1.0 / 3.0**2, method="dSPM")

        max_time = (len(df) - 1) / self.sfreq
        stc.crop(0.5, 10.0)
        return stc

    def _create_and_save_one_view(self, view_name: str, set_camera_fn, base_name: str, out_dir: str):
        """
        - set_camera_fn: a function that will be called as set_camera_fn(brain)
        """
        # 1) Create a fresh Brain instance
        brain = mne.viz.Brain(
            "fsaverage",
            subjects_dir=self.subjects_dir,
            show=True,
            background="white",
            cortex="classic",
            alpha=0.4,
            size=(800, 608)
        )

        # 2) Add the precomputed STC data (time argument ensures brain._times is set)
        add_kwargs = dict(
            fmin=self.stc.data.min(),
            fmax=self.stc.data.max(),
            alpha=0.25,
            smoothing_steps="nearest",
            time=self.stc.times
        )
        brain.add_data(self.stc.lh_data, hemi="lh", vertices=self.stc.lh_vertno, **add_kwargs)
        brain.add_data(self.stc.rh_data, hemi="rh", vertices=self.stc.rh_vertno, **add_kwargs)

        # 3) **Call set_camera_fn with this new brain instance** to position the camera
        set_camera_fn(brain)

        # 4) Save the movie
        fname = f"brain_video_{view_name}.mp4"
        out_path = os.path.join(out_dir, fname)
        brain.save_movie(out_path, framerate=0.8)
        brain.close()
        return out_path

    def visualize(self):
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        out_dir = os.path.join("brain_eeg_videos", base_name)
        os.makedirs(out_dir, exist_ok=True)

        # 2) Define each view as a function that takes a Brain instance 'b'
        views = {
            "xz_back":   lambda b: b.plotter.view_xz(),
            "xy_top":    lambda b: b.plotter.view_xy(),
            "yx_bottom": lambda b: b.plotter.view_yx(),
            "yz_right":  lambda b: b.plotter.view_yz(),
            # For "zx_front", do view_xz then roll 90° on that same 'b'
            "zx_front":  lambda b: (b.plotter.view_xz(), b.plotter.camera.Roll(90)),
            "zy_left":   lambda b: b.plotter.view_zy(),
        }

        saved_paths = []
        for view_name, set_camera_fn in views.items():
            path = self._create_and_save_one_view(view_name, set_camera_fn, base_name, out_dir)
            saved_paths.append(path)

        return saved_paths
    
def make_collage(video_paths: list[str], csv_path: str):
    print(video_paths)
    print("RUNNING MAKE COLLAGE")
    # 1) Load each clip
    clips = [VideoFileClip(p) for p in video_paths]

    # 2) (Optional) Make sure they all have exactly the same duration.
    #    If they differ by a fraction, you can cut or loop them to match.
    #    For example, force all to the shortest duration:
    min_duration = min(c.duration for c in clips)
    clips = [c.subclip(0, min_duration) for c in clips]

    # 3) Resize each clip to a common size (e.g. 480×360). 
    #    You could also pick a height and preserve aspect ratio:
    target_width, target_height = 480, 360
    clips_resized = [
        c.fx(vfx.resize, newsize=(target_width, target_height))
        for c in clips
    ]

    # 4) Arrange them into a 3×2 grid:
    #    clips_array takes a list of rows, each row is a list of clips.
    #    For example, a 2‐row by 3‐column layout:
    grid = clips_array([
        [clips_resized[0], clips_resized[1], clips_resized[2]],
        [clips_resized[3], clips_resized[4], clips_resized[5]],
    ])

    # 6) Write out the final collage (using a reasonable codec/bitrate):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join("brain_eeg_videos", base_name)
    output_path = os.path.join(out_dir, "collage_brain.mp4")
    grid.write_videofile(
        output_path,
        fps=24,              # match your source fps (or pick 24)
        codec="libx264",
        audio_codec="aac",
        bitrate="3000k"      # adjust as needed
    )
    # Open the generated video using the default media player
    os.startfile(output_path) if os.name == 'nt' else subprocess.run(['open', output_path])



def launch_in_subprocess(csv_path: str):
    subprocess.Popen(
        [sys.executable, __file__, csv_path]
        # creationflags=subprocess.CREATE_NO_WINDOW  # optional: hide terminal popup
    )



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <csv_path>")
    else:
        path = sys.argv[1]
        video_paths = EEGVisualizer(path).visualize()
        make_collage(video_paths, path)

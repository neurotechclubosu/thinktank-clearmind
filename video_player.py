import cv2
from PIL import Image, ImageTk

# ─────────────────────────────────────────────────────────────────────────────
# CustomVideoPlayer: reads via OpenCV, schedules frames with Tkinter .after()
# ─────────────────────────────────────────────────────────────────────────────

class CustomVideoPlayer:
    def __init__(self, video_path, label_widget, frame_delay_ms=33, loop=True, width=640, height=360):
        """
        - video_path: full path to your .mp4 file.
        - label_widget: the tk.Label (not CTkLabel) that will display the frames.
        - frame_delay_ms: milliseconds between frames (e.g. 33 ms ≈ 30 FPS; 100 ms ≈ 10 FPS).
        - loop: if True, it restarts from frame 0 when it reaches the end.
        - width, height: the final display size for each frame (will be resized).
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.label = label_widget
        self.frame_delay = frame_delay_ms
        self.loop = loop
        self.width = width
        self.height = height
        self._job_id = None  # store the .after() ID so we can cancel if needed

        # Pre-allocate a placeholder PhotoImage so Tkinter won’t flicker on the first frame
        self.label.imgtk = None

    def _show_frame(self):
        """
        Internal: grab one frame, convert, display, then schedule the next call.
        """
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            # End of video. If looping, restart; otherwise, stop.
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return
            else:
                return

        # Resize to (width×height) and convert BGR → RGB
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image → PhotoImage
        pil_img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)

        # Assign to label (keep a reference to avoid garbage-collection)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        # Schedule next frame
        self._job_id = self.label.after(self.frame_delay, self._show_frame)

    def play(self):
        """Start playback."""
        # If there’s already a scheduled callback, don’t start again.
        if self._job_id is None:
            self._show_frame()

    def stop(self):
        """Stop playback and release resources."""
        if self._job_id is not None:
            self.label.after_cancel(self._job_id)
            self._job_id = None

        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
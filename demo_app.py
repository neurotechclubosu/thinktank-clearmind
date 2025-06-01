import customtkinter as ctk
import tkinter as tk
import cv2
from PIL import Image, ImageTk

# ─────────────────────────────────────────────────────────────────────────────
class CustomVideoPlayer:
    def __init__(self, video_path, label_widget, frame_delay_ms=33, loop=True, width=640, height=360):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.label = label_widget
        self.frame_delay = frame_delay_ms
        self.loop = loop
        self.width = width
        self.height = height
        self._job_id = None

        self.label.imgtk = None

    def _show_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return
            else:
                return

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)

        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        self._job_id = self.label.after(self.frame_delay, self._show_frame)

    def play(self):
        if self._job_id is None:
            self._show_frame()

    def stop(self):
        if self._job_id is not None:
            self.label.after_cancel(self._job_id)
            self._job_id = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None


# ─────────────────────────────────────────────────────────────────────────────
video_label = None
_CURRENT_VIDEO_PLAYER = None

def embed_video(video_path, frame_delay_ms=100):
    global video_label, _CURRENT_VIDEO_PLAYER
    if video_label is None:
        return

    try:
        _CURRENT_VIDEO_PLAYER.stop()
    except Exception:
        pass

    player = CustomVideoPlayer(
        video_path=video_path,
        label_widget=video_label,
        frame_delay_ms=frame_delay_ms,
        loop=False,
        width=640,
        height=360
    )
    _CURRENT_VIDEO_PLAYER = player
    player.play()


def demo_app():
    root = ctk.CTk()
    root.title("Demo Video Slow-Player")
    root.geometry("700x600")

    frame = ctk.CTkFrame(root)
    frame.pack(pady=20, padx=20, fill="both")

    # Embedding a fixed 640×360 video area:
    video_frame = ctk.CTkFrame(frame, width=640, height=360, fg_color="#000")
    video_frame.grid_columnconfigure(0, weight=1)
    video_frame.grid(row=0, column=0, pady=(0, 20))
    video_frame.grid_propagate(False)

    global video_label
    video_label = tk.Label(
        video_frame,
        text="Video will appear here",
        bg="black",
        fg="white",
        font=("Segoe UI", 14)
    )
    video_label.place(relx=0, rely=0, relwidth=1, relheight=1)

    # After 1 second, attempt to play "test.mp4" at 100 ms/frame
    def start_later():
        embed_video("test.mp4", frame_delay_ms=1000)

    root.after(1000, start_later)
    root.mainloop()


if __name__ == "__main__":
    demo_app()

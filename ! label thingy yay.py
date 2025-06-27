import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import csv

class VideoLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame Labeler (with Hotkeys)")
        self.root.geometry("1200x800")

        # --- State Variables ---
        self.video_path = ""
        self.cap = None
        self.total_frames = 0
        self.current_frame_num = 0
        self.fps = 30 # Default fps
        self.is_playing = False
        self.labels = {}  # Dictionary to store {frame_number: "label"}

        # --- GUI Layout ---
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top Frame for file selection and info
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X)

        self.btn_open = ttk.Button(top_frame, text="Open Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, padx=5, pady=5)

        self.info_label = ttk.Label(top_frame, text="No video loaded.")
        self.info_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Middle Frame for Video and Labels
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Video Canvas
        self.canvas = tk.Canvas(middle_frame, bg="black")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Labels Display Area
        labels_frame = ttk.Frame(middle_frame)
        labels_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Label(labels_frame, text="Saved Labels").pack(anchor=tk.W)
        self.labels_text = tk.Text(labels_frame, width=40, height=25, state=tk.DISABLED)
        self.labels_text.pack(fill=tk.Y, expand=True)

        # Slider (Seekbar)
        self.scale = ttk.Scale(main_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.scale.pack(fill=tk.X, pady=5)

        # Controls Frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)

        self.btn_prev = ttk.Button(controls_frame, text="<< Prev Frame", command=self.prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_play_pause = ttk.Button(controls_frame, text="Play", command=self.play_pause)
        self.btn_play_pause.pack(side=tk.LEFT, padx=5)

        self.btn_next = ttk.Button(controls_frame, text="Next Frame >>", command=self.next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        # Labeling Frame
        labeling_frame = ttk.Frame(main_frame)
        labeling_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(labeling_frame, text="Label for current frame:").pack(side=tk.LEFT)
        self.label_entry = ttk.Entry(labeling_frame, width=50)
        self.label_entry.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.btn_save_label = ttk.Button(labeling_frame, text="Save Label", command=self.save_label)
        self.btn_save_label.pack(side=tk.LEFT, padx=5)

        # Export Frame
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(fill=tk.X, pady=5)
        self.btn_export = ttk.Button(export_frame, text="Export Labels to CSV", command=self.export_csv)
        self.btn_export.pack(side=tk.RIGHT)

        # --- Setup Hotkeys ---
        self.setup_hotkeys()

    def setup_hotkeys(self):
        """Binds keyboard shortcuts to functions."""
        self.root.bind('<space>', self.on_space_key)
        self.root.bind('<Right>', self.on_arrow_key)
        self.root.bind('<Left>', self.on_arrow_key)
        
        # Bind Enter/Return key specifically to the entry widget
        self.label_entry.bind('<Return>', self.on_enter_key)

    def on_space_key(self, event):
        """Handle spacebar press for play/pause."""
        # Only trigger play/pause if the user is NOT typing in the entry box
        if self.root.focus_get() != self.label_entry:
            self.play_pause()

    def on_arrow_key(self, event):
        """Handle left/right arrow key press for frame navigation."""
        # Only trigger if the user is NOT typing in the entry box
        if self.root.focus_get() != self.label_entry:
            if event.keysym == 'Right':
                self.next_frame()
            elif event.keysym == 'Left':
                self.prev_frame()

    def on_enter_key(self, event):
        """Handle Enter key press to save a label."""
        self.save_label()
        # Move focus away from the entry box to the main canvas
        # This allows immediate use of space/arrow keys again.
        self.canvas.focus_set()
        return "break" # Prevents the default Enter key behavior (like adding a newline)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if not self.video_path: return

        if self.cap: self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame_num = 0
        self.labels = {}
        self.is_playing = False
        self.btn_play_pause.config(text="Play")
        self.update_labels_display()

        self.info_label.config(text=f"{self.video_path.split('/')[-1]} | Total Frames: {self.total_frames}")
        self.scale.config(to=self.total_frames - 1)
        
        self.load_frame(self.current_frame_num)

    def load_frame(self, frame_num):
        if not self.cap or not self.cap.isOpened() or not (0 <= frame_num < self.total_frames):
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame_num = frame_num
            self.scale.set(self.current_frame_num)
            
            current_label = self.labels.get(self.current_frame_num, "")
            self.label_entry.delete(0, tk.END)
            self.label_entry.insert(0, current_label)

            self.display_frame(frame)
        
        self.info_label.config(text=f"{self.video_path.split('/')[-1]} | Frame: {self.current_frame_num + 1}/{self.total_frames}")

    def display_frame(self, frame):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        h, w, _ = frame.shape
        if h > 0 and w > 0 and canvas_width > 1 and canvas_height > 1:
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h))
        else: return

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        self.photo = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        x_pos = (canvas_width - new_w) // 2
        y_pos = (canvas_height - new_h) // 2
        self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.photo)

    def on_slider_move(self, value):
        frame_num = int(float(value))
        if frame_num != self.current_frame_num:
            if self.is_playing:
                self.play_pause() # Stop playback when slider is moved
            self.load_frame(frame_num)

    def play_pause(self):
        if not self.cap: return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play_pause.config(text="Pause")
            self.update_video()
        else:
            self.btn_play_pause.config(text="Play")

    def update_video(self):
        if self.is_playing and self.current_frame_num < self.total_frames - 1:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_num += 1
                self.scale.set(self.current_frame_num)
                self.display_frame(frame)
                self.info_label.config(text=f"{self.video_path.split('/')[-1]} | Frame: {self.current_frame_num + 1}/{self.total_frames}")
                self.root.after(int(1000 / self.fps), self.update_video)
            else:
                self.is_playing = False
                self.btn_play_pause.config(text="Play")
        else:
            self.is_playing = False
            self.btn_play_pause.config(text="Play")

    def next_frame(self):
        if self.current_frame_num < self.total_frames - 1:
            self.load_frame(self.current_frame_num + 1)

    def prev_frame(self):
        if self.current_frame_num > 0:
            self.load_frame(self.current_frame_num - 1)

    def save_label(self):
        label_text = self.label_entry.get().strip()
        if label_text:
            self.labels[self.current_frame_num] = label_text
        elif self.current_frame_num in self.labels:
            del self.labels[self.current_frame_num]
        
        self.update_labels_display()
        self.next_frame()

    def update_labels_display(self):
        self.labels_text.config(state=tk.NORMAL)
        self.labels_text.delete("1.0", tk.END)
        if self.labels:
            sorted_labels = sorted(self.labels.items())
            for frame_num, label in sorted_labels:
                self.labels_text.insert(tk.END, f"Frame {frame_num}: {label}\n")
        self.labels_text.config(state=tk.DISABLED)

    def export_csv(self):
        if not self.labels:
            messagebox.showinfo("Info", "No labels to export.")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Labels As"
        )
        if not save_path: return

        try:
            with open(save_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['frame_number', 'label'])
                for frame_num, label in sorted(self.labels.items()):
                    writer.writerow([frame_num, label])
            messagebox.showinfo("Success", f"Labels successfully exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export labels: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoLabelerApp(root)
    root.mainloop()
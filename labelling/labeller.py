import tkinter as tk
from tkinter import filedialog, messagebox, Scale
import cv2
from PIL import Image, ImageTk
import json
import time
import math

# --- Configuration ---
MAX_DISPLAY_WIDTH = 800
MAX_DISPLAY_HEIGHT = 600
# --- Configuration ---

class VideoLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Labeler")

        # --- State Variables ---
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame_num = 0 # The integer frame number currently displayed

        self.labels = {}  # Dictionary to store {frame_number: label}
        self.photo = None # To keep a reference to the PhotoImage
        self.is_labeling = False
        # self.label_entry and self.label_frame are created on demand in start_labeling
        self.label_entry = None
        self.label_frame = None

        # --- UI Elements ---

        # Menu Bar
        self.menu_bar = tk.Menu(root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open Video", command=self.open_video)
        self.file_menu.add_command(label="Save Labels", command=self.save_labels)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        root.config(menu=self.menu_bar)

        # Video Display Label
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)
        # Set a default placeholder size
        self.placeholder_img = ImageTk.PhotoImage(Image.new('RGB', (MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT//2), 'gray'))
        self.img_label.config(image=self.placeholder_img)

        # Frame Number Display
        self.frame_info_label = tk.Label(root, text="Frame: - / -")
        self.frame_info_label.pack()

        # Slider (Scale) - Standard behavior via command
        self.scale_var = tk.DoubleVar()
        self.slider = Scale(root, from_=0, to=0, orient=tk.HORIZONTAL,
                            variable=self.scale_var, command=self.on_scale_move,
                            showvalue=0, length=MAX_DISPLAY_WIDTH, relief='flat')
        self.slider.pack(fill=tk.X, padx=20, pady=5)

        # Label Entry (created on demand)
        self.label_entry_var = tk.StringVar()
        # self.label_prompt_label is created along with label_entry in start_labeling

        # --- Bindings ---
        # Only bind labeling keys to the root window
        self.root.bind("<KeyPress-e>", lambda event: self.start_labeling()) # Bind specifically to 'e'
        self.root.bind("<KeyPress-E>", lambda event: self.start_labeling()) # Bind specifically to 'E'
        # Bind Return and Escape globally *when not labeling*
        # These bindings might need careful management if other widgets consume them
        # Let's keep them bound only during the labeling state, handled by the entry widget bindings


    def open_video(self):
        """Opens a video file and initializes the UI."""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 files", "*.mp4"),
                       ("AVI files", "*.avi"),
                       ("All files", "*.*"))
        )
        if not filepath:
            return

        self.video_path = filepath
        if self.cap:
            self.cap.release() # Release previous video if any

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            self.cap = None
            self.total_frames = 0
            self.current_frame_num = 0
            self.slider.config(to=0) # Reset slider range
            self.scale_var.set(0)
            self.update_frame_info_label() # Update display
            self.img_label.config(image=self.placeholder_img) # Show placeholder
            self.root.title("Video Labeler")
            self.cancel_labeling() # Ensure labeling state is reset
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Handle cases where total_frames might be 0 or 1
        if self.total_frames <= 1:
            messagebox.showwarning("Warning", "Video has less than 2 frames. Labeling may be limited.")
            if self.total_frames == 0:
                 self.total_frames = 1 # Treat as 1 frame for slider range

        self.current_frame_num = 0
        self.labels = {} # Reset labels for new video
        self.cancel_labeling() # Ensure labeling state is reset

        # Configure slider (range is 0 to total_frames-1)
        self.slider.config(to=self.total_frames - 1)
        self.scale_var.set(0)

        # Show the first frame
        self.show_frame(0)
        # Set window title using the video filename without extension
        try:
             video_filename = self.video_path.split('/')[-1]
             base_name = video_filename.rsplit('.', 1)[0]
             self.root.title(f"Video Labeler - {base_name}")
        except Exception:
             self.root.title("Video Labeler") # Fallback title


    def get_display_size(self, frame_width, frame_height):
        """Calculates the display size maintaining aspect ratio."""
        # (This remains the same as it's about fitting the video into the window)
        aspect_ratio = frame_width / frame_height
        display_width = frame_width
        display_height = frame_height

        if display_width > MAX_DISPLAY_WIDTH:
            display_width = MAX_DISPLAY_WIDTH
            display_height = int(display_width / aspect_ratio)

        if display_height > MAX_DISPLAY_HEIGHT:
            display_height = MAX_DISPLAY_HEIGHT
            display_width = int(display_height * aspect_ratio)

        # Ensure minimum size if image is tiny or aspect ratio is extreme
        min_dim = 100
        if display_width < min_dim or display_height < min_dim:
             if aspect_ratio and aspect_ratio > 0:
                # Scale up based on the smaller dimension to reach min_dim
                if display_width < display_height:
                    display_width = min_dim
                    display_height = int(min_dim / aspect_ratio)
                else:
                    display_height = min_dim
                    display_width = int(min_dim * aspect_ratio)
             else: # Handle zero/invalid aspect ratio case
                display_width, display_height = min_dim, min_dim

        # Clamp again to prevent exceeding MAX if original was very large
        display_width = min(display_width, MAX_DISPLAY_WIDTH)
        display_height = min(display_height, MAX_DISPLAY_HEIGHT)
        # Ensure dimensions are at least 1 pixel
        display_width = max(1, display_width)
        display_height = max(1, display_height)

        return display_width, display_height

    def show_frame(self, frame_num):
        """Displays the specified frame number."""
        if self.cap is None or not self.cap.isOpened() or self.total_frames == 0:
            return

        # Ensure frame_num is a valid integer index
        frame_to_show = max(0, min(int(frame_num), self.total_frames - 1))

        # Avoid seeking if we're already on this integer frame
        if frame_to_show == self.current_frame_num and self.photo:
             # Frame is already displayed, just update the info label if needed
             self.update_frame_info_label()
             return

        # Seek to the frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_show)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame_num = frame_to_show

            # Resize frame for display
            h, w, _ = frame.shape
            display_w, display_h = self.get_display_size(w, h)
            frame_resized = cv2.resize(frame, (display_w, display_h))

            # Convert to Tkinter format
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.photo = ImageTk.PhotoImage(image=img_pil)

            # Update image label
            self.img_label.config(image=self.photo)
            self.img_label.image = self.photo # Keep reference!

            # Update frame number display
            self.update_frame_info_label()

            # Update the slider variable to match the displayed frame number
            # This is important if show_frame is called from places other than the slider's command
            # Use a small tolerance when checking float equality before setting
            current_scale_val = self.scale_var.get()
            if abs(current_scale_val - frame_to_show) > 0.001: # Use tolerance for float comparison
                self.scale_var.set(float(frame_to_show)) # Set as float


        else:
            print(f"Warning: Could not read frame {frame_to_show}")
            # Optionally show a placeholder or keep the last frame

    def update_frame_info_label(self):
        """Updates the text label showing frame number and label status."""
        label_text = self.labels.get(self.current_frame_num, "")
        label_display = f" | Label: '{label_text}'" if label_text else ""
        total_display = self.total_frames - 1 if self.total_frames > 0 else '-'
        self.frame_info_label.config(
            text=f"Frame: {self.current_frame_num} / {total_display}{label_display}"
        )

    def on_scale_move(self, value_str):
        """Called by the Scale widget's command when its value changes."""
        if self.cap and self.total_frames > 0 and not self.is_labeling:
            try:
                # The scale widget gives a string, convert to float then round to nearest integer frame
                new_frame_float = float(value_str)
                new_frame_int = int(round(new_frame_float))

                # Show the frame. show_frame handles the check if it's already the current frame.
                self.show_frame(new_frame_int)

            except ValueError:
                 # Handle cases where value_str might be invalid float
                 pass

    def start_labeling(self):
        """Initiates the UI for entering a label."""
        if not self.cap or self.total_frames == 0 or self.is_labeling:
             return # Cannot label without video, or if already labeling

        self.is_labeling = True
        self.slider.config(state=tk.DISABLED) # Disable slider during labeling

        # Create entry frame, prompt, and widget if they don't exist
        if not self.label_frame:
            self.label_frame = tk.Frame(self.root)
            # self.label_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5) # Pack later when needed

            self.label_prompt_label = tk.Label(self.label_frame, text="Label:")
            self.label_prompt_label.pack(side=tk.LEFT, padx=(0, 5))

            self.label_entry = tk.Entry(self.label_frame, textvariable=self.label_entry_var)
            self.label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Bind Enter and Escape directly to the entry widget
            self.label_entry.bind("<Return>", lambda event: self.commit_label())
            self.label_entry.bind("<Escape>", lambda event: self.cancel_labeling())

        # Ensure the frame is packed and visible
        self.label_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)

        # Clear previous entry and show current label if exists
        existing_label = self.labels.get(self.current_frame_num, "")
        self.label_entry_var.set(existing_label)

        self.label_entry.focus_set() # Set focus to the entry box
        self.label_entry.selection_range(0, tk.END) # Select existing text


    def commit_label(self):
        """Saves the entered label for the current frame."""
        if not self.is_labeling:
            return

        label_text = self.label_entry_var.get().strip()
        if label_text: # Only save non-empty labels
            self.labels[self.current_frame_num] = label_text
            print(f"Frame {self.current_frame_num}: Label saved = '{label_text}'")
        elif self.current_frame_num in self.labels: # Remove label if entry is cleared
            del self.labels[self.current_frame_num]
            print(f"Frame {self.current_frame_num}: Label removed.")

        self.cancel_labeling()
        self.update_frame_info_label() # Update display to show/remove label info

    def cancel_labeling(self):
        """Hides the label entry and resets labeling state."""
        # Check if the label_frame attribute exists before trying to use it
        if hasattr(self, 'label_frame') and self.label_frame and self.label_frame.winfo_exists() and self.label_frame.winfo_ismapped():
             self.label_frame.pack_forget()

        self.is_labeling = False
        self.label_entry_var.set("") # Clear variable
        self.slider.config(state=tk.NORMAL) # Re-enable slider
        self.root.focus_set() # Return focus to main window so key bindings work again


    def save_labels(self):
        """Saves the current labels dictionary to a file."""
        if not self.labels:
            messagebox.showwarning("No Labels", "There are no labels to save.")
            return

        # Suggest filename based on video name
        base_name = "video_labels"
        if self.video_path:
             try:
                 # Get filename without extension
                 video_filename = self.video_path.split('/')[-1]
                 base_name = video_filename.rsplit('.', 1)[0] + "_labels"
             except Exception:
                 pass # Keep default if parsing fails

        filepath = filedialog.asksaveasfilename(
            title="Save Labels As",
            defaultextension=".json",
            initialfile=base_name,
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )

        if not filepath:
            return

        try:
            # Store keys as strings for JSON compatibility
            labels_to_save = {str(k): v for k, v in self.labels.items()}
            with open(filepath, 'w') as f:
                json.dump(labels_to_save, f, indent=4)
            messagebox.showinfo("Success", f"Labels saved successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error Saving Labels", f"An error occurred:\n{e}")


    def run(self):
        """Starts the Tkinter main loop."""
        self.root.mainloop()
        # Release video capture on exit
        if self.cap:
            self.cap.release()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    # Optional: Set a minimum size for the window
    root.minsize(MAX_DISPLAY_WIDTH + 40, 400) # Adjust height as needed
    app = VideoLabelerApp(root)
    app.run()
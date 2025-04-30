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
FRAME_SKIP_SHIFT = 5 # Number of frames to skip when Shift is held
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
        # Set trace_add to handle slider release event specifically for smoother updates if needed
        # self.scale_var.trace_add("write", self.on_scale_drag) # Use command for simplicity now
        self.slider = Scale(root, from_=0, to=0, orient=tk.HORIZONTAL,
                            variable=self.scale_var, command=self.on_scale_move,
                            showvalue=0, length=MAX_DISPLAY_WIDTH, relief='flat')
        self.slider.pack(fill=tk.X, padx=20, pady=5)

        # Label Entry (created on demand)
        self.label_entry_var = tk.StringVar()
        # self.label_prompt_label is created along with label_entry in start_labeling

        # --- Bindings ---
        self.root.bind("<KeyPress-e>", lambda event: self.start_labeling()) # Bind specifically to 'e'
        self.root.bind("<KeyPress-E>", lambda event: self.start_labeling()) # Bind specifically to 'E'

        # --- Add frame navigation bindings ---
        self.root.bind("<KeyPress-Left>", self.navigate_frame)
        self.root.bind("<KeyPress-Right>", self.navigate_frame)
        # The Shift modifier will be checked within the navigate_frame function


    def open_video(self):
        """Opens a video file and initializes the UI."""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 files", "*.mp4"),
                       ("AVI files", "*.avi"),
                       ("MOV files", "*.mov"), # Added MOV
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
        if self.total_frames < 1: # Corrected condition
             messagebox.showwarning("Warning", "Video has no frames or could not read frame count.")
             self.total_frames = 1 # Treat as 1 frame for slider range to avoid division by zero etc.
             self.cap.release()
             self.cap = None # Treat as invalid video
             # Keep UI elements reflecting this state
             self.slider.config(to=0)
             self.scale_var.set(0)
             self.update_frame_info_label()
             self.img_label.config(image=self.placeholder_img)
             self.root.title("Video Labeler")
             self.cancel_labeling()
             return
        elif self.total_frames == 1:
             messagebox.showwarning("Warning", "Video has only 1 frame.")


        self.current_frame_num = 0
        self.labels = {} # Reset labels for new video
        self.cancel_labeling() # Ensure labeling state is reset

        # Configure slider (range is 0 to total_frames-1)
        self.slider.config(to=max(0, self.total_frames - 1)) # Ensure 'to' is not negative
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
        if frame_width <= 0 or frame_height <= 0: # Handle invalid frame dimensions
             return MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT // 2 # Return placeholder size

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
        min_dim = 50 # Reduced minimum slightly
        if display_width < min_dim or display_height < min_dim:
             if aspect_ratio and aspect_ratio > 0:
                # Scale up based on the smaller dimension to reach min_dim
                if display_width < display_height:
                    scale_factor = min_dim / display_width
                    display_width = min_dim
                    display_height = int(display_height * scale_factor)
                else:
                    scale_factor = min_dim / display_height
                    display_height = min_dim
                    display_width = int(display_width * scale_factor)
             else: # Handle zero/invalid aspect ratio case
                display_width, display_height = min_dim, min_dim

        # Clamp again to prevent exceeding MAX if calculations went over
        display_width = min(display_width, MAX_DISPLAY_WIDTH)
        display_height = min(display_height, MAX_DISPLAY_HEIGHT)

        # Ensure dimensions are at least 1 pixel
        display_width = max(1, int(display_width))
        display_height = max(1, int(display_height))

        return display_width, display_height

    def show_frame(self, frame_num):
        """Displays the specified frame number and updates the slider."""
        if self.cap is None or not self.cap.isOpened() or self.total_frames <= 0: # Changed check to <= 0
            # Update slider even if video isn't loaded, ensures consistency
            clamped_frame = max(0, min(int(frame_num), self.total_frames - 1 if self.total_frames > 0 else 0))
            self.current_frame_num = clamped_frame
            if abs(self.scale_var.get() - clamped_frame) > 0.001:
                self.scale_var.set(float(clamped_frame))
            self.update_frame_info_label()
            return

        # Ensure frame_num is a valid integer index
        # Use math.floor or int() for consistency - int() truncates towards zero
        frame_to_show = max(0, min(int(frame_num), self.total_frames - 1))

        # Avoid seeking if we're already on this integer frame *and* the image exists
        # This prevents flicker if show_frame is called repeatedly for the same frame
        if frame_to_show == self.current_frame_num and self.photo:
             # Frame is likely already displayed, just ensure slider and label are sync'd
             if abs(self.scale_var.get() - frame_to_show) > 0.001:
                 self.scale_var.set(float(frame_to_show))
             self.update_frame_info_label()
             return

        # Seek to the frame
        # Using set() can be slow; only set if the frame number actually changed significantly
        # However, for precise control, setting it is necessary.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_show)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame_num = frame_to_show # Update current frame *after* successful read

            # Resize frame for display
            h, w = frame.shape[:2] # Get height and width safely
            display_w, display_h = self.get_display_size(w, h)
            frame_resized = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA) # Use INTER_AREA for shrinking

            # Convert to Tkinter format
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.photo = ImageTk.PhotoImage(image=img_pil)

            # Update image label
            self.img_label.config(image=self.photo)
            # self.img_label.image = self.photo # Keep reference! Not strictly needed if self.photo is maintained

            # Update frame number display
            self.update_frame_info_label()

            # Update the slider variable to match the *actual* displayed frame number
            current_scale_val = self.scale_var.get()
            if abs(current_scale_val - self.current_frame_num) > 0.001: # Use tolerance
                self.scale_var.set(float(self.current_frame_num)) # Set as float

        else:
            # Frame read failed, could be end of video or an error
            print(f"Warning: Could not read frame {frame_to_show}. Trying previous frame or resetting.")
            # Attempt to reset position slightly, maybe read got stuck
            if frame_to_show > 0:
                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_show -1) # Go back one frame
                 ret, frame = self.cap.read()
                 if ret:
                     self.current_frame_num = frame_to_show -1
                     print(f"Read frame {self.current_frame_num} instead.")
                     # Call show_frame again for the *actual* frame read
                     # Need to prevent infinite recursion if read always fails
                     # For now, just update based on what we have
                     self.show_frame(self.current_frame_num) # Re-call to display the fallback frame
                 else:
                      print("Fallback read also failed.")
                      # If read fails consistently, maybe show placeholder or keep last good frame?
                      # For now, just update UI state
                      self.current_frame_num = frame_to_show # Keep target number for UI consistency? Or fallback?
                      if abs(self.scale_var.get() - self.current_frame_num) > 0.001:
                         self.scale_var.set(float(self.current_frame_num))
                      self.update_frame_info_label() # Update label even on failure
            else:
                 # Already at frame 0, cannot go back
                 self.current_frame_num = 0 # Reset to 0
                 if abs(self.scale_var.get() - self.current_frame_num) > 0.001:
                     self.scale_var.set(float(self.current_frame_num))
                 self.update_frame_info_label()


    def update_frame_info_label(self):
        """Updates the text label showing frame number and label status."""
        label_text = self.labels.get(self.current_frame_num, "")
        label_display = f" | Label: '{label_text}'" if label_text else ""
        # Handle total_frames = 0 or 1 correctly for display
        total_display = max(0, self.total_frames - 1) if self.total_frames > 0 else '-'
        current_display = self.current_frame_num if self.cap else '-' # Show '-' if no video loaded

        self.frame_info_label.config(
            text=f"Frame: {current_display} / {total_display}{label_display}"
        )

    def on_scale_move(self, value_str):
        """Called by the Scale widget's command when its value changes (drag or click)."""
        if self.cap and self.total_frames > 0 and not self.is_labeling:
            try:
                # The scale widget gives a string, convert to float then round to nearest integer frame
                new_frame_float = float(value_str)
                new_frame_int = int(round(new_frame_float)) # Round to nearest frame

                # Only call show_frame if the target *integer* frame is different
                # from the current one to avoid unnecessary seeks during fine dragging
                if new_frame_int != self.current_frame_num:
                     self.show_frame(new_frame_int)
                # Else: Slider moved slightly but corresponds to the same frame, do nothing extra.
                # The slider variable itself is already updated via its 'variable' option.

            except ValueError:
                 pass # Ignore if the value is somehow not a float string

    def navigate_frame(self, event):
        """Handles Left/Right arrow key presses for frame navigation."""
        # Do nothing if no video is loaded or if currently labeling
        if not self.cap or self.total_frames <= 0 or self.is_labeling:
            return

        # Check if Shift key is pressed (modifier state)
        # State masks can vary slightly, checking for Shift mask (usually 0x0001)
        is_shift_pressed = (event.state & 0x0001) != 0

        step = FRAME_SKIP_SHIFT if is_shift_pressed else 1

        new_frame_num = self.current_frame_num

        if event.keysym == "Left":
            new_frame_num -= step
        elif event.keysym == "Right":
            new_frame_num += step
        else:
            return # Should not happen if binding is correct, but good practice

        # Clamp the frame number to valid range [0, total_frames - 1]
        new_frame_num = max(0, min(new_frame_num, self.total_frames - 1))

        # If the target frame is different from the current one, show it
        if new_frame_num != self.current_frame_num:
            self.show_frame(new_frame_num)
            # show_frame already updates the slider via self.scale_var.set()


    def start_labeling(self):
        """Initiates the UI for entering a label."""
        if not self.cap or self.total_frames <= 0 or self.is_labeling:
             return # Cannot label without video, or if already labeling

        self.is_labeling = True
        self.slider.config(state=tk.DISABLED) # Disable slider during labeling

        # Create entry frame, prompt, and widget if they don't exist
        if not self.label_frame:
            self.label_frame = tk.Frame(self.root)
            # Pack is deferred until needed

            self.label_prompt_label = tk.Label(self.label_frame, text="Label:")
            self.label_prompt_label.pack(side=tk.LEFT, padx=(0, 5))

            self.label_entry = tk.Entry(self.label_frame, textvariable=self.label_entry_var, width=50) # Give it a decent width
            self.label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Bind Enter and Escape directly to the entry widget
            self.label_entry.bind("<Return>", lambda event: self.commit_label())
            self.label_entry.bind("<Escape>", lambda event: self.cancel_labeling())
            # Prevent arrow keys used for navigation from propagating when entry has focus
            self.label_entry.bind("<Left>", lambda e: "break")
            self.label_entry.bind("<Right>", lambda e: "break")


        # Pack the frame below the slider (or wherever appropriate)
        # Insert it *before* the frame info label for better layout? Or just pack at bottom?
        # Packing below the slider seems reasonable.
        self.label_frame.pack(fill=tk.X, padx=20, pady=(0, 5), before=self.frame_info_label) # Place above frame info


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
        if hasattr(self, 'label_frame') and self.label_frame and self.label_frame.winfo_exists():
             # Check if it's packed before trying to forget
             if self.label_frame.winfo_ismapped():
                  self.label_frame.pack_forget()

        self.is_labeling = False
        self.label_entry_var.set("") # Clear variable
        if self.cap and self.total_frames > 0: # Only enable slider if video is loaded
             self.slider.config(state=tk.NORMAL)
        else:
             self.slider.config(state=tk.DISABLED)

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
                 video_filename = self.video_path.replace('\\', '/').split('/')[-1] # Handle windows paths
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
            # Sort labels by frame number (key) before saving for readability
            labels_to_save = {str(k): v for k, v in sorted(self.labels.items())}
            with open(filepath, 'w') as f:
                json.dump(labels_to_save, f, indent=4)
            messagebox.showinfo("Success", f"Labels saved successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error Saving Labels", f"An error occurred:\n{e}")


    def run(self):
        """Starts the Tkinter main loop."""
        # Set focus to the root window initially so key bindings work immediately
        self.root.focus_set()
        self.root.mainloop()
        # Release video capture on exit
        if self.cap:
            self.cap.release()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    # Optional: Set a minimum size for the window
    # Calculate min height based on placeholder image and controls
    min_height = (MAX_DISPLAY_HEIGHT // 2) + 150 # Placeholder height + slider + labels + padding
    root.minsize(MAX_DISPLAY_WIDTH + 40, min_height) # Adjust height as needed
    app = VideoLabelerApp(root)
    app.run()
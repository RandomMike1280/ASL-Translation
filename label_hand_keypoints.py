"""
label_hand_keypoints.py

Tkinter UI to label recorded hand keypoints data.
Loads a .npz data file (distances & frame filenames), displays frames and distance vector dimensions,
and provides 26 buttons (A-Z) to label each sample. Labeled data saved to ./dataset with matching indices.
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import os
from PIL import Image, ImageTk
import string


def main():
    # Select .npz data file
    root = tk.Tk()
    root.withdraw()
    npz_path = filedialog.askopenfilename(
        parent=root,
        title="Select data.npz file",
        filetypes=[("NPZ files", "*.npz")]
    )
    root.destroy()
    if not npz_path:
        return

    data = np.load(npz_path, allow_pickle=True)
    distances = data['distances']  # shape (n_samples, features)
    frames = data['frames']        # filenames array
    n_samples = len(distances)
    features = distances.shape[1]

    # Frames directory
    basedir = os.path.dirname(npz_path)
    frames_dir = os.path.join(basedir, 'frames')

    labels = [''] * n_samples
    idx = 0

    # Build labeling UI
    ui = tk.Tk()
    ui.title("Hand Keypoints Labeler")

    img_label = tk.Label(ui)
    img_label.pack()

    info_label = tk.Label(ui, text="")
    info_label.pack(pady=5)

    btn_frame = tk.Frame(ui)
    btn_frame.pack(pady=5)

    def load_sample(i):
        # Load and display image
        img_path = os.path.join(frames_dir, frames[i])
        pil_img = Image.open(img_path)
        # Optionally resize
        pil_img = pil_img.resize((400, 300))
        tk_img = ImageTk.PhotoImage(pil_img, master=ui)
        img_label.configure(image=tk_img)
        img_label.image = tk_img
        # Display info
        info_label.config(
            text=f"Index: {i+1}/{n_samples}    Features: {features}"
        )

    def on_label(letter):
        nonlocal idx
        labels[idx] = letter
        idx += 1
        if idx >= n_samples:
            # Save labeled data
            save_dir = 'dataset'
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(npz_path))[0]
            # generate unique filename with sequential suffix
            i = 1
            while True:
                fname = f"{base}_labels_{i:03d}.npz"
                out_path = os.path.join(save_dir, fname)
                if not os.path.exists(out_path):
                    break
                i += 1
            np.savez(out_path,
                     distances=distances,
                     labels=np.array(labels))
            messagebox.showinfo("Saved", f"Labeled data saved to {out_path}")
            ui.destroy()
        else:
            load_sample(idx)

    # Create A-Z buttons in two rows
    for i, letter in enumerate(string.ascii_uppercase):
        btn = tk.Button(
            btn_frame,
            text=letter,
            width=3,
            command=lambda l=letter: on_label(l)
        )
        btn.grid(row=i//13, column=i%13, padx=2, pady=2)

    # Add 'None' button for no valid sign or no hand
    none_btn = tk.Button(
        btn_frame,
        text='None',
        width=5,
        command=lambda: on_label('None')
    )
    none_btn.grid(row=2, column=0, padx=2, pady=5, columnspan=2)

    # Bind left arrow key to go back to previous sample
    def on_prev(event):
        nonlocal idx
        if idx > 0:
            idx -= 1
            load_sample(idx)

    ui.bind('<Left>', on_prev)

    # Start with first sample
    load_sample(idx)
    ui.mainloop()


if __name__ == '__main__':
    main()

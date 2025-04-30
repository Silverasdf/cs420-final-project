import numpy as np
import matplotlib.pyplot as plt
import os
import mido
from hopfield_midi_utils import (
    get_time_steps_from_midi,
    midi_to_pattern,
    imprint_patterns,
    add_noise_to_pattern,
    recall_pattern,
    pattern_to_midi,
)

# === Parameters ===
midi_file = "midi_files/STARSSF.mid"
output_dir = "recall"
noise_level = 0.5  # 50% noise
save_visualization = True
save_midis = True

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# === Load and Convert Pattern ===
time_steps = min(get_time_steps_from_midi(midi_file), 32)
time_steps = max(1, (1024 // 88))
num_neurons = 88 * time_steps

original_pattern = midi_to_pattern(midi_file, num_neurons, time_steps)
weights = imprint_patterns(np.array([original_pattern]), num_neurons)

noisy_pattern = add_noise_to_pattern(original_pattern, noise_level)
recalled_pattern = recall_pattern(weights, noisy_pattern)

# === Save MIDIs ===
if save_midis:
    pattern_to_midi(original_pattern, os.path.join(output_dir, "original_50.mid"), time_steps=time_steps, save=True)
    pattern_to_midi(noisy_pattern, os.path.join(output_dir, "noisy_50.mid"), time_steps=time_steps, save=True)
    pattern_to_midi(recalled_pattern, os.path.join(output_dir, "recalled_50.mid"), time_steps=time_steps, save=True)

# === Visualize Patterns ===
def visualize_patterns(original, noisy, recalled, time_steps, save_path=None):
    shape = (time_steps, 88)

    def to_matrix(p):
        return p.reshape(shape)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Original", "Noisy (50%)", "Recalled"]

    for i, pattern in enumerate([original, noisy, recalled]):
        axs[i].imshow(to_matrix(pattern), aspect='auto', cmap='gray_r', interpolation='nearest')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Note Index")
        axs[i].set_ylabel("Time Step")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

if save_visualization:
    vis_path = os.path.join(output_dir, "pattern_comparison_50.png")
    visualize_patterns(original_pattern, noisy_pattern, recalled_pattern, time_steps, vis_path)
else:
    visualize_patterns(original_pattern, noisy_pattern, recalled_pattern, time_steps)

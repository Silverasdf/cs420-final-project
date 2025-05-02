import os
import numpy as np
import matplotlib.pyplot as plt
from new_hopfield import (
    get_time_steps_from_midi,
    midi_to_pattern,
    add_noise_to_pattern,
    imprint_patterns,
    recall_pattern,
    pattern_to_midi
)

# ====== SETUP ======
midi_file = "demo/littlestar.mid"
noise_level = 0.5
output_dir = "demo_outputs"
os.makedirs(output_dir, exist_ok=True)

# ====== Convert MIDI to pattern ======
time_steps = min(get_time_steps_from_midi(midi_file), 32)
num_neurons = 88 * time_steps


original_pattern = midi_to_pattern(midi_file, num_neurons, time_steps)
noisy_pattern = add_noise_to_pattern(original_pattern, noise_level)

# ====== Train and Recall ======
weights = imprint_patterns(np.array([original_pattern]), num_neurons)
recalled_pattern = recall_pattern(weights, noisy_pattern)

# ====== Save MIDI files ======
pattern_to_midi(original_pattern, os.path.join(output_dir, "original.mid"), save=True)
pattern_to_midi(noisy_pattern, os.path.join(output_dir, "noisy.mid"), save=True)
pattern_to_midi(recalled_pattern, os.path.join(output_dir, "recalled.mid"), save=True)

# ====== Visualization ======
def plot_patterns(original, noisy, recalled, steps, output_path):
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    axs[0].imshow(original.reshape((steps, 88)), aspect='auto', cmap='gray')
    axs[0].set_title("Original Pattern")
    
    axs[1].imshow(noisy.reshape((steps, 88)), aspect='auto', cmap='gray')
    axs[1].set_title("Noisy Pattern (50% noise)")
    
    axs[2].imshow(recalled.reshape((steps, 88)), aspect='auto', cmap='gray')
    axs[2].set_title("Recalled Pattern")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")
    plt.close()

plot_patterns(original_pattern, noisy_pattern, recalled_pattern, time_steps,
              os.path.join(output_dir, "pattern_comparison.png"))

import numpy as np
import mido
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# ============================ MIDI PROCESSING FUNCTIONS ============================

# These are the new functions that I needed to write -- used internet tools for this

def get_time_steps_from_midi(midi_file, resolution=10):
    """ Determine number of time steps for a MIDI file. """
    mid = mido.MidiFile(midi_file)
    total_seconds = mid.length or 5  # Default 5s if length missing
    return max(1, int(total_seconds * resolution))

def midi_to_pattern(midi_file, num_neurons=100, time_steps=32):
    """ Convert a MIDI file into a binary pattern for a Hopfield network. """
    min_note, max_note = 21, 108
    original_neurons = 88 * time_steps
    pattern_matrix = np.full((time_steps, 88), -1)

    mid = mido.MidiFile(midi_file)
    time_bin = 0
    active_notes = set()

    for msg in mid.play():
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes.add(msg.note)
        elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:
            active_notes.discard(msg.note)

        if time_bin < time_steps:
            for note in active_notes:
                if min_note <= note <= max_note:
                    pattern_matrix[time_bin, note - min_note] = 1
            time_bin += 1
        else:
            break

    flattened_pattern = pattern_matrix.flatten()
    if num_neurons < original_neurons:
        pca = PCA(n_components=num_neurons)
        reduced_pattern = pca.fit_transform(flattened_pattern.reshape(1, -1))
        return np.sign(reduced_pattern.flatten())
    elif num_neurons > original_neurons:
        padding = np.full(num_neurons - original_neurons, -1)
        return np.concatenate((flattened_pattern, padding))
    else:
        return flattened_pattern

# ============================ HOPFIELD NETWORK FUNCTIONS ============================

# These are just from Lab 3
def imprint_patterns(patterns, num_neurons):
    """ Train a Hopfield network by imprinting patterns as weights. """
    weights = np.zeros((num_neurons, num_neurons))
    for pattern in patterns:
        weights += np.outer(pattern, pattern)
    weights /= num_neurons
    np.fill_diagonal(weights, 0)  # No self-connections
    return weights

def recall_pattern(weights, noisy_pattern, num_iterations=10):
    """ Recover a stored pattern from a noisy input using the Hopfield update rule. """
    state = np.copy(noisy_pattern)
    for _ in range(num_iterations):
        for i in range(len(state)):
            h_i = np.dot(weights[i], state)
            state[i] = 1 if h_i > 0 else -1  # Sign function
    return state

# ============================ NOISE INJECTION ============================

# Literally sample random bits and flip the signs
def add_noise_to_pattern(pattern, noise_level=0.1):
    """ Introduce controlled noise by flipping a fraction of bits. """
    noisy_pattern = np.copy(pattern)
    num_flip = int(len(pattern) * noise_level)
    flip_indices = np.random.choice(len(pattern), num_flip, replace=False)
    noisy_pattern[flip_indices] *= -1
    return noisy_pattern

# ============================ MIDI OUTPUT ============================

# Again, new midi stuff that I needed to write -- used internet tools for this
def pattern_to_midi(pattern, output_file, num_notes=88, save=False):
    """ Convert a recalled pattern back into a MIDI file. """
    min_note = 21  # Start of piano range
    pattern_size = len(pattern)

    # Infer time steps dynamically
    time_steps = pattern_size // num_notes
    if time_steps * num_notes != pattern_size:
        raise ValueError(f"Pattern size {pattern_size} is not a multiple of {num_notes}.")

    pattern = pattern.reshape((time_steps, num_notes))

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for t, time_step in enumerate(pattern):
        for note_index, value in enumerate(time_step):
            if value == 1:
                track.append(mido.Message('note_on', note=min_note + note_index, velocity=64, time=t * 100))
        track.append(mido.Message('note_off', note=min_note, velocity=64, time=100))

    if save:
        mid.save(output_file)
        print(f"Saved recalled melody: {output_file}")
    else:
        print(f"{output_file} not saved. Set save=True to save the output.")

# ============================ EXPERIMENT FUNCTION ============================

def run_noise_experiment(midi_file, noise_levels, num_trials=5):
    """
    Run experiments where different levels of noise are added to a stored pattern.

    Parameters:
    - midi_file: str, path to a MIDI file
    - noise_levels: list of floats, different noise percentages to test
    - num_trials: int, number of times to run each noise level

    Returns:
    - accuracy_results: list of recall accuracy per noise level
    """
    time_steps = get_time_steps_from_midi(midi_file)
    num_neurons = 88 * time_steps

    # Convert MIDI to a pattern
    original_pattern = midi_to_pattern(midi_file, num_neurons, time_steps)

    # Train the Hopfield network
    weights = imprint_patterns(np.array([original_pattern]), num_neurons)

    accuracy_results = []

    for noise in noise_levels:
        correct_retrievals = 0

        for _ in range(num_trials):
            noisy_pattern = add_noise_to_pattern(original_pattern, noise)
            recalled_pattern = recall_pattern(weights, noisy_pattern)

            # Compare recall success (matching bits)
            match_ratio = np.mean(recalled_pattern == original_pattern)
            correct_retrievals += match_ratio

            # Save one recalled MIDI per noise level
            midi_output = f"recalled_noise_{midi_file[:-4]}_{int(noise * 100)}.mid"
            pattern_to_midi(recalled_pattern, midi_output)

        avg_accuracy = correct_retrievals / num_trials
        accuracy_results.append(avg_accuracy)

        print(f"Noise Level {noise}: Recall Accuracy {avg_accuracy:.2f}")

    return accuracy_results

# ============================ PLOT EXPERIMENT RESULTS ============================

def plot_recall_vs_noise(noise_levels, accuracy_results):
    """ Plot recall accuracy vs. noise level. """
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, accuracy_results, marker='o', linestyle='-')
    plt.xlabel("Noise Level")
    plt.ylabel("Recall Accuracy")
    plt.title("Hopfield Network Recall Accuracy vs. Noise")
    plt.grid(True)
    plt.savefig("recall_accuracy_vs_noise.png")

# ============================ RUN EXPERIMENT ============================

midi_file = "STARSSF.mid"

import numpy as np

# Define noise levels (0% to 70% flipped bits)
noise_levels = list(np.arange(0, 0.71, 0.01))  # [0.0, 0.1, ..., 0.7]

# Run experiment
accuracy_results = run_noise_experiment(midi_file, noise_levels, num_trials=5)

# Plot results
plot_recall_vs_noise(noise_levels, accuracy_results)

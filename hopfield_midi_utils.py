import numpy as np
import mido

def get_time_steps_from_midi(midi_file, resolution=10):
    mid = mido.MidiFile(midi_file)
    total_seconds = mid.length or 5
    return max(1, int(total_seconds * resolution))

def midi_to_pattern(midi_file, num_neurons=100, time_steps=32):
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
        indices = np.round(np.linspace(0, original_neurons - 1, num_neurons)).astype(int)
        return flattened_pattern[indices]
    elif num_neurons > original_neurons:
        padding = np.full(num_neurons - original_neurons, -1)
        return np.concatenate((flattened_pattern, padding))
    else:
        return flattened_pattern

def imprint_patterns(patterns, num_neurons):
    weights = np.zeros((num_neurons, num_neurons), dtype=np.float32)
    for pattern in patterns:
        pattern = pattern.astype(np.float32)
        weights += np.outer(pattern, pattern)
    weights /= num_neurons
    np.fill_diagonal(weights, 0)
    return weights

def recall_pattern(weights, noisy_pattern, num_iterations=10):
    state = np.copy(noisy_pattern)
    for _ in range(num_iterations):
        for i in range(len(state)):
            h_i = np.dot(weights[i], state)
            state[i] = 1 if h_i > 0 else -1
    return state

def add_noise_to_pattern(pattern, noise_level=0.1):
    noisy_pattern = np.copy(pattern)
    num_flip = int(len(pattern) * noise_level)
    flip_indices = np.random.choice(len(pattern), num_flip, replace=False)
    noisy_pattern[flip_indices] *= -1
    return noisy_pattern

def pattern_to_midi(pattern, output_file, time_steps=32, num_notes=88, save=False):
    min_note = 21
    pattern_size = len(pattern)

    if time_steps * num_notes != pattern_size:
        raise ValueError(f"Pattern size {pattern_size} is not a multiple of {num_notes} with time_steps={time_steps}.")

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

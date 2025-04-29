# don't forget to pip install pretyy_midi
import pretty_midi
import numpy as np
import os

INPUT_MIDI_PATH = "demo/littlestar.mid"
OUTPUT_FOLDER = "demo/"
TIME_BIN_SIZE = 0.25  # quarter note resolution (in seconds)

def midi_to_binary_vector(midi_path, bin_size=TIME_BIN_SIZE):
    midi = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = midi.get_piano_roll(fs=1/bin_size)
    piano_roll = (piano_roll > 0).astype(int)
    
    # Crop to valid MIDI note range (A0=21 to C8=108 → 88 keys)
    piano_roll = piano_roll[21:109, :]
    
    binary_vector = piano_roll.T.flatten()
    return binary_vector, piano_roll.shape[1]  # also return number of time steps

def binary_to_midi(binary_vector, num_time_bins, bin_size=TIME_BIN_SIZE, output_path="output.mid"):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    binary_vector = binary_vector.reshape((num_time_bins, 88))
    
    for t in range(num_time_bins):
        notes = np.where(binary_vector[t] == 1)[0]
        for note in notes:
            pitch = note + 21
            start = t * bin_size
            end = start + bin_size
            instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))

    midi.instruments.append(instrument)
    midi.write(output_path)

def add_noise(binary_vector, noise_percent):
    noisy_vector = binary_vector.copy()
    num_flips = int(len(noisy_vector) * (noise_percent / 100))
    flip_indices = np.random.choice(len(noisy_vector), size=num_flips, replace=False)
    noisy_vector[flip_indices] ^= 1  # flip bits
    return noisy_vector

def main():
    if not os.path.exists(INPUT_MIDI_PATH):
        raise FileNotFoundError(f"MIDI file not found at {INPUT_MIDI_PATH}")

    print(f"Loading original MIDI from: {INPUT_MIDI_PATH}")
    original_binary, num_time_bins = midi_to_binary_vector(INPUT_MIDI_PATH)

    # Save clean version (to verify pipeline)
    binary_to_midi(original_binary, num_time_bins, output_path=os.path.join(OUTPUT_FOLDER, "clean.mid"))

    for percent in [10, 40, 60]:
        noisy_vector = add_noise(original_binary, percent)
        output_file = os.path.join(OUTPUT_FOLDER, f"noisy_{percent}percent.mid")
        binary_to_midi(noisy_vector, num_time_bins, output_path=output_file)
        print(f"Saved noisy MIDI with {percent}% noise to {output_file}")

if __name__ == "__main__":
    main()

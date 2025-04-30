import os
import numpy as np
import pretty_midi

# 1. Extract note events from MIDI
def midi_to_note_events(midi_path, quantization=0.25):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitch = note.pitch
            start = round(note.start / quantization) * quantization
            end = round(note.end / quantization) * quantization
            duration = max(end - start, quantization)
            notes.append((pitch, start, duration))
    notes.sort(key=lambda x: x[1])
    return notes

# 2. Convert note events to binary vector
def note_events_to_vector(notes, quantization=0.25, time_range=60.0):
    vector = []
    time_bins = int(time_range / quantization)
    for t in range(time_bins):
        active_notes = [pitch for pitch, start, dur in notes
                        if start <= t * quantization < start + dur]
        bin_vector = [0] * 128
        for pitch in active_notes:
            bin_vector[pitch] = 1
        vector.extend(bin_vector)
    return np.array(vector, dtype=np.uint8)

# 3. Add structured noise: controlled pitch shifting and note dropping
def add_structured_note_noise(vector, pitch_shift_prob=0.1, drop_note_prob=0.05):
    piano_roll = np.reshape(vector.copy(), (-1, 128))
    num_timesteps, num_notes = piano_roll.shape

    for t in range(num_timesteps):
        notes_on = np.where(piano_roll[t] == 1)[0]
        for note in notes_on:
            # Possibly drop note
            if np.random.rand() < drop_note_prob:
                piano_roll[t, note] = 0
                continue

            # Possibly shift note pitch
            if np.random.rand() < pitch_shift_prob:
                shift = np.random.choice([-2, -1, 1, 2])
                new_note = note + shift
                if 0 <= new_note < num_notes:
                    piano_roll[t, note] = 0
                    piano_roll[t, new_note] = 1

    return piano_roll.flatten()

# 4. Convert vector back to note events
def vector_to_note_events(vector, quantization=0.25):
    num_bins = len(vector) // 128
    piano_roll = np.reshape(vector, (num_bins, 128))
    notes = []
    for pitch in range(128):
        is_note_on = False
        start = 0
        for t in range(num_bins):
            if piano_roll[t][pitch] == 1 and not is_note_on:
                start = t * quantization
                is_note_on = True
            elif piano_roll[t][pitch] == 0 and is_note_on:
                end = t * quantization
                duration = end - start
                notes.append((pitch, start, duration))
                is_note_on = False
        if is_note_on:
            end = num_bins * quantization
            duration = end - start
            notes.append((pitch, start, duration))
    return notes

# 5. Write note events to MIDI
def note_events_to_midi(notes, output_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, duration in notes:
        note = pretty_midi.Note(velocity=100, pitch=pitch,
                                start=start, end=start + duration)
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    midi.write(output_path)

# 6. Main processing function
def process_with_structured_noise_levels(input_path, output_folder, noise_levels, quantization=0.25, time_range=60.0):
    os.makedirs(output_folder, exist_ok=True)

    print("Extracting clean note events...")
    clean_notes = midi_to_note_events(input_path, quantization)
    clean_vector = note_events_to_vector(clean_notes, quantization, time_range)

    # Save clean version
    reconstructed_notes = vector_to_note_events(clean_vector, quantization)
    clean_output_path = os.path.join(output_folder, "clean.mid")
    note_events_to_midi(reconstructed_notes, clean_output_path)
    print(f"Saved clean version to {clean_output_path}")

    # Save structured noisy versions
    for level in noise_levels:
        print(f"Generating structured noise at {int(level * 100)}%...")
        noisy_vector = add_structured_note_noise(
            clean_vector,
            pitch_shift_prob=level,
            drop_note_prob=level * 0.5  # drop rate is half of shift rate
        )
        noisy_notes = vector_to_note_events(noisy_vector, quantization)
        output_file = os.path.join(output_folder, f"noisy_{int(level * 100)}.mid")
        note_events_to_midi(noisy_notes, output_file)
        print(f"Saved structured noisy file to {output_file}")

# 7. Execute
if __name__ == "__main__":
    input_midi = "demo/littlestar.mid"
    output_dir = "demo2"
    noise_percentages = [0.05, 0.10, 0.30, 0.50, 0.70]
    process_with_structured_noise_levels(input_midi, output_dir, noise_percentages)

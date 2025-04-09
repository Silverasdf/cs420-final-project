import os
import threading

files = os.listdir("midi_files/")

# Split list of files into 5 files -- make a thread for each set of files

files = [files[i:i + 5] for i in range(0, len(files), 5)]
print(files)

def run_noise_experiment(files, num_trials, output_dir):
    """ Run the noise experiment on a set of MIDI files. """
    for midi_file in files:
        # Call the function to run the experiment on each file
        #print(f"python new_hopfield.py --midi_file midi_files/{midi_file} --num_trials 10 --output_dir outputs")
        try:
            print(f"Processing {midi_file}...")
            os.system(f"python new_hopfield.py --midi_file midi_files/{midi_file} --num_trials 10 --output_dir outputs")
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")

# For each set of files, run the experiment in a separate thread
threads = []
for file_set in files:
    thread = threading.Thread(target=run_noise_experiment, args=(file_set, 10, "outputs"))
    threads.append(thread)
    thread.start()
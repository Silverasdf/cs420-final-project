import os

for midi_file in os.listdir("midi_files"):
    os.system(f"python new_hopfield.py --midi_file midi_files/{midi_file} --num_trials 10 --output_dir outputs")
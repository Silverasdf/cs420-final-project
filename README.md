# cs420-final-project

## Description

This is our final project for CS 420: Bio Inspired Computing. The goal of this project is to use a Hopfield network to recall MIDI files after adding noise to them.

## Usage

This project is a script that takes in a MIDI file and outputs a csv file with data and a plot of the recall versus the noise. I have a script (script.py)
that you can run that automatically runs this. All you need to do is:

```bash
# Install the required packages
pip install scikit-learn matplotlib pandas numpy mido
python script.py
```

You will the csv file and the plot under the outputs directory.

## Outputs

The plots we already have are in the outputs/ directory, containing csv files and plots for each midi file. For an example of what a noisy MIDI file sounds like, we have included samples under demo/ and demo2/ directories.

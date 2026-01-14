"""
Standalone script (command-line version) to compute individual Alpha Peak Frequency (iAPF)
from resting-state EEG.

iAPF is defined as the highest peak in the 8–12 Hz range of the periodic
spectrum (i.e., the spectrum after aperiodic component removal using FOOOF).

Author: Raphael Bordas <bordasraph@gmail.com>
Last update: Jan 2025

Input
=====
The input is an EEG resting-state recording in a format readable by MNE.
See <https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html> for a list
of accepted formats.

Output
======
The mean and SD of the iapf of sub-XX are then displayed on the standard output.
It is also possible to save the csv by specifying the output path (argument --output)

Usage
=====
1. Check that the configuration variables are correct
2. Ensure you have installed the alpha_detection package as described in the readme
3. Go to command-line and run from the scripts directory
```
python iapf_detection_cli.py --input path/to/your/raw-data/sub-XX
```

Requirements
============
This script uses the get_band_peak function from the alpha_detection package.
"""
import argparse
import mne
import pandas as pd
import numpy as np
from fooof import FOOOFGroup

from alpha_detection.fooof_functions import get_band_peak

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

## TODO: check the channel names correspond to your system
# Channels list from (Bordas & van Wassenhove, 2026), suitable to 
# a 32-ch system following the international system
parietal_channels = [
    "C3", "CP5", "CP1", "Pz", "P3", "P4", "P8", "CP6", "CP2"
]

## TODO: check that the minimum and maximum time points (in seconds)
## are consistent with the way the resting-state was recorded
tmin, tmax = 0, 119

fooof_settings = dict(
    peak_width_limits=(1.0, 8.0),
    max_n_peaks=6,
    min_peak_height=0.01,
    peak_threshold=2.0,
    aperiodic_mode="fixed",
)

freq_range = (1, 40)
alpha_band = (8, 12)


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compute individual Alpha Peak Frequency (iAPF) from EEG data"
)
parser.add_argument("--input", type=str, required=True, help="Path to EEG file")
parser.add_argument("--output", type=str, required=False, 
                    help="Path for the CSV output. If not provided, the results is just displayed on the standard output.")
parser.add_argument("--verbose", action="store_true", help="Enable verbose MNE output")

args = parser.parse_args()

fname = args.input
out_fname = args.output
verbose = args.verbose


# ---------------------------------------------------------------------
# Minimal preprocessing of the EEG data
# ---------------------------------------------------------------------

raw = mne.io.read_raw(fname, preload=True, verbose=verbose)

if tmin < raw.times[0] or tmax > raw.times[-1]:
    raise ValueError("tmin or tmax are outside the time points of the provided EEG data.")
raw.crop(tmin, tmax)

raw.set_eeg_reference("average", projection=False, verbose=verbose)

# Check that all requested parietal channels are present in the data
missing = [ch for ch in parietal_channels if ch not in set(raw.ch_names)]
if missing:
    raise ValueError(f"Missing expected EEG channels: {missing}")
raw.pick(parietal_channels)

raw.filter(1.0, 45.0, verbose=verbose)
raw.resample(250, verbose=verbose)


# ---------------------------------------------------------------------
# Power spectrum estimation
# ---------------------------------------------------------------------

# create 5-s epochs (0.2 Hz frequency resolution)
epochs = mne.make_fixed_length_epochs(
    raw,
    duration=5.0,
    overlap=0.0,
    preload=True,
    verbose=verbose,
)

# compute the PSD averaged over epochs
spectrum = epochs.compute_psd(
    method="multitaper",
    fmin=1.0,
    fmax=45.0,
    n_jobs=1,
    verbose=verbose,
)
psds, freqs = spectrum.get_data(exclude=[], return_freqs=True)
psds = psds.mean(axis=0)  # shape: (n_channels, n_freqs)


# ---------------------------------------------------------------------
# FOOOF
# ---------------------------------------------------------------------

fg = FOOOFGroup(**fooof_settings, verbose=False)
fg.fit(freqs, psds, freq_range=freq_range, n_jobs=1)


# ---------------------------------------------------------------------
# iAPF extraction
# ---------------------------------------------------------------------

iapf_results = pd.concat([
    get_band_peak(fg.get_fooof(ch_idx), band=alpha_band)
    for ch_idx in range(len(parietal_channels))
])
iapf_results.index = parietal_channels


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

print("=" * 48)
print("iAPF RESULTS")
print("=" * 48)
print(f"File          : {fname}")
print(f"Channels (n)  : {len(iapf_results)}")

valid_peaks = iapf_results["peak_freq"].dropna()

if valid_peaks.empty:
    print("iAPF          : No alpha peak detected")
else:
    print(f"iAPF (mean)   : {valid_peaks.mean():.2f} Hz")
    print(f"iAPF (SD)     : {valid_peaks.std():.2f} Hz")

print("=" * 48)

if out_fname:
    iapf_results.to_csv(out_fname)
    print(f"Saved to      : {out_fname}")

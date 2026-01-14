""" Computation of the oscillatory spectrum, based on the IRASA method.

This is usually done between 1 and 45Hz, to extract the oscillatory component from the aperiodic one (the 1/f trend).

This first part of the script compute the oscillatory spectrum for all participants and blocks. The second part
plots the output spectrum with confidence intervals and cluster averages.

Input files
-----------
- <sid>/<sid>_<block_name>-epo.fif

Output files
------------
- <sid>/<sid>_<block_name>_irasa_psds_<window_duration>s_all_sensors.npy
- <sid>/<sid>_<block_name>_irasa_freqs_<window_duration>s_all_sensors.npy
"""
# %% Configuration
import datetime
import os

import numpy as np
import mne

import matplotlib
from tqdm import tqdm

import alpha_detection.example_config as config
from alpha_detection.irasa import IRASA

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 8

# a process is a tuple (participant_id, condition) and correspond
# to a single file
processes = [
    ("S2", "resting"),
    ("S3", "resting"),
    ("S4", "resting"),
]

# those variables are the same for all subjects
info = mne.io.read_info(config.info_path)
sigsens = info.ch_names

hset_label = "hset=np.arange(1.1, 2.95, 0.05)"  # used for logging
hset = np.arange(1.1, 2.95, 0.05)
n_fft, n_per_seg, n_overlap = 2500, 2500, 0
irasa = IRASA(
    fs=info["sfreq"],
    hset=np.arange(1.1, 2.95, 0.05),
    n_jobs=-1,
    n_fft=2500,
    n_per_seg=2500,
    n_overlap=0
)

# to select only a subset of epochs
epo_max = None
window_label = ""

# %% IRASA computation for all participants

for pid, c in tqdm(processes):
    epos = mne.read_epochs(
        config.epo_data / pid / f"{pid}_ses-01_{c}_proc-clean_epo.fif",
        preload=True,
        verbose=False
    )
    if epo_max is not None and len(epos) < epo_max:
        print(f"Skipping {pid} {c} due to insufficient epochs")
    data = epos.get_data(copy=True)[:epo_max]
    fitted = irasa.fit(data, f_range=[1, 45], log10=True)
    freqs = fitted[0]
    PSDS_SB = np.array(fitted[1:])

    # ensure output directory exists
    if not os.path.exists(config.irasa_path / pid):
        os.mkdir(config.irasa_path / pid)

    np.save(config.irasa_path / pid / f"{pid}_{c}_irasa_psds{window_label}_all_sensors.npy", PSDS_SB)
    np.save(config.irasa_path / pid / f"{pid}_{c}_irasa_freqs{window_label}_all_sensors.npy", freqs)

now = datetime.datetime.now()
out = f"""-------------- IRASA analysis --------------
Datetime:               \t {now.strftime('%Y-%m-%d %H:%M:%S')}
Number of runs:         \t {len(processes)}
Window:                 \t {window_label} (epo 1 to {epo_max})
Frequency range:        \t 1 - 45 Hz
Log10 transform:        \t True
IRASA parameters:       \t {hset_label}
Spectrum computation parameters:
\t- n_fft:              \t {n_fft}
\t- n_per_seg:          \t {n_per_seg}
\t- n_overlap:          \t {n_overlap}
--------------------------------------------
"""
print(out)
with open(config.irasa_path / f"irasa_run_{window_label}.txt", "w") as f:
    f.write(out)

""" Example config containing all the required environment variables
"""
from collections import OrderedDict
from pathlib import Path

from fooof.data import FOOOFSettings

# definition of the usual frequency bands
bands = OrderedDict(
    theta=(3, 7),
    alpha=(8, 12),
    beta=(15, 30),
)

### FOOOF CONFIGURATION
# see https://fooof-tools.github.io/fooof/auto_tutorials/plot_04-MoreFOOOF.html#settings-attributes
fooof_settings = FOOOFSettings(
    peak_width_limits=(1.0, 8.0),
    max_n_peaks=6,
    min_peak_height=0.01,
    peak_threshold=2.0,
    aperiodic_mode='fixed'
)

freq_range = [1, 40]

### IRASA CONFIGURATION
# => see alpha_detection.irasa.spectrum_parameters for default parameters of IRASA computations

### PATH MANAGEMENT
# location of input data
epo_data = Path("/home/rb266841/Bureau/test_alpha_peak_detection/derivatives/mne-preproc/eeg")
psd_data = Path("/home/rb266841/Bureau/test_alpha_peak_detection/psds")
freq_base_fname = str(psd_data / "{method}_freqs.npy")
psds_base_fname = str(psd_data / "{pid}/{pid}_ses-01_{condition}_{method}_psds.npy")

# location of the info object (e.g., one participant)
info_path = epo_data / "P02" / "P02_ses-01_resting_proc-clean_epo.fif"

# location of output data
fooof_path = Path("/home/rb266841/Bureau/test_alpha_peak_detection/results/fooof")
irasa_path = Path("/home/rb266841/Bureau/test_alpha_peak_detection/results/irasa")

### CHANNEL NAMES
# /!\ should be identical to the sensor name and order from the psds /!\
# below we provide only 27 from a standard 32-ch BrainAmp system
ch_names = [
    'Fp1', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz',
    'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'Oz'
]

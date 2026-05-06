""" Example script to run FOOOF peak detection on resting-state data """
import os

from alpha_detection.fooof_pipeline import FOOOFPipeline
from alpha_detection import example_config
from alpha_detection.fooof_report import PeakDetector

# a process is a tuple (participant_id, condition) and correspond
# to a single file
processes = [
    ("S2", "resting"),
    ("S3", "resting"),
    ("S4", "resting"),
]

fooofPip = FOOOFPipeline(config=example_config, processes=processes)

# ensure output paths exist
for sub, cond in processes:
    if not os.path.exists(example_config.fooof_path / sub):
        os.mkdir(example_config.fooof_path / sub)

# FOOOF models over the whole recording duration
fooofPip.run_fooof_all_processes()
fooofPip.save(f"fooof_characteristics_all_sensors_5s_epo.csv")

# FOOOF models over 6-epoch windows (i.e., 30 seconds)
fooofPip.run_fooof_all_processes(n_epochs=6)
fooofPip.save(f"fooof_characteristics_all_sensors_5s_epo_6epochs_win.csv")

# FOOOF models over the first 10 epochs (i.e., the first 60 seconds of each recording)
fooofPip.run_fooof_all_processes(epo_start=0, epo_end=10)
fooofPip.save(f"fooof_characteristics_all_sensors_5s_epo_0_to_10_epo.csv")

peaks = PeakDetector(
    processes,
    example_config.fooof_path,
    bands={"alpha": [8, 12]},
    ch_names=example_config.ch_names
)
peaks.detect_all_recordings()
dfs = peaks.detect_all_recordings()
dfs.to_csv(example_config.fooof_path / "fooof_alpha_peaks_all_sensors_5s_epo.csv")

peaks.report()

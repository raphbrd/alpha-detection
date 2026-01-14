import os

from alpha_detection.fooof_pipeline import FOOOFPipeline
from alpha_detection import example_config

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

# %% Imports and config
import numpy as np
import pandas as pd
import mne
import tqdm
from scipy import signal

from alpha_detection.irasa import compute_aperiodic_statistics
import alpha_detection.example_config as config

bands = config.bands

# a process is a tuple (participant_id, condition) and correspond
# to a single file
processes = [
    ("sub-0003", "task-rest"),
    ("sub-0004", "task-rest"),
    ("sub-0006", "task-rest"),
]

# those variables are the same for all subjects
info = mne.io.read_info(config.info_path)

# to select only a subset of epochs
# should be identical to run_irasa_step_01.py
epo_max = None
window_label = ""

# %% IRASA on alpha and theta band
outs = []
all_chs = info.ch_names
n_chs = len(all_chs)

for pid, block in tqdm.tqdm(processes):
    # first dimension : original psd
    # second dimension : aperiodic spectrum
    # third dimension : oscillatory spectrum (= original - aperiodic)
    irasa_out = np.load(config.irasa_path / pid / f"{pid}_{block}_irasa_psds{window_label}_all_sensors.npy")
    freqs = np.load(config.irasa_path / pid / f"{pid}_{block}_irasa_freqs{window_label}_all_sensors.npy")
    osc_spectrum = irasa_out[2]

    dfs = []
    for band in bands:
        freqs_mask = np.logical_and(freqs >= bands[band][0], freqs <= bands[band][1])
        freqs_ranged = freqs[freqs_mask]

        for sen_idx, sensor in enumerate(all_chs):
            x = osc_spectrum[sen_idx, freqs_mask]

            # peaks detection
            peaks, _ = signal.find_peaks(x)

            peak_freq, peak_pw = np.nan, np.nan
            if len(peaks) > 0:
                prominences = signal.peak_prominences(x, peaks)[0]
                idx_max_prominence = np.argmax(prominences)
                idx_best_peak = peaks[idx_max_prominence]

                peak_freq = freqs_ranged[idx_best_peak]
                peak_pw = x[idx_best_peak]

            params, ss_res, r2 = compute_aperiodic_statistics(freqs, irasa_out[1, sen_idx])

            df = pd.DataFrame({
                # General info
                "NIP": pid,
                "block": block,
                "sensor": sensor,
                "band": band,

                # Info on the most prominent peak
                "peak_freq": np.round(peak_freq, 4),  # limited frequency resolution
                "peak_pw": peak_pw,

                # Total power in the band
                "total_pw": x.sum(),
                "avg_pw": x.mean(),
                "int_pw": np.trapz(x, x=freqs_ranged),

                # Total number of peaks
                "n_band_peaks": len(peaks),

                # Aperiodic info
                "aperiodic_intercept": params[0],
                "aperiodic_slope": params[1],
                "ap_fit_r2": r2,
            }, index=[sen_idx])
            dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    out.to_csv(config.irasa_path / pid / f"{pid}_{block}_irasa_stats{window_label}.csv", index=False)
    outs.append(out)

all_pids = pd.concat(outs)
all_pids.to_csv(config.irasa_path / f"irasa_stats{window_label}_detailed_per_channel.csv", index=False)

all_pids_avg = all_pids.groupby(["NIP", "block", "band"], sort=False).mean(numeric_only=True)
all_pids_avg.to_csv(config.irasa_path / f"irasa_stats{window_label}_avg_all_channels.csv", index=True)

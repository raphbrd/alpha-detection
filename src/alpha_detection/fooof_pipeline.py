"""
Parametrize PSDs with FOOOF algorithm
=====================================

@author: Raphael Bordas

Compute FOOOF group model for each condition
One fooof group gather models for each electrode for one tuple (pid, condition), by keeping the order of the psds

All fooof groups are saved in the fooof_path directory.

There are three ways to compute FOOOF models currently:
- over the whole epochs => use run_fooof_all_processes(processes)
- over sliding windows of n_epochs length => use run_fooof_all_processes(processes, n_epochs=n_epochs)
- over a specific window [epo_start, epo_end] => use run_fooof_all_processes(processes, epo_start=m, epo_end=n)

When not using a specific window, the fooof models are saved in the fooof_path directory with the extension 0_to_None.

Config requirements (e.g., example_config.py Python module)
- fooof_path
- freqs_base_fname
- psds_base_fname
- freq_range
- fooof_settings

The json files saved in this script can then be used by the PeakDetector in the report.py function

Input files
----------
- multitaper_freqs.npy
- <pid>/<pid>_<condition>_multitaper_psds.npy

Output files
-----------
- <sid>/<sid>_<condition>_fg_all_sensors.json
- <sid>/<sid>_all_sens_<epo_start>_to_<epo_end>.json
- <sid>/<sid>_<condition>_report_<epo_start>_to_<epo_end>.pdf
"""
import os
import logging
import numpy as np
import pandas as pd

from .fooof_functions import FOOOFGroup, parallelize


class FOOOFPipeline(object):

    def __init__(self, config, processes):
        """ Initialize the FOOOF pipeline

        Parameters
        ----------
        config: module
            A python module containing all the config variables required for this pipeline.
        processes: list
            A list of tuple (participant_id, condition_id) to process by this pipeline
        """
        self.processes = processes
        self.config = config
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # freqs are identical for all recordings
        self.freqs = np.load(self.config.freq_base_fname.format(method="multitaper"))

    def get_windowed_fooof_groups(self, n_epochs: int = None, epo_start: int = 0, epo_end: int = None):
        """ Retrieve fooof models computed on the windowed blocks as dataframes.
        Typically called after FOOOFPipeline.compute_windowed_fooof_model
        """
        if self.processes is None:
            raise ValueError("self.processes has not been set yet (likely because the models were not run).")
        dfs = []
        cols = ["pid", "block", "sen", "epo_start", "epo_end", "r_squared", "error", "ap_offset", "ap_exponent", "cf",
                "pw", "bw"]
        ch_names = self.config.ch_names

        for pid, block in self.processes:
            if n_epochs is not None:
                last_idx = 0
                file_exist = True
                # for each epoch range, we try to find the corresponding fooof models. If it does not exist, it assumes
                # fooof computation did not go further
                while file_exist:
                    fg_fname = f"{pid}_{block}_fg_all_sens_{last_idx}_to_{last_idx + n_epochs}.json"

                    if not os.path.exists(self.config.fooof_path / pid / fg_fname):
                        print("File not found, passing pid: ", fg_fname)
                        break

                    fg = FOOOFGroup(*self.config.fooof_settings)
                    fg.load(file_name=fg_fname, file_path=self.config.fooof_path / pid)
                    df_tmp = fg.get_dataframe()

                    df_tmp["epo_start"] = last_idx
                    df_tmp["epo_end"] = last_idx + n_epochs - 1
                    df_tmp["pid"] = pid
                    df_tmp["block"] = block

                    # remap the sensor index to its name, with a trick to silence warnings in new pandas version
                    df_tmp["sen_name"] = ""
                    for idx, ch_name in enumerate(ch_names):
                        df_tmp.loc[df_tmp.sen == idx, "sen_name"] = ch_name
                    df_tmp = df_tmp.drop("sen", axis=1)
                    df_tmp = df_tmp.rename({"sen_name": "sen"}, axis="columns")

                    # append the dataframe with the columns ordered as in cols
                    dfs.append(df_tmp[cols])

                    last_idx += n_epochs
            else:
                # whether FOOOF was computed over the whole block or a window, the file name is the same
                fg_fname = f"{pid}_{block}_fg_all_sens_{epo_start}_to_{epo_end}.json"
                fg = FOOOFGroup(*self.config.fooof_settings)
                fg.load(file_name=fg_fname, file_path=self.config.fooof_path / pid)
                df_tmp = fg.get_dataframe()
                df_tmp["pid"] = pid
                df_tmp["block"] = block
                df_tmp["epo_start"] = epo_start
                df_tmp["epo_end"] = epo_end

                # remap the sensor index to its name, with a trick to silence warnings in new pandas version
                df_tmp["sen_name"] = ""
                for idx, ch_name in enumerate(ch_names):
                    df_tmp.loc[df_tmp.sen == idx, "sen_name"] = ch_name
                df_tmp = df_tmp.drop("sen", axis=1)
                df_tmp = df_tmp.rename({"sen_name": "sen"}, axis="columns")

                dfs.append(df_tmp[cols])

        return pd.concat(dfs, ignore_index=True)

    def compute_windowed_fooof_model(self, pid, condition, n_epochs: int = None):
        """ compute peak characteristics. Computation is performed by averaging the
        epochs over sliding windows of n_epochs length.

        Parameters
        ----------
        pid : str
        condition : str
        n_epochs: int
            number of epochs per window
        """
        psds = np.load(self.config.psds_base_fname.format(pid=pid, condition=condition, method="multitaper"))
        n = psds.shape[0] // n_epochs  # number of different windows
        last_idx = 0
        for i in range(n):
            self.compute_fooof_one_window(pid, condition, last_idx, last_idx + n_epochs)
            last_idx += n_epochs

    def compute_fooof_one_window(self, pid: str, condition: str, epo_start: int = 0, epo_end: int = None):
        """ compute peak characteristics over the averaged epochs for a single window

        Parameters
        ----------
        pid : str
        condition : str
        epo_start: int
            index of the first epoch of the window
        epo_end: int
            index of the last epoch of the window. If none (default), goes until the end of the block
        """
        psds = np.load(self.config.psds_base_fname.format(pid=pid, condition=condition, method="multitaper"))

        # default is the whole block duration, otherwise psds averaged on the epochs of the specified window
        # in any case: shape required for FOOOF = (n_sensors, n_freqs)
        # if epo_end is not None: the effective epochs used will be from `epo_start` idx to `epo_end` - 1 idx
        x = psds[epo_start:epo_end].mean(0)

        fg = FOOOFGroup(*self.config.fooof_settings, verbose=False)  # cf. base_config for the settings
        fg.fit(self.freqs, x, freq_range=self.config.freq_range,
               n_jobs=1)  # parallel computing is done in run_fooof_all_processes

        fg.save(file_name=f"{pid}_{condition}_fg_all_sens_{epo_start}_to_{epo_end}",
                file_path=self.config.fooof_path / pid, save_results=True, save_settings=True, save_data=True)
        fg.save_report(f"{pid}_{condition}_report_{epo_start}_to_{epo_end}",
                       file_path=self.config.fooof_path / pid)

    def run_fooof_all_processes(self, n_epochs: int = None, epo_start: int = 0, epo_end: int = None):
        """ Run FOOOF computation for all specified processes. If no n_epochs is provided, compute the model over
        the averaged psd. Otherwise, n_epochs will be used the length of the window

        If epo_start or epo_end are provided, all fooof models will be computed on the [epo_start, epo_end] window.
        n_epochs should be None in that case
        """
        if n_epochs is not None:
            # a window of n_epochs is specified: compute fooof by sliding this window over the epochs
            self.logger.info(f"Computing fooof models over the windowed epochs")
            parallelize(self.compute_windowed_fooof_model, n_jobs=-1, processes=self.processes, n_epochs=n_epochs)
        elif epo_start != 0 or epo_end is not None:
            # a window is specified
            self.logger.info(f"Computing fooof models over the specified window: epochs {epo_start} to {epo_end}")
            parallelize(self.compute_fooof_one_window, n_jobs=-1, processes=self.processes, epo_start=epo_start,
                        epo_end=epo_end)
        else:
            # compute fooof over the whole epochs
            self.logger.info(f"Computing fooof models over the whole epochs")
            parallelize(self.compute_fooof_one_window, n_jobs=-1, processes=self.processes)

    def save(self, filename: str, n_epochs: int = None, epo_start: int = 0, epo_end: int = None):
        # the correct files will be chosen in get_windowed_fooof_groups, whatever the type of fooof computation
        df = self.get_windowed_fooof_groups(n_epochs=n_epochs, epo_start=epo_start, epo_end=epo_end)
        df.to_csv(self.config.fooof_path / filename, index=False)

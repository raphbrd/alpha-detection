""" Generate CSV files with the peaks summary for each subject and block. """
import os

import mne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from fooof.analysis.periodic import get_band_peak as fooof_get_band_peak

from .fooof_functions import get_band_peak, FOOOFGroup


class PeakDetector:
    def __init__(self, recordings_list, fooof_path, bands: dict, ch_names, epo_start=0, epo_end=None, n_epochs=None,
                 verbose=False):

        if n_epochs is not None and epo_end is not None:
            raise ValueError("Both n_epochs and epo_end cannot be specified at the same time.")

        self.recordings_list = recordings_list
        self.fooof_path = fooof_path
        self.bands = bands
        self.epo_start = epo_start
        self.epo_end = epo_end
        self.n_epochs = n_epochs
        self.ch_names = ch_names
        self.verbose = verbose

    def get_fg_peaks(self, fg, band):
        """ Get the peaks from the FOOOFGroup object in a specific frequency band."""
        dfs = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            df = get_band_peak(fg.get_fooof(ind=ch_idx, regenerate=True), band, get_n_peaks=True)
            df.insert(0, "ch", ch_name)
            df.insert(1, "epo_start", self.epo_start)
            df.insert(2, "epo_end", self.epo_end)
            dfs.append(df)

        return pd.concat(dfs)

    def fooof_peak_summary(self, subject, recording_id, band):
        """ For each frequency band specified in self.bands, find the highest peak for every channel. All channels
        are assumed to be gathered in a single FOOOFGroup object, with the same order as in the mne.Info object.

        Parameters
        ----------
        subject : str
        recording_id : str
        band : tuple of float
            frequency band of interest
        """

        if self.n_epochs is not None:

            df_all_chs = []
            last_idx = 0
            file_exist = True
            # for each epoch range, we try to find the corresponding fooof models. If it does not exist, it assumes
            # fooof computation did not go further
            while file_exist:
                fg_fname = f"{subject}_{recording_id}_fg_all_sens_{last_idx}_to_{last_idx + self.n_epochs}.json"
                # print(fg_fname)
                if not os.path.exists(os.path.join(self.fooof_path, subject, fg_fname)):
                    if self.verbose:
                        print("File not found, passing pid: ", fg_fname)
                    break

                fg = FOOOFGroup()
                fg.load(file_name=fg_fname, file_path=os.path.join(self.fooof_path, subject))
                df = self.get_fg_peaks(fg, band)
                df["epo_start"] = last_idx
                df["epo_end"] = last_idx + self.n_epochs - 1
                df_all_chs.append(df)
                last_idx += self.n_epochs

            # format the sensors dataframe for the current subject and block (= 1 row for each sensor)
            df_all_chs = pd.concat(df_all_chs)
            df_all_chs.insert(0, "NIP", subject)
            df_all_chs.insert(1, "recording_id", recording_id)
        else:
            fg = FOOOFGroup()
            fg.load(file_name=f"{subject}_{recording_id}_fg_all_sens_{self.epo_start}_to_{self.epo_end}",
                    file_path=os.path.join(self.fooof_path, subject))
            df_all_chs = self.get_fg_peaks(fg, band)
            df_all_chs.insert(0, "NIP", subject)
            df_all_chs.insert(1, "recording_id", recording_id)

        return df_all_chs

    def detect_all_recordings(self) -> pd.DataFrame:
        """ Generate CSV files with the peaks summary for each subject and block. """
        dfs = []
        desc = f"{len(self.bands.keys())} bands"
        if self.n_epochs is not None:
            desc += f", {self.n_epochs} epo. win."
        else:
            desc += f", epo {self.epo_start}-{self.epo_end}"
        for s, r in tqdm(self.recordings_list, desc=desc):
            for band_name, band_range in self.bands.items():
                df = self.fooof_peak_summary(s, r, band_range)
                df.insert(2, "band", band_name)
                dfs.append(df)
        results = pd.concat(dfs, ignore_index=True)

        return results

    def report(self):
        report = mne.Report("Spectral analysis of EEG signals", verbose=False)

        if not os.path.exists(os.path.join(self.fooof_path, "figures")):
            os.makedirs(os.path.join(self.fooof_path, "figures"))

        for s, r in tqdm(self.recordings_list, desc="FOOOF plots per ch."):
            fg = FOOOFGroup()
            fg.load(file_name=f"{s}_{r}_fg_all_sens_{self.epo_start}_to_{self.epo_end}",
                    file_path=os.path.join(self.fooof_path, s))

            fig, axes = plt.subplots(8, 4, sharex=True, sharey=True, figsize=(12, 12))
            axes = axes.flatten()
            for i in range(len(fg)):
                axes[i].axvspan(3, 7, facecolor="green", alpha=.2, edgecolor="none")
                axes[i].axvspan(8, 12, facecolor="red", alpha=.2, edgecolor="none")
                axes[i].axvspan(14, 30, facecolor="blue", alpha=.2, edgecolor="none")
                fm = fg.get_fooof(i)
                color = "C0" if fm.r_squared_ > 0.95 else "red"
                # axes[i].plot(fm.freqs, fm.fooofed_spectrum_, color=color)
                axes[i].plot(fm.freqs, fm._peak_fit, color=color)
                peak_params = fm.get_params("peak_params")
                if len(peak_params) > 0 and not np.isnan(peak_params).any():
                    theta_cf, theta_pw, _ = fooof_get_band_peak(peak_params, (3, 7), select_highest=True)
                    alpha_cf, alpha_pw, _ = fooof_get_band_peak(peak_params, (8, 12), select_highest=True)
                    beta_cf, beta_pw, _ = fooof_get_band_peak(peak_params, (14, 30), select_highest=True)

                    if ~np.isnan(theta_cf):
                        axes[i].plot(theta_cf, theta_pw, "o", markersize=4, color="green")
                    if ~np.isnan(alpha_cf):
                        axes[i].plot(alpha_cf, alpha_pw, "o", markersize=4, color="red")
                    if ~np.isnan(beta_cf):
                        axes[i].plot(beta_cf, beta_pw, "o", markersize=4, color="blue")

                    peak_params = [p for p in peak_params if p[0] != theta_cf and p[0] != alpha_cf and p[0] != beta_cf]
                    for p in peak_params:
                        # ap_point = np.interp(p[0], fm.freqs, fm._ap_fit)
                        axes[i].plot(p[0], p[1], "o", markersize=4, color="k")

                axes[i].set_xlabel(self.ch_names[i])
                axes[i].xaxis.set_label_position('top')
                axes[i].spines.right.set_visible(False)
                axes[i].spines.top.set_visible(False)
                if i % 4 != 0:
                    axes[i].get_yaxis().set_visible(False)
                    axes[i].spines.left.set_visible(False)
                else:
                    axes[i].set_xticks([10, 30])
                # axes[i].set_yticks([0, 0.5, 1])
            for i in range(len(fg), len(axes)):
                axes[i].set_axis_off()
            fig.supxlabel("Frequency (Hz)", fontsize=14)
            fig.supylabel("Power (a.u.)", fontsize=14)
            fig.tight_layout()
            plt.subplots_adjust(wspace=0.02)
            fig.savefig(os.path.join(self.fooof_path, "figures", f"{s}_{r}_fooof_peaks.png"))
            report.add_figure(fig=fig, title=f'[{s} - {r}] FOOOF peaks', image_format='png', tags=('FOOOF_peaks',))
            plt.close()

            fig, axes = plt.subplots(8, 4, sharex=True, sharey=True, figsize=(12, 12))
            axes = axes.flatten()
            for i in range(len(fg)):
                axes[i].axvspan(3, 7, facecolor="green", alpha=.2, edgecolor="none")
                axes[i].axvspan(8, 12, facecolor="red", alpha=.2, edgecolor="none")
                axes[i].axvspan(14, 30, facecolor="blue", alpha=.2, edgecolor="none")
                fm = fg.get_fooof(i)
                color = "C0" if fm.r_squared_ > 0.95 else "red"
                axes[i].plot(fm.freqs, fm.fooofed_spectrum_, color=color)
                peak_params = fm.get_params("peak_params")
                if len(peak_params) > 0 and not np.isnan(peak_params).any():
                    theta_cf, theta_pw, _ = fooof_get_band_peak(peak_params, (3, 7), select_highest=True)
                    alpha_cf, alpha_pw, _ = fooof_get_band_peak(peak_params, (8, 12), select_highest=True)
                    beta_cf, beta_pw, _ = fooof_get_band_peak(peak_params, (14, 30), select_highest=True)

                    if ~np.isnan(theta_cf):
                        ap_point = np.interp(theta_cf, fm.freqs, fm._ap_fit)
                        axes[i].plot(theta_cf, theta_pw + ap_point, "o", markersize=4, color="green")
                    if ~np.isnan(alpha_cf):
                        ap_point = np.interp(alpha_cf, fm.freqs, fm._ap_fit)
                        axes[i].plot(alpha_cf, alpha_pw + ap_point, "o", markersize=4, color="red")
                    if ~np.isnan(beta_cf):
                        ap_point = np.interp(beta_cf, fm.freqs, fm._ap_fit)
                        axes[i].plot(beta_cf, beta_pw + ap_point, "o", markersize=4, color="blue")

                    peak_params = [p for p in peak_params if p[0] != theta_cf and p[0] != alpha_cf and p[0] != beta_cf]
                    for p in peak_params:
                        ap_point = np.interp(p[0], fm.freqs, fm._ap_fit)
                        axes[i].plot(p[0], p[1] + ap_point, "o", markersize=4, color="k")

                axes[i].set_xlabel(self.ch_names[i])
                axes[i].xaxis.set_label_position('top')
                axes[i].spines.right.set_visible(False)
                axes[i].spines.top.set_visible(False)
                if i % 4 != 0:
                    axes[i].get_yaxis().set_visible(False)
                    axes[i].spines.left.set_visible(False)
                else:
                    axes[i].set_xticks([10, 30])
                # axes[i].set_yticks([0, 0.5, 1])
            for i in range(len(fg), len(axes)):
                axes[i].set_axis_off()
            fig.supxlabel("Frequency (Hz)", fontsize=14)
            fig.supylabel("Power (a.u.)", fontsize=14)
            fig.tight_layout()
            plt.subplots_adjust(wspace=0.02)
            fig.savefig(os.path.join(self.fooof_path, "figures", f"{s}_{r}_fooof_peaks_full_spectrum.png"))
            report.add_figure(fig=fig, title=f'[{s} - {r}] FOOOF peaks', image_format='png', tags=('FOOOF_full',))
            plt.close()
        report.save(os.path.join(self.fooof_path, "fooof_report.html"), overwrite=True)

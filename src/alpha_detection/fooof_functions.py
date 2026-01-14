import numpy as np
import pandas as pd

from mne.parallel import parallel_func
from fooof.core.errors import NoModelError
from fooof.core.info import get_indices
from fooof.analysis.periodic import get_band_peak as fooof_get_band_peak, get_highest_peak
from fooof.objs import FOOOFGroup as BaseFOOOFGroup


def parallelize(func, processes, n_jobs=1, **args):
    """ run a function in parallelized computation.

    Processes must be a list of either subjects (str) or tuple (sub, block)
    """
    if not isinstance(processes, list):
        raise ValueError("Processes must be a list")

    if isinstance(processes[0], str):
        parallel, run_func, _ = parallel_func(func, n_jobs=n_jobs, total=len(processes))
        parallel(run_func(sid, **args) for sid in processes)
    elif isinstance(processes[0], tuple):
        if len(processes[0]) != 2:
            raise ValueError("If processes are a list of tuples, each process must have a length of 2.")
        parallel, run_func, _ = parallel_func(func, n_jobs=n_jobs, total=len(processes))
        parallel(run_func(sid, block, **args) for sid, block in processes)
    else:
        raise ValueError("Processes must be a list of str or tuples")


def get_band_peak(fm, band, get_n_peaks=False):
    """ retrieve the highest peak in the given band and its parameters,
    in a dataframe with the columns : [peak_freq, peak_pw, peak_bw, r_squared, exponent, comment]

    Parameters
    ----------
        fm: FOOOF Object
        band: tuple (fmin, fmax) of the frequency band
        get_n_peaks: bool, default False
            if True, return the number of peaks found in the band. Otherwise, return n_peaks=1 if at least one
            peak was found in the band, or 0 if no peak was found.

    Returns
    -------
        pd.DataFrame
        - peak_freq: float
        - peak_pw: float
        - peak_bw: float
        - n_peaks: int
        - r_squared: float
        - ap_exponent: float
        - offset: float
        - no_peak: bool
        - bad_fit: bool
        - comment: str
    """
    if fm.r_squared_ > 0.95:
        # get all the alpha characteristics under the form [freq, power, bandwidth]
        # if no peak found, the function returns [nan, nan, nan]
        # if multiple peaks, return the highest (default behavior) if select_highest is True
        # else, return all the peaks with the first dimension being the number of peaks
        band_peaks = fooof_get_band_peak(fm.peak_params_, band, select_highest=not get_n_peaks)
        # get all the aperiodic params under the form [offset, epx]
        offset, exponent = fm.aperiodic_params_

        # no peak was found: setting everything to nan, except exponent that does not depend on the peak detection
        if True in np.isnan(band_peaks):
            return pd.DataFrame(dict(
                peak_freq=[np.nan],
                peak_pw=[np.nan],
                peak_bw=[np.nan],
                n_peaks=[0],
                r_squared=[fm.r_squared_],
                ap_exponent=[exponent],
                offset=[offset],
                no_peak=[True],
                bad_fit=[False],
                comment=[f"No peak found in the {band} band"]
            ))

        n_peaks = 1
        if get_n_peaks and band_peaks.ndim > 1:
            n_peaks = band_peaks.shape[0]
            band_peaks = get_highest_peak(band_peaks)
        peak_freq, peak_pw, peak_bw = band_peaks

        # the r_squared is OK and a peak was found
        return pd.DataFrame(dict(
            peak_freq=[peak_freq],
            peak_pw=[peak_pw],
            peak_bw=[peak_bw],
            n_peaks=n_peaks,
            r_squared=[fm.r_squared_],
            ap_exponent=[exponent],
            offset=[offset],
            no_peak=[False],
            bad_fit=[False],
            comment=["OK"]
        ))
    # if the r_squared is too low, all parameters are considered to be undefined
    # including the exponent, which is not defined if the model failed
    return pd.DataFrame(dict(
        peak_freq=[np.nan],
        peak_pw=[np.nan],
        peak_bw=[np.nan],
        n_peaks=[0],
        r_squared=[fm.r_squared_],
        ap_exponent=[np.nan],
        offset=[np.nan],
        no_peak=[False],
        bad_fit=[True],
        comment=[f"Goodness of fit too low"]
    ))


class FOOOFGroup(BaseFOOOFGroup):
    def __init__(self, *args, **kwargs):
        """Initialize object with desired settings."""

        BaseFOOOFGroup.__init__(self, *args, **kwargs)

    def get_params(self, name, col=None):
        """Return model fit parameters for specified feature(s).

        Parameters
        ----------
        name : {'aperiodic_params', 'peak_params', 'gaussian_params', 'error', 'r_squared'}
            Name of the data field to extract across the group.
        col : {'CF', 'PW', 'BW', 'offset', 'knee', 'exponent'} or int, optional
            Column name / index to extract from selected data, if requested.
            Only used for name of {'aperiodic_params', 'peak_params', 'gaussian_params'}.

        Returns
        -------
        out : ndarray
            Requested data.

        Raises
        ------
        NoModelError
            If there are no model fit results available.
        ValueError
            If the input for the `col` input is not understood.

        Notes
        -----
        Adapted from fooof.obj.group
        """

        if not self.has_model:
            raise NoModelError("No model fit results are available, can not proceed.")

        # Allow for shortcut alias, without adding `_params`
        if name in ['aperiodic', 'peak', 'gaussian']:
            name = name + '_params'

        # If col specified as string, get mapping back to integer
        if isinstance(col, str):
            col = get_indices(self.aperiodic_mode)[col]
        elif isinstance(col, int):
            if col not in [0, 1, 2]:
                raise ValueError("Input value for `col` not valid.")

        # Pull out the requested data field from the group data
        # As a special case, peak_params are pulled out in a way that appends
        #  an extra column, indicating which FOOOF run each peak comes from
        if name in ('peak_params', 'gaussian_params'):
            out = [np.insert(getattr(data, name), 3, index, axis=1)
                   for index, data in enumerate(self.group_results)]
            # This updates index to grab selected column, and the last column
            #  This last column is the 'index' column (FOOOF object source)
            if col is not None:
                col = [col, -1]
        else:
            out = np.array([getattr(data, name) for data in self.group_results])

        # Some data can end up as a list of separate arrays
        #   If so, concatenate it all into one 2d array
        if isinstance(out[0], np.ndarray):
            out = np.concatenate([arr.reshape(1, len(arr))
                                  if arr.ndim == 1 else arr for arr in out], 0)

        # Select out a specific column, if requested
        if col is not None:
            out = out[:, col]

        return out

    def get_dataframe(self):
        df_peaks = pd.DataFrame(self.get_params("peak_params"), columns=["cf", "pw", "bw", "sen"])

        df_aperiodic = pd.DataFrame(self.get_params("aperiodic"), columns=["ap_offset", "ap_exponent"])
        df_aperiodic["sen"] = df_aperiodic.index
        df = pd.DataFrame(self.get_results())[['r_squared', 'error']]
        df["sen"] = df.index
        df = pd.merge(df, df_aperiodic, on="sen")
        df_fg = pd.merge(df_peaks, df, on="sen")

        return df_fg

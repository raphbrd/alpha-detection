import fractions

import numpy as np

from mne.parallel import parallel_func
from mne.time_frequency import psd_array_welch
from scipy import signal

# default parameters for the IRASA spectrum computation
# can be overridden by passing them as keyword arguments to the IRASA class
spectrum_parameters = {
    "n_per_seg": 2500,
    "n_fft": 2500,
    "n_overlap": 1000,
    "average": "mean",
    "window": "hann"
}


def compute_aperiodic_statistics(freqs, aperiodic_spectrum):
    """ Compute the aperiodic fit (linear regression) of the aperiodic spectrum in the
    log-log space.

    To follow conventions in the rest of this code, freqs is assumed to be in Hz in linear space, and
    the aperiodic spectrum to already be in log space.

    Parameters
    ----------
    freqs : 1d array
        The frequency values of the power spectrum. Shape of (n_freqs,)
    aperiodic_spectrum : 1d array
        The aperiodic component of the power spectrum for a single channel. Shape of  n_freqs,)

    Returns
    -------
    params : 1d array
        The parameters of the linear fit, in the form [intercept, slope]
    ss_res : float
        The sum of squared residuals of the fit
    r2 : float
        The R^2 value of the fit
    """
    if aperiodic_spectrum.ndim != 1:
        raise ValueError("Aperiodic spectrum should be 1d array. This function only takes the data from a single "
                         "channel as input.")

    # reminder : the log transform for the spectrum was x = 10 * np.log10(x)
    xdata = np.log10(freqs)  # freqs are assumed to be in linear space
    ydata = aperiodic_spectrum / 10  # ensure freqs and spectrum had the same transform

    ##### aperiodic component fit #####
    # in log-log space, this is equivalent to fit a line
    # thus, aperiodic = beta @ [1 xdata] = beta_0 + beta_1 * xdata
    design_matrix = np.vstack([np.ones_like(xdata), xdata]).T
    params, ss_res, _, _ = np.linalg.lstsq(design_matrix, ydata, rcond=None)
    ss_res = np.squeeze(ss_res)
    r2 = 1 - ss_res / np.sum((ydata - ydata.mean()) ** 2)

    return params, ss_res, r2


class IRASA:
    """ Irregular Resampling Auto-Spectral Analysis (IRASA) of EEG data (Wen & Liu 2016).

    Adapted from the Python implementation of the yasa and neurodsp packages """

    def __init__(self, fs, hset=None, n_jobs=-1, **spectrum_kwargs):
        self.fs = fs
        self.n_jobs = n_jobs

        # checking default spectrum parameters
        self.spectrum_kwargs = dict(spectrum_parameters)
        self.spectrum_kwargs.update(spectrum_kwargs)

        if hset is None:
            hset = np.arange(1.1, 1.95, 0.05)
        self.hset = np.round(hset, 4)  # avoiding floating point precision errors

    def fit(self, data, f_range=None, epoch_range=None, log10=False):
        """ Compute the IRASA model on the epoch data.

        The log transform is 10 * log10(x) if log10 is True.

        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_channels, n_times)
            The epochs to compute the IRASA model to.
        f_range : tuple of float | None
            The frequency range to restrict the spectrum to.
        epoch_range : tuple of int | None
            The epoch range to restrict the spectrum to.
        log10 : bool
            If True, return the log10 of the power spectrum.

        Returns
        -------
        freqs : ndarray, shape (n_freqs,)
            The frequency values of the power spectrum.
        psd : ndarray, shape (n_channels, n_freqs)
            The original power spectrum averaged across epochs.
        psd_aperiodic : ndarray, shape (n_channels, n_freqs)
            The aperiodic component of the power spectrum.
        psd_periodic : ndarray, shape (n_channels, n_freqs)
            The periodic component of the power spectrum, equal to the original - aperiodic.
        """
        # Calculate the original spectrum across the whole signal
        psd, freqs = psd_array_welch(data, self.fs, verbose=False, n_jobs=self.n_jobs, **self.spectrum_kwargs)
        if epoch_range is not None:
            psd = psd[epoch_range[0]:epoch_range[1]]
        psd = psd.mean(0)

        # Calculate the resampled psds to get the aperiodic and periodic components
        parallel, p_fun, n_jobs = parallel_func(self.compute_resampled_spectrum, n_jobs=self.n_jobs, verbose=False)
        psds = parallel(
            p_fun(data, h, up, dn, epoch_range) for (h, up, dn) in self._get_resampled_factors()
        )
        psd_aperiodic = np.median(psds, axis=0)

        if log10:
            psd = 10. * np.log10(psd)
            psd_aperiodic = 10. * np.log10(psd_aperiodic)

        psd_periodic = psd - psd_aperiodic

        # Restrict spectrum to requested frequency range
        if f_range is not None:
            psds = np.array([psd_aperiodic, psd_periodic])
            f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
            freqs = freqs[f_mask]
            psd_aperiodic, psd_periodic = psds[..., f_mask]
            psd = psd[..., f_mask]

        if epoch_range is not None:
            psd_aperiodic = psd_aperiodic[epoch_range[0]:epoch_range[1]]
            psd_periodic = psd_periodic[epoch_range[0]:epoch_range[1]]

        return freqs, psd, psd_aperiodic, psd_periodic

    def _get_resampled_factors(self):
        h_vals = []
        for h_val in self.hset:
            h_rat = fractions.Fraction(str(h_val))
            up, dn = h_rat.numerator, h_rat.denominator
            h_vals.append((h_val, up, dn))
        return h_vals

    def compute_resampled_spectrum(self, sig, h_, up_, dn_, epoch_range=None):
        sig_up = signal.resample_poly(sig, up_, dn_, axis=-1)
        sig_dn = signal.resample_poly(sig, dn_, up_, axis=-1)

        # Calculate the power spectrum of the resampled signals
        psd_up, _ = psd_array_welch(sig_up, h_ * self.fs, verbose=False, n_jobs=1, **self.spectrum_kwargs)
        psd_dn, _ = psd_array_welch(sig_dn, self.fs / h_, verbose=False, n_jobs=1, **self.spectrum_kwargs)

        # geometric mean of h and 1/h and average over epochs
        spec = np.sqrt(psd_up * psd_dn)
        if epoch_range is not None:
            spec = spec[epoch_range[0]:epoch_range[1]]

        return spec.mean(0)

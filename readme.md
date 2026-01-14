# Individual alpha peak detection in EEG data

**Authors:** 
- Raphael Bordas <bordasraph@gmail.com>
- Virginie van Wassenhove <virginie.van.wassenhove@gmail.com>

**Associated publication**

Bordas & van Wassenhove (in press) Spontaneous Oscillatory Activity in Episodic Timing: An EEG Replication Study and Its Limitations. eNeuro

A typical spectrum in M/EEG data presents a peak between 8 and 12 Hz (alpha frequency band) of particularly high power. 
This repository provides tools to detect and parameterize this peak by separating periodic and aperiodic components.

## Methods Supported
1. **FOOOF (specparam):** Parameterizing neural power spectra (Donoghue et al., 2020). Our implementation is simply an extension of their package: [fooof](https://fooof-tools.github.io/fooof/)
2. **IRASA:** Separating fractal and oscillatory components (Wen & Liu, 2016). We provide a custom implementation of their method in the `src` folder.

## Installation

1. Clone the repo
2. Install the requirements through your favorite package manager, e.g.,
```
uv sync
source .venv/bin/activate # 
```
3. Install the python package as an editable
```{bash}
# From the root directory of this repo
pip install -e .
```

Warning: it is important to install this repo as an editable (flag -e), as you will most likely edit the config file.

## Usage

In the following code, we provide a Python implementation of both methods, along with
example scripts of how to run the code with an example dataset.

### Standalone script for fast iAPF detection

Script to use: `scripts/iapf_detection_cli.py`.

### Pipelines

1. Check all parameters in the provided config files, or create your own. The config files are located in the src directory for convenience.
2. Check the paths in the scripts.

## References

Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T, Lara AH, Wallis JD,
Knight RT, Shestyuk A, & Voytek B (2020). Parameterizing neural power spectra into periodic
and aperiodic components. Nature Neuroscience, 23, 1655-1665. DOI: [10.1038/s41593-020-00744-x](https://doi.org/10.1038/s41593-020-00744-x)

Wen, H., & Liu, Z. (2016). Separating fractal and oscillatory components in the power spectrum of neurophysiological 
signal. Brain topography, 29(1), 13-26. DOI: [10.1007/s10548-015-0448-0](https://doi.org/10.1007/s10548-015-0448-0)

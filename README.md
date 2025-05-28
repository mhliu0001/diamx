# Diamx: Dark matter Inference with Alternative Models for Xenon-based detectors 

[![CI](https://github.com/mhliu0001/diamx/actions/workflows/ci.yaml/badge.svg)](https://github.com/mhliu0001/diamx/actions/workflows/ci.yaml) [![Coverage Status](https://coveralls.io/repos/github/mhliu0001/diamx/badge.svg?branch=master&t=ODttn4)](https://coveralls.io/github/mhliu0001/diamx?branch=master)

The project aims to open the black box of the complicated analysis of XENON-based dark matter direct detection experiments. Using openly available data from publications and data releases, and with some assumptions that simplify the analysis, `diamx` can reproduce the standard spin-independent WIMP upper limits in publications. You can also fit your own dark matter model (only nuclear recoil models are supported now) and see whether the experiments are in favor of your model. `diamx` also gives you the freedom to combine experiments to give better constraints.

Currently we support XENONnT (SR0 & SR1) and LUX-ZEPLIN (WS2022, WS2024).

## Installation

The project has not yet been uploaded to `PyPI` yet, so you need to clone the source files to your computer and use `pip` to install.

```bash
cd diamx
pip install -e .
```

## Basic Usage

All you need is a `json` configuration file, and the energy deposition spectrum of your dark matter model in xenon. Some examples are shown in the `config` folder.

`benchmark.ipynb` reproduces the WIMP limits in literature, and is a good place to start. 

## Data Files

If not specified, all XENONnT SR0 related files come from [[1]](#xenonnt_sr0), all XENONnT SR1 related files are from [[2]](#xenonnt_sr1), all LUX-ZEPLIN WS2022 related files from [[3]](#lz_ws2022), and all LUX-ZEPLIN WS2024 related files from [[4]](#lz_ws2024).

| Data File                                                    | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `c14_spectrum.csv`                                           | Spectrum of $^{14}$C beta decay, calculated with [BetaShape](https://github.com/IAEA-NSDDNetwork/BetaShape/) [[5]](#beta_shape). Units: energy [keV], differential rate [a.u.] |
| `tritium_spectrum.csv`                                       | Spectrum of $^3$H beta decay from KATRIN [[6]](#katrin_tritium). Units: energy [keV], differential rate [a.u.] |
| `xe136_spectrum.csv`                                         | Spectrum of $^{136}$Xe double beta decay from [nucleartheory.yale.edu](https://nucleartheory.yale.edu/double-beta-decay-phase-space-factors) [[7]](#xe136_double_beta). Units: energy [keV], differential rate [a.u.] |
| `wimp_{6,40,200,1000}GeV.{csv,json}`                         | Spectrum of the SI WIMP-nucleus scattering in the standard halo model, assuming cross section of $1\times 10^{-44}\text{cm}^2$. Calculated with [wimprates](https://github.com/JelleAalbers/wimprates) [[8]](#wimprates). Units: energy [keV], differential rate [events/ton/year] |
| `xenonnt_sr{0,1a,1b}_wimp_data.csv`                          | WIMP data for the XENONnT experiment. The files are not official and are extracted from figures. A few events may be lost during extraction. Units: cS1 [PE], cS2 [PE] |
| `xenonnt_sr{0,1}_wimp_ci.csv`                                | SI WIMP 90% confidence interval for the XENONnT experiment. Note that this is without applying PCL to ensure an apple-to-apple comparison between `diamx` and literature results. Units: WIMP mass [GeV], total cross section [$\text{cm}^2$] |
| `xenonnt_sr{0,1a,1b}_eff_{lower,median,upper}.csv`           | NR signal (peak reconstruction + event building & selection) efficiency as a function of energy. Note that this is not the total efficiency with ROI efficiency added, because `diamx` handles ROI automatically. The lower and upper efficiency curves are obtained by assuming the same relative uncertainty with the total efficiency. Since the curves are overlapping at small energies, the uncertainties in the region come from extrapolation, so there may be some discrepancies between this file and the true efficiency. |
| `xenonnt_sr1b_ac_{1,2}sigma.csv`                             | The contours of the accidental coincidence (AC) component in the XENONnT experiment, extracted from literature. These are used to generate a contour-driven AC template. For now SR1a and SR1b share the same AC template. Units: cS1 [PE], cS2 [PE] |
| `template_XENONnT_sr1_ac_contour_driven.h5`                  | The contour-driven AC template for XENONnT SR1. To generate this file, run `ac_template_contour.ipynb`. This is not an official file from the XENON collaboration and involves a lot of guesswork, but fortunately it works well in reproducing WIMP upper limits. |
| `xenonnt_sr0_s1_recon_eff_{lower,median,upper}.csvPart of this work is derived from three open-source projects: appletree, nest and alea. The diamx package uses modified source code from these projects. Please see LICENSE.{package_name} for the corresponding license.` | S1 reconstruction efficiency as a function of number of detected photons. In `diamx` a toy model is used to reconstruct number of detected photons for an event, and then the S1 reconstruction efficiency is applied. The same map is applied for both SR0 and SR1. The map is extracted from Fig. 5 in [[9]](#xenonnt_analysis_paper_1). Units: number of photons detected [dimensionless], efficiency [dimensionless]. |
| `xenonnt_sr0_neutron_spectrum.csv`                           | NR spectrum of the neutron background in the XENONnT experiment. The file comes from Fig. 3 in the XENONnT sensitivity paper [[10]](#xenonnt_sensitivity).  Units: recoil energy [keV], rate [1/(keV ton year)] |
| `xenonnt_sr{0,1}_{er,nr}_model.json`, `xenonnt_sr0_{er,nr}_par.json` | Detector and micro-physics related parameters used in the XENONnT experiment, adapted from appletree [[11]](#appletree). The micro-physics parameters come from [[12]](#xenonnt_analysis_paper_two) and are shared for both SR0 and SR1. $g_1$, $g_2$, gas gain, and the electric field comes from [[1]](#xenonnt_sr0) and [[2]](#xenonnt_sr1), although $g_2$ for SR1 has to be adjusted to 15.8 PE/electron in order to match with data without changing the physics model. |
| `lz_ws{2022,2024}_wimp_data.csv`                             | WIMP data for the LUX-ZEPLIN experiment. The files come from official data releases. Units: S1c [phd], S2c [phd] |
| `lz_ws{2022,2024}_wimp_ci.csv`                               | SI WIMP 90% confidence interval for the LZ experiment. Note that this is without applying PCL to ensure an apple-to-apple comparison between `diamx` and literature results. Units: WIMP mass [GeV], total cross section [$\text{cm}^2$] |
| `lz_ws{2022,2024}_eff_{lower,median,upper}.csv`              | NR signal (trigger + S1 threshold + SS + data analysis cuts) efficiency as a function of energy. Note that this is not the total efficiency with ROI added, because `diamx` handles ROI automatically. The lower and upper efficiency curves are obtained by assuming the same relative uncertainty with the total efficiency. Since the curves are overlapping at small energies, the uncertainties in the region come from extrapolation, so there may be some discrepancies between this file and the true efficiency. |
| `LZ_WS2024.hh`                                               | Header file for NEST v2.4.0 [[13]](#nest)[[14]](#nest_package) in order to model LZ WS2024 detector. This file comes from the data release in [[4]](#lz_ws2024), and only naming is changed to avoid conflict with `LZ_SR1.hh`. |

## Acknowledgements

Part of this work is derived from three open-source projects: [appletree](https://github.com/XENONnT/appletree), [nest](https://github.com/NESTCollaboration/nest) and [alea](https://github.com/XENONnT/alea). The `diamx` package uses modified source code from these projects. Please see `LICENSE.{package_name}` for the corresponding license.

## References

<a id="xenonnt_sr0">[1]</a> E. Aprile et al., First Dark Matter Search with Nuclear Recoils from the XENONnT Experiment, [Phys. Rev. Lett. **131**, 041003 (2023)](https://doi.org/10.1103/PhysRevLett.131.041003).

<a id="xenonnt_sr1">[2]</a> E. Aprile et al., WIMP Dark Matter Search Using a 3.1 Tonne $\times$ Year Exposure of the XENONnT Experiment, [arXiv:2502.18005](https://doi.org/10.48550/arXiv.2502.18005).

<a id="lz_ws2022">[3]</a> J. Aalbers et al., First Dark Matter Search Results from the LUX-ZEPLIN (LZ) Experiment, [Phys. Rev. Lett. **131**, 041002 (2023)](https://doi.org/10.1103/PhysRevLett.131.041002).

<a id="lz_ws2024">[4]</a> J. Aalbers et al., Dark Matter Search Results from 4.2 Tonne-Years of Exposure of the LUX-ZEPLIN (LZ) Experiment, [arXiv:2410.17036](https://doi.org/10.48550/arXiv.2410.17036).

<a id="beta_shape">[5]</a> Mougeot, Atomic exchange correction in forbidden unique beta transitions, [Applied Radiation and Isotopes **201**, 111018 (2023)](https://doi.org/10.1016/j.apradiso.2023.111018).

<a id="katrin_tritium">[6]</a> M. Kleesiek et al., $$\upbeta $$-Decay spectrum, response function and statistical model for neutrino mass measurements with the KATRIN experiment, [Eur. Phys. J. C **79**, 204 (2019)](https://doi.org/10.1140/epjc/s10052-019-6686-7).

<a id="xe136_double_beta">[7]</a> J. Kotila and F. Iachello, Phase-space factors for double-${\beta}$ decay, [Phys. Rev. C **85**, 034316 (2012)](https://doi.org/10.1103/PhysRevC.85.034316).

<a id="wimprates">[8]</a> J. Aalbers, B. Pelssers, Joran R. Angevaare, and K. D. Morå, JelleAalbers/wimprates: v0.5.0, [10.5281/zenodo.7636982(2023)](https://doi.org/10.5281/zenodo.7636982).

<a id="xenonnt_analysis_paper_one">[9]</a> E. Aprile et al., XENONnT analysis: Signal reconstruction, calibration, and event selection, [Phys. Rev. D **111**, 062006 (2025)](https://doi.org/10.1103/PhysRevD.111.062006).

<a id="xenonnt_sensitivity">[10]</a> E. Aprile et al., Projected WIMP sensitivity of the XENONnT dark matter experiment, [J. Cosmol. Astropart. Phys. **2020**, 031 (2020)](https://doi.org/10.1088/1475-7516/2020/11/031).

<a id="appletree">[11]</a> D. Xu, Z. Xu, M. Liu, Y. Ma, J. R. Angevaare, L. Yuan, and L. Hoetzsch, XENONnT/appletree: v0.5.4,  [10.5281/zenodo.14794915 (2025)](https://doi.org/10.5281/zenodo.14794915).

<a id="xenonnt_analysis_paper_two">[12]</a> E. Aprile et al., XENONnT WIMP Search: Signal & Background Modeling and Statistical Inference, [arXiv:2406.13638](https://doi.org/10.48550/arXiv.2406.13638).

<a id="nest">[13]</a> M. Szydagis, N. Barry, K. Kazkaz, J. Mock, D. Stolp, M. Sweany, M. Tripathi, S. Uvarov, N. Walsh, and M. Woods, NEST: a comprehensive model for scintillation yield in liquid xenon, [J. Inst. **6**, P10002 (2011)](https://doi.org/10.1088/1748-0221/6/10/P10002).

<a id="nest_package">[14]</a> M. Szydagis et al., Noble Element Simulation Technique: v2.4.0beta, [10.5281/zenodo.8215927 (2023)](https://doi.org/10.5281/zenodo.8215927).
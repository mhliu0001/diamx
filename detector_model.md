# Detector Modeling in diamx

**Minghao Liu, May 29, 2025**

In `diamx`, we implement complete background and signal models for the XENONnT and LUX-ZEPLIN experiments. This note describes how these models are constructed and implemented.

## XENONnT Model

### Common Model Components

We analyze events in the **cS1–cS2** space. Every physics model comprises three modules:

1. **Micro-physics model**: Converts deposited energy into scintillation photons and ionization electrons using the so-called NESTv1 model.
   
2. **Detector response model**: Simulates photon/electron transport, S1/S2 signal formation and corrections, and reconstruction biases.
   
3. **Efficiency model**: Accounts for S1 reconstruction efficiency and the overall event-building + selection efficiency.

All three modules are constructed with `appletree`; we add custom code in `diamx.appletree` for NR micro-physics.

#### Micro-physics Model

We use the NEST v1 paramatrization, as detailed in Appendices A and B of [[5]](#xenonnt_analysis_paper_two).  ER and NR emission parameters are taken from the marginal posteriors in Table II of [[5]](#xenonnt_analysis_paper_two). While the ER model is fully implemented via `appletree`.  Custom implementation is needed in `diamx.appletree` to match the same NEST v1 behavior for NR.

#### Detector Response Model

The detector model is explained in Appendix C of [[5]](#xenonnt_analysis_paper_two). For lack of input, we make the following simplifications:

- Disable **S1/S2 spatial corrections** (including electron-lifetime correction).  
- Disable any **reconstruction bias**.  

The model is implemented fully in `appletree`.

#### Efficiency Model

We split the efficiency into two parts:

- **S1 reconstruction efficiency**;
- **Event-building + selection efficiency**.

**For ER backgrounds:** We assume that the event-building + selection efficiency is flat in energy, and model the S1 reconstruction efficiency as a function of detected S1 photons using the map from Fig. 5 of [[6]](#xenonnt_analysis_paper_one). The value of the flat efficiency does not matter because we need to normalize the entire template by the input background rate.

**For NR signals:** We take the total efficiency curve as a function of NR energy **without ROI cut** from Fig. 2 of [[1]](#xenonnt_sr0).  Then, we decompose the total efficiency into the two parts mentioned above by the following steps:

1. Use `appletree` to map the S1-reconstruction efficiency vs detected S1 photons into a function of NR energy.  
2. Divide the published total efficiency by this reconstructed S1 reconstruction efficiency component to isolate the **event-building + selection efficiency**. 
3. Apply both effects independently in `diamx`.

### XENONnT SR0

In [[1]](#xenonnt_sr0), there are five backgrounds modeled: **ER**, **neutrons**, **CE$\nu$NS**, **AC**, and **surface**.

First, we argue that the **CE$\nu$NS** background can be neglected, because it only conrtibutes to $0.23 \pm 0.06$ events in the ROI, and only $0.022 \pm 0.006$ events are signal-like, compared to 152 observed events in the ROI and 3 signal-like events. 

Next, the AC background and the surface background use data-driven templates, and are not publicly available. Luckily, nearly all AC and surface events lie in the region of cS2 < 800PE, so these two backgrounds can be mitigated if we narrow the ROI to cS2 > 800PE.

The remaining two backgrounds are modeled: ER and neutrons. The ER background is modeled by a flat spectrum, and the neutron background uses the input spectrum from [[7]](#xenonnt_sensitivity). 



## References

<a id="xenonnt_sr0">[1]</a> E. Aprile et al., First Dark Matter Search with Nuclear Recoils from the XENONnT Experiment, [Phys. Rev. Lett. **131**, 041003 (2023)](https://doi.org/10.1103/PhysRevLett.131.041003).

<a id="xenonnt_sr1">[2]</a> E. Aprile et al., WIMP Dark Matter Search Using a 3.1 Tonne $\times$ Year Exposure of the XENONnT Experiment, [arXiv:2502.18005](https://doi.org/10.48550/arXiv.2502.18005).

<a id="lz_ws2022">[3]</a> J. Aalbers et al., First Dark Matter Search Results from the LUX-ZEPLIN (LZ) Experiment, [Phys. Rev. Lett. **131**, 041002 (2023)](https://doi.org/10.1103/PhysRevLett.131.041002).

<a id="lz_ws2024">[4]</a> J. Aalbers et al., Dark Matter Search Results from 4.2 Tonne-Years of Exposure of the LUX-ZEPLIN (LZ) Experiment, [arXiv:2410.17036](https://doi.org/10.48550/arXiv.2410.17036).

<a id="xenonnt_analysis_paper_two">[5]</a> E. Aprile et al., XENONnT WIMP Search: Signal & Background Modeling and Statistical Inference, [arXiv:2406.13638](https://doi.org/10.48550/arXiv.2406.13638).

<a id="xenonnt_analysis_paper_one">[6]</a> E. Aprile et al., XENONnT analysis: Signal reconstruction, calibration, and event selection, [Phys. Rev. D **111**, 062006 (2025)](https://doi.org/10.1103/PhysRevD.111.062006).

<a id="xenonnt_sensitivity">[7]</a> E. Aprile et al., Projected WIMP sensitivity of the XENONnT dark matter experiment, [J. Cosmol. Astropart. Phys. **2020**, 031 (2020)](https://doi.org/10.1088/1475-7516/2020/11/031).
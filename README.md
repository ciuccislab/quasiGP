# Project
## qGP-DRT: The quasi-Gaussian Process Distribution of Relaxation Times 

This repository contains some of the source code used for the paper titled *Probabilistic deconvolution of the distribution of relaxation times from multiple electrochemical impedance spectra*. Journal of Power Sources, 621, 235236. https://doi.org/10.1016/j.jpowsour.2024.235236. The article is available online at [Link](https://doi.org/10.1016/j.jpowsour.2024.235236) and in the [docs](docs) folder. 

# Introduction

Electrochemical impedance spectroscopy (EIS) is a widely used technique for investigating the properties of electrochemical materials and systems, but analyzing EIS data can be quite difficult [1]. The distribution of relaxation times (DRT) has emerged as promising non-parametric method [1,2]. Among DRT inversion techniques [3-5], those based on Gaussian processes (GP) are particularly advantageous because they offer uncertainty estimates for both EIS and DRT [6,7]. However, existing GP-based DRT methods are limited to analyzing one spectrum at a time [6,7]. This study introduces a new approach, the quasi-Gaussian process distribution of relaxation times, which enables the simultaneous analysis of multiple impedance spectra across different experimental conditions [8]. 


![image](https://github.com/user-attachments/assets/5d1d33b6-c1db-4fe3-a138-900531c97c5f)


# Dependencies
numpy

scipy

matplotlib

pandas

# Tutorials
1. **ex1_1xZARC Model.ipynb**: This notebook demonstrates how to deconvolve DRTs from multiple EIS spectra (from 1xZARC model) over a frequency range of 1E-2 to 1E6 Hz with 10 points per decade.
2. **ex2_2xZARC Model.ipynb**: This notebook shows how the qGP-DRT method captures distant timescales using 2xZARC models in series, over the same frequency range as in **ex1_1xZARC Model.ipynb**.
3. **ex3_LFP.ipynb**: This notebook examines a lithium-ion battery with an LiFePO4 (LFP) cathode, lithium-metal anode, and 1M LiPF6 in ethylene carbonate: a 1:1 v/v diethyl carbonate as an electrolyte. EIS data were collected from 0.1 Hz to 7 MHz at a 5C rate for cycles 30, 60, 90, and 120.

# Citation
```
@article{maradesa2024probabilistic,
  title={Probabilistic deconvolution of the distribution of relaxation times from multiple electrochemical impedance spectra},
  author={Maradesa, Adeleke and Py, Baptiste and Ciucci, Francesco},
  journal={Journal of Power Sources},
  volume={621},
  pages={235236},
  year={2024},
  publisher={Elsevier}
}

```

# References

[1] Ciucci, F. (2018). Modeling electrochemical impedance spectroscopy. Current Opinion in Electrochemistry.132-139. https://doi.org/10.1016/j.coelec.2018.12.003. 

[2] Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499. https://doi.org/10.1016/j.electacta.2015.09.097.

[3] Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study. Electrochimica Acta, 147, 470-482. https://doi.org/10.1016/j.electacta.2014.09.058.

[4] Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454. https://doi.org/10.1016/j.electacta.2015.03.123.

[5] Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. Electrochimica Acta, 247, 1117-1129. https://doi.org/10.1016/j.electacta.2017.07.050.

[6] Liu, J., & Ciucci, F. (2020). The Gaussian process distribution of relaxation times: A machine learning tool for the analysis and prediction of electrochemical impedance spectroscopy data. Electrochimica Acta, 135316. https://doi.org/10.1016/j.electacta.2019.135316.

[7] Maradesa, A., Py, B., Quattrocchi, E., & Ciucci, F. (2022). The probabilistic deconvolution of the distribution of relaxation times with finite Gaussian processes. Electrochimica Acta, 413, 140119. https://doi.org/10.1016/j.electacta.2022.140119.

[8] Maradesa, A., Py, B., & Ciucci, F. (2024). Probabilistic deconvolution of the distribution of relaxation times from multiple electrochemical impedance spectra. Journal of Power Sources621, 235236, . https://doi.org/10.1016/j.jpowsour.2024.235236

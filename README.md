# WhittleLikelihoodInference

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jakegrainger.github.io/WhittleLikelihoodInference.jl/stable/)
[![Build Status](https://github.com/JakeGrainger/WhittleLikelihoodInference.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JakeGrainger/WhittleLikelihoodInference.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/JakeGrainger/WhittleLikelihoodInference.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JakeGrainger/WhittleLikelihoodInference.jl)
[![DOI](https://zenodo.org/badge/438011702.svg)](https://zenodo.org/badge/latestdoi/438011702)

A julia package for Whittle and debiased Whittle likelihood inference. Provides the following functionality:

- Simulating Gaussian processes with a given autocovariance.
- Computation of the (negative) Whittle likelihood, its gradient and its Hessian (including with an arbitrary taper).
- Computation of the (negative) debiased Whittle likelihood, its gradient and its Hessian (including with an arbitrary taper).
- Support for reverse-mode automatic differentiation via [*ChainRules.jl*](https://github.com/JuliaDiff/ChainRules.jl).
- A fit function which uses solvers from [*Optim.jl*](https://github.com/JuliaNLSolvers/Optim.jl) with the option to use dpss tapers from [*DSP.jl*](https://github.com/JuliaDSP/DSP.jl).
- Plotting recipes for second-order properties of interest, including the spectral density function and autocovariance.
- Approximation of the autocovariance from the spectral density function (and for gradients and Hessians).
- Example models including 1D OU and 1D Matern, 2D correlated OU process and arbitrary dimensional Matern.
- Basic non-parametric estimators including the periodogram and Bartlett's method with plotting recipes.

## References

Sykulski, A.M., Olhede, S.C., Guillaumin, A.P., Lilly, J.M., Early, J.J. (2019). The debiased Whittle likelihood. *Biometrika* 106 (2), 251–266.

Grainger, J. P., Sykulski, A. M., Jonathan, P., and Ewans, K. (2021). Estimating the parameters of ocean wave
spectra. *Ocean Engineering*, 229:108934.

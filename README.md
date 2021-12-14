# WhittleLikelihoodInference

[![CI](https://github.com/JakeGrainger/WhittleLikelihoodInference.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JakeGrainger/WhittleLikelihoodInference.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/JakeGrainger/WhittleLikelihoodInference.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/JakeGrainger/WhittleLikelihoodInference.jl/actions/workflows/documentation.yml)

A julia package for Whittle and debiased Whittle likelihood inference. Provides the following functionality:

- Simulating Gaussian processes with a given autocovariance.
- Computation of the Whittle likelihood, its gradient and its Hessian (including with an arbitrary taper). 
- Computation of the Debiased Whittle likelihood, its gradient and its Hessian (including with an arbitrary taper).
- Plotting recipes for second-order properties of interest, including the spectral density function and autocovariance.
- Approximation of the autocovariance from the spectral density function (and for gradients and Hessians).
- Example models including 1D OU and 1D Matern, 2D correlated OU process and arbitrary dimentional Matern.
- Basic non-parametric estimators including the Periodogram and Bartlett's method with plotting recipes.

## References

Sykulski, A.M., Olhede, S.C., Guillaumin, A.P., Lilly, J.M., Early, J.J. (2019). The debiased Whittle likelihood. *Biometrika* 106 (2), 251–266.

Grainger, J. P., Sykulski, A. M., Jonathan, P., and Ewans, K. (2021). Estimating the parameters of ocean wave
spectra. *Ocean Engineering*, 229:108934.
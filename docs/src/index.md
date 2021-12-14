# WhittleLikelihoodInference Package

The [*WhittleLikelihoodInference*] package provides an efficient implementation of the Whittle likelihood and debiased Whittle likelihood for univariate and multivariate time series.
Features include:

- Simulating Gaussian processes with a given autocovariance.
- Whittle likelihood and computation of gradient and Hessian (including with an arbitrary taper). 
- Debiased Whittle likelihood and computation of gradient, Hessian and expected Hessian (including with an arbitrary taper).
- Plotting recipes for second-order properties of interest, including the spectral density function and autocovariance.
- Aproximation of the autocovariance from the spectral density function (and for gradients and Hessians).
- Example models including 1D OU and 1D Matern, 2D correlated OU process and arbitrary dimentional Matern.
- Basic non-parametric estimators including the Periodogram and Bartlett's method with plotting recipes.
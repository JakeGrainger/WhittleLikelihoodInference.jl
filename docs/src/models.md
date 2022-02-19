# Models

## Generic model framework

Models for time series in `WhittleLikelihoodInference` are specified by defining a new type which is a subtype of `TimeSeriesModel{D,T}` where `D` is the dimension of the time series and `T` is the type of the entry in the time series (i.e. Float64 for real valued and ComplexF64 for complex valued).

### Univariate
So if we wish to define a univariate time series model which is real valued, we would write the following:

```julia
struct MyUniModel <: TimeSeriesModel{1,Float64}
    α::Float64
    β::Float64
end
```

In this case, we created a model called `MyUniModel` with parameters `α` and `β`.
We then need to define the `npars` function to return the number of parameters, and `sdf` and `acv`.

### Multivariate
Similarly, if we wish to define a bivariate real model, we write

```julia
struct My2dModel <: TimeSeriesModel{2,Float64}
    α::Float64
    β::Float64
end
```
In this case, we again need to define the `npars` function to return the number of parameters, and now `add_sdf!` and `acv!`.
`add_sdf!` takes a preallocated vector as its first argument.
This vector is the lower triangle of the spectral density matrix at $\omega$, and the spectral density should be added to it. The lower triangle is encoded so that it goes down each column in order, i.e. in the case `D=2` we have $[s_{1,1},s_{2,1},s_{2,2}]$. The remaining entries are recovered by conjugate symmetry.
`acv!` is analogous, but the acv replaces the preallocated vector, and not added to it.

### Unknown autocovariance
Sometimes, the autocovariance may not be known analytically (or may be expensive to compute). In this case, the autocovariance can be approximated from the spectral density function. To specify such a model, do the following:

```julia
struct MyUnknownExample <: UnknownAcvTimeSeriesModel{D} end
```

Then only `npars` and `add_sdf!` (`sdf` if univariate) need to be specified.

### Gradients and Hessians
To use gradient and Hessian features the functions `grad_add_sdf!`, `grad_acv!`, `hess_add_sdf!` and `hess_acv!` need to be specified (the acv versions should not be specified if the model is a `UnknownAcvTimeSeriesModel`.)

For the gradient, the preallocated vector has first dimension the size of the lower triangle of the spectral density matrix, and second dimension the number of parameters. The Hessian is similar, but the second dimension is the lower triangle of the Hessian matrix, encoded in the same was as the lower triangle of the spectral density matrix.

## Included models
This package includes some basic models with both gradients and Hessians specified.

### OU
The first model is a stationary OU process, with autocovariance

$$c_{X}(\tau) = \frac{\sigma^2}{\theta}\exp\{-\theta|\tau|\}$$

and spectral density function

$$f_{X}(\omega) = \frac{\sigma^2}{\pi(\theta^2+\omega^2)}.$$

The OU process can be constructed by using
```julia
OU(σ,θ)
```

### CorrelatedOU
Consider two independent OU processes $X_1$ and $X_2$ with the same $\sigma$ and $\theta$ parameters. Then define $Y_1 = X_1$ and $Y_2 = \rho X_1 + \sqrt{1-\rho^2}X_2$. Then we say that $Y=[Y_1,Y_2]^T$ is a correlated OU process. We have that

$$\mathbb{E}[Y_1(\tau)Y_1(0)] = c_{X}(\tau)$$

and

$\begin{align*} 
    \mathbb{E}[Y_2(\tau)Y_2(0)] &= \rho^2\mathbb{E}[X_1(\tau)X_1(0)] + (1-\rho^2)\mathbb{E}[X_2(\tau)X_2(0)]\\
    &= c_{X}(\tau)
\end{align*}$

and

$\begin{align*} 
    \mathbb{E}[Y_2(\tau)Y_1(0)] &= \rho c_{X}(\tau)
\end{align*}.$

Therefore, we have

$\begin{align*}
    c_{Y}(\tau) = \begin{bmatrix}
        c_X(\tau) & \rho c_X(\tau) \\
        \rho c_X(\tau) & c_X(\tau)
    \end{bmatrix},
\end{align*}$

and

$\begin{align*}
    f_{Y}(\omega) = \begin{bmatrix}
        f_X(\omega) & \rho f_X(\omega) \\
        \rho f_X(\omega) & f_X(\omega)
    \end{bmatrix}.
\end{align*}$

So we have taken two OU processes and correlated them with correlation parameter $\rho$. This correlated OU process can be constructed in julia with the following:
```julia
CorrelatedOU(σ,θ,ρ)
```

### Matern
The package also includes an implementation of the multivariate Matérn model described by Gneiting et al. (2010).
Care should be taken when using this model to ensure the correct bounds are placed on the parameter space.

### References
Gneiting, T., Kleiber, W., & Schlather, M. (2010). Matérn cross-covariance functions for multivariate random fields. Journal of the American Statistical Association, 105(491), 1167-1177.
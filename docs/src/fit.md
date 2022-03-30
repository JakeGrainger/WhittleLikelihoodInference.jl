# Fitting

The fit function can be used to fit a model to a recorded series with either the Whittle or debiased Whittle likelihoods using the interior point Newton method as implemented in [*Optim.jl*](https://github.com/JuliaNLSolvers/Optim.jl).

First load the package:
```@example
using WhittleLikelihoodInference
```

As a basic example, consider a univariate Gaussian process with JONSWAP spectral density function. We can simulate such a process with the following:
```@example
import Random # hide
Random.seed!(1234) # hide
using WhittleLikelihoodInference # hide
n = 2048
Δ = 1.0
nreps = 1
σ = 0.8
θ = 0.6
ts = simulate_gp(OU(σ,θ),n,Δ,nreps)[1]
```
In this case, we have simulated a Gaussian process with OU spectral density function with parameters `σ = 0.8, θ = 0.6` of length `2048` sampled every `1.0` seconds.
The function `simulate_gp` will simulate a vector of `nreps` series, which is why we recover the first of these to get one series.

We can now fit a model with the following:
```@example
import Random # hide
Random.seed!(1234) # hide
using WhittleLikelihoodInference # hide
n = 2048 # hide
Δ = 1.0 # hide
nreps = 1 # hide
σ = 0.8 # hide
θ = 0.6 # hide
ts = simulate_gp(OU(σ,θ),n,Δ,nreps)[1] # hide
x₀ = [σ,θ] .+ 0.1
res = fit(ts,model=OU,x₀=x₀)
```
Here `x₀` is the vector of initial parameter guesses.
The estimated parameter vector can be recovered by doing:
```@example
import Random # hide
Random.seed!(1234) # hide
using WhittleLikelihoodInference # hide
n = 2048 # hide
Δ = 1.0 # hide
nreps = 1 # hide
σ = 0.8 # hide
θ = 0.6 # hide
ts = simulate_gp(OU(σ,θ),n,Δ,nreps)[1] # hide
x₀ = [σ,θ] .+ 0.1 # hide
res = fit(ts,model=OU,x₀=x₀) # hide
x̂ = res.minimizer
```

The full example is:

```@example
import Random
Random.seed!(1234)
using WhittleLikelihoodInference
n = 2048
Δ = 1.0
nreps = 1
σ = 0.8
θ = 0.6
ts = simulate_gp(OU(σ,θ),n,Δ,nreps)[1]
x₀ = [σ,θ] .+ 0.1
res = fit(ts,model=OU,x₀=x₀)
x̂ = res.minimizer
```

For more options, see below:

```@docs
fit
```
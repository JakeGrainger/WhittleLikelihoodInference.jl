# Getting Started

## Installation
WhittleLikelihoodInference.jl can be installed by running the following command:
```julia
] add https://github.com/JakeGrainger/WhittleLikelihoodInference.jl
```

The package can then be used by running
```julia
using WhittleLikelihoodInference
```

## A basic example (univariate)
For a simple univariate example, consider a stationary OU process.
The stationary OU process has parameters $\theta$ and $\sigma$ with autocovariance

$$c(\tau) = \frac{\sigma^2}{\theta}\exp\{-\theta|\tau|\}$$

and spectral density function

$$f(\omega) = \frac{\sigma^2}{\pi(\theta^2+\omega^2)}.$$

The OU process is a continuous time Gaussian process, which we are interested in fitting to a sampled process which has finite length. We can simulate such a process with length $n$ and sampling interval $\Delta$ with the following code:

```@example
using WhittleLikelihoodInference # hide
import Random # hide
Random.seed!(1234) # hide
σ = 2
θ = 0.5
n = 1000
Δ = 0.5
nreps = 1
ts = simulate_gp(OU(σ,θ),n,Δ,nreps)[1]
```

In this case, we only wanted one series, but `simulate_gp` will simulate a vector of TimeSeries of length nreps. Therefore, we pull out the first index of this array as we only want one series.

Alternatively, simulation can be performed by running the following:
```@example
using WhittleLikelihoodInference # hide
import Random # hide
Random.seed!(1234) # hide
σ = 2 # hide
θ = 0.5 # hide
n = 1000 # hide
Δ = 0.5 # hide
import WhittleLikelihoodInference: GaussianProcess
X = GaussianProcess(OU(σ,θ),n,Δ)
ts = rand(X)
```

To perform Whittle likelihood inference we need to first create the objective function. This is done with the following:
```@example
using WhittleLikelihoodInference # hide
import Random # hide
Random.seed!(1234) # hide
σ = 2 # hide
θ = 0.5 # hide
n = 1000 # hide
Δ = 0.5 # hide
nreps = 1 # hide
ts = simulate_gp(OU(σ,θ),n,Δ,nreps)[1] # hide
whittle_objective = WhittleLikelihood(OU,ts)
```

The resulting objective function is set up to work with `Optim` so we can use

```@repl
using WhittleLikelihoodInference # hide
import Random # hide
Random.seed!(1234) # hide
σ = 2 # hide
θ = 0.5 # hide
n = 1000 # hide
Δ = 0.5 # hide
nreps = 1 # hide
ts = simulate_gp(OU(σ,θ),n,Δ,nreps)[1] # hide
whittle_objective = WhittleLikelihood(OU,ts) # hide
whittle_objective([2, 0.5])
F,G,H = 0,zeros(2),zeros(2,2)
whittle_objective(F,G,H,[2, 0.5])
G
H
```


We can optimise this function with `Optim` by using

```@example
using WhittleLikelihoodInference # hide
import Random # hide
Random.seed!(1234) # hide
σ = 2 # hide
θ = 0.5 # hide
n = 1000 # hide
Δ = 0.5 # hide
nreps = 1 # hide
ts = simulate_gp(OU(σ,θ),n,Δ,nreps)[1] # hide
whittle_objective = WhittleLikelihood(OU,ts) # hide
using Optim
constraints = Optim.TwiceDifferentiableConstraints(zeros(2),fill(Inf,2))
obj = TwiceDifferentiable(Optim.only_fgh!(whittle_objective),ones(2))
res = optimize(obj, constraints, [1.5, 0.2], IPNewton())
```

The full example is:

```@example
import Random # hide
Random.seed!(1234) # hide
using WhittleLikelihoodInference, Optim
import WhittleLikelihoodInference: GaussianProcess
σ = 2
θ = 0.5
n = 1000
Δ = 0.5
ts = rand(GaussianProcess(OU(σ,θ),n,Δ))
whittle_objective = WhittleLikelihood(OU,ts)
constraints = Optim.TwiceDifferentiableConstraints(zeros(2),fill(Inf,2))
obj = TwiceDifferentiable(Optim.only_fgh!(whittle_objective),ones(2))
res = optimize(obj, constraints, [1.5, 0.2], IPNewton())
```

## A basic example (multivariate)

An analagous example for a multivariate process is a bivariate correlated OU process.
Consider two independent OU processes $X_1$ and $X_2$ with the same $\sigma$ and $\theta$ parameters. Then define $Y_1 = X_1$ and $Y_2 = \rho X_1 + \sqrt{1-\rho^2}X_2$. We have that

$$\mathbb{E}[Y_1(\tau)Y_1(0)] = \mathbb{E}[X_1(\tau)X_1(0)]$$

and

$\begin{align*} 
    \mathbb{E}[Y_2(\tau)Y_2(0)] &= \rho^2\mathbb{E}[X_1(\tau)X_1(0)] + (1-\rho^2)\mathbb{E}[X_2(\tau)X_2(0)]\\
    &= \mathbb{E}[X_1(\tau)X_1(0)]
\end{align*}$

and

$\begin{align*} 
    \mathbb{E}[Y_2(\tau)Y_1(0)] &= \rho\mathbb{E}[X_1(\tau)X_1(0)]
\end{align*}.$

So we have taken two OU processes and correlated them with correlation parameter $\rho$.
We can fit the Whittle likelihood to a simulated example with the below code.

```@example
import Random # hide
Random.seed!(1234) # hide
using WhittleLikelihoodInference, Optim
import WhittleLikelihoodInference: GaussianProcess
σ = 2
θ = 0.5
ρ = 0.7
n = 1000
Δ = 0.5
ts = rand(GaussianProcess(CorrelatedOU(σ,θ,ρ),n,Δ))
whittle_objective = WhittleLikelihood(CorrelatedOU,ts)
constraints = Optim.TwiceDifferentiableConstraints(zeros(3),[Inf,Inf,1])
obj = TwiceDifferentiable(Optim.only_fgh!(whittle_objective),ones(3))
res = optimize(obj, constraints, [1.5, 0.2, 0.5], IPNewton())
```
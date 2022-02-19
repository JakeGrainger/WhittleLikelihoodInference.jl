# Background

The primary concern of this package is parameter estimation using the Whittle likelihood for continuous time processes (though you can define models for discrete time if you wish).

The main concepts are as follows:

- We have some continuous-time stochastic process $\{X(t)\}_{t\in\mathbb{R}}$ where $X(t)\in\mathbb{R}^d$ for all $t\in\mathbb{R}$.
- We assume that this process is mean-zero and stationary, i.e. $\forall t \in \mathbb{R}$,
    1) $\mathbb{E}[X(t)] = 0,$
    2) $\mathbb{E}[X(\tau+t)X(t)^T] = \mathbb{E}[X(\tau)X(0)^T],$
    3) $\text{tr}(\mathbb{E}[X(t)X(t)^T]) < \infty.$
- The autocovariance is defined as $c(\tau)=\mathbb{E}[X(\tau)X(0)^T]$.
- Sampling $\{X(t)\}_{t\in\mathbb{R}}$ with a sampling interval of $\Delta$, results in a discrete-time stochastic process $\{X_{t\Delta}\}_{t\in\mathbb{Z}}$ where $X_{t\Delta} = X(t\Delta)$ for all $t\in\mathbb{Z}$.
- The spectral density function of the continuous-time process is
$$f(\omega)=\frac{1}{2\pi}\int_{-\infty}^{\infty} c(\tau) e^{-i\omega\tau}\text{d}\tau.$$
- The spectral density of the sampled process (also referred to as the aliased spectral density) is
$$f_\Delta(\omega)=\frac{\Delta}{2\pi}\sum_{\tau-\infty}^{\infty} c(\tau\Delta) e^{-i\omega\tau\Delta}.$$
- **Warning: Note that the convention for the Fourier transform used here (dividing by $2\pi$) is essentially arbitrary, and different authors use different conventions. This only matters if you add a new model, or expect the output of a spectral density function to be the Fourier transform of the autocovariance under another convention.**
- We also have the inverse relations:
$\begin{align*}
	c(\tau) &= \int_{-\infty}^{\infty} f(\omega) e^{i\omega\tau} \text{d}\omega & \text{for }\tau\in\mathbb{R},\\
	c(\tau\Delta) &= \int_{-\pi/\Delta}^{\pi/\Delta} f_\Delta(\omega) e^{i\omega\tau\Delta} \text{d}\omega & \text{for }\tau\in\mathbb{Z}.
\end{align*}$
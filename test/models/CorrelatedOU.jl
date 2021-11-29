@testset "CorrelatedOU" begin
    θ₀ = [2.0,0.7,0.4]
    ω = 1.2
    τ = 3.0
    @test approx_gradient(θ -> sdf(CorrelatedOU(θ), ω), θ₀)      ≈ grad_sdf(CorrelatedOU(θ₀), ω)
    @test approx_hessian( θ -> grad_sdf(CorrelatedOU(θ), ω), θ₀) ≈ hess_sdf(CorrelatedOU(θ₀), ω)
    @test approx_gradient(θ -> acv(CorrelatedOU(θ), τ), θ₀)      ≈ grad_acv(CorrelatedOU(θ₀), τ)
    @test approx_hessian( θ -> grad_acv(CorrelatedOU(θ), τ), θ₀) ≈ hess_acv(CorrelatedOU(θ₀), τ)
end
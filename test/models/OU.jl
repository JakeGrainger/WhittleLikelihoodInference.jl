@testset "OU" begin
    θ₀ = [2.0,0.4]
    ω = 1.1
    τ = 2.0
    @test approx_gradient_uni(θ -> sdf(OU(θ), ω), θ₀) ≈ grad_sdf(OU(θ₀), ω)
    @test approx_hessian_uni( θ -> grad_sdf(OU(θ), ω), θ₀)                  ≈ hess_sdf(OU(θ₀), ω)
    @test approx_gradient_uni(θ -> acv(OU(θ), τ), θ₀) ≈ grad_acv(OU(θ₀), τ)
    @test approx_hessian_uni( θ -> grad_acv(OU(θ), τ), θ₀)                  ≈ hess_acv(OU(θ₀), τ)
end
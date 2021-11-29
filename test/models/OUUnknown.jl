@testset "OUUnknown" begin
    θ₀ = [2.0,0.4]
    model_known_acv = OU(θ₀)
    model_unknown_acv = OUUnknown{5}(θ₀)
    ω = 1.1
    τ = 2.0
    @test sdf(model_known_acv, ω)           = sdf(model_unknown_acv, ω)
    @test grad_sdf(model_known_acv, ω)      = grad_sdf(model_unknown_acv, ω)
    @test hess_sdf(model_known_acv, ω)      = hess_sdf(model_unknown_acv, ω)
    @test acv(model_known_acv, 100, 1)      ≈ acv(model_unknown_acv, 100, 1)
    @test grad_acv(model_known_acv, 100, 1) ≈ grad_acv(model_unknown_acv, 100, 1)
    @test hess_acv(model_known_acv, 100, 1) ≈ hess_acv(model_unknown_acv, 100, 1)
end
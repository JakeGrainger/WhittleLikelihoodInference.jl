@testset "CorrelatedOUUnknown" begin
    θ₀ = [3.0,0.8,0.2]
    model_known_acv = CorrelatedOU(θ₀)
    model_unknown_acv = WhittleLikelihoodInference.CorrelatedOUUnknown{5}(θ₀)
    ω = 1.1
    @test sdf(model_known_acv, ω)      == sdf(model_unknown_acv, ω)
    @test grad_sdf(model_known_acv, ω) == grad_sdf(model_unknown_acv, ω)
    @test hess_sdf(model_known_acv, ω) == hess_sdf(model_unknown_acv, ω)
end
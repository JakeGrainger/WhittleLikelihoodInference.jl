@testset "OUUnknown" begin
    θ₀ = [2.0,0.4]
    model_known_acv = OU(θ₀)
    model_unknown_acv = WhittleLikelihoodInference.OUUnknown{5}(θ₀)
    ω = 0.6
    @test sdf(model_known_acv, ω)      == sdf(model_unknown_acv, ω)
    @test grad_sdf(model_known_acv, ω) == grad_sdf(model_unknown_acv, ω)
    @test hess_sdf(model_known_acv, ω) == hess_sdf(model_unknown_acv, ω)
end
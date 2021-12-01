import WhittleLikelihoodInference: OUUnknown
@testset "OUUnknown" begin
    θ₀ = [2.0,0.4]
    model_known_acv = OU(θ₀)
    model_unknown_acv = OUUnknown{5}(θ₀)
    ω = 0.6
    @testset "Function sdf" begin
        @test sdf(model_known_acv, ω)      == sdf(model_unknown_acv, ω)
    end
    @testset "Gradient sdf" begin
        @test grad_sdf(model_known_acv, ω) == grad_sdf(model_unknown_acv, ω)
    end
    @testset "Hessian sdf" begin
        @test hess_sdf(model_known_acv, ω) == hess_sdf(model_unknown_acv, ω)
    end
    @testset "Error handling" begin
        @test_throws ArgumentError OUUnknown{2}(-1.0,1.0)
        @test_throws ArgumentError OUUnknown{2}(1.0,-1.0)
        @test_throws ArgumentError OUUnknown{2}(ones(1))
        @test_throws ArgumentError OUUnknown{2}(ones(3))
    end
end
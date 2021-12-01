import WhittleLikelihoodInference: CorrelatedOUUnknown
@testset "CorrelatedOUUnknown" begin
    θ₀ = [3.0,0.8,0.2]
    model_known_acv = CorrelatedOU(θ₀)
    model_unknown_acv = CorrelatedOUUnknown{5}(θ₀)
    ω = 1.1
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
        @test_throws ArgumentError CorrelatedOUUnknown{2}(-1.0,1.0,0.5)
        @test_throws ArgumentError CorrelatedOUUnknown{2}(1.0,-1.0,0.5)
        @test_throws ArgumentError CorrelatedOUUnknown{2}(1.0,1.0,-0.5)
        @test_throws ArgumentError CorrelatedOUUnknown{2}(1.0,1.0, 1.0)
        @test_throws ArgumentError CorrelatedOUUnknown{2}(fill(0.5,2))
        @test_throws ArgumentError CorrelatedOUUnknown{2}(fill(0.5,4))
    end
end
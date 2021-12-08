@testset "simulation" begin
    @test WhittleLikelihoodInference.GaussianProcess(OU(1.0,1.0), 10, 1.0).X isa WhittleLikelihoodInference.Distributions.MvNormal
    @test WhittleLikelihoodInference.GaussianProcess(CorrelatedOU(1.0,1.0,0.5), 10, 1.0).X isa WhittleLikelihoodInference.Distributions.MatrixReshaped
    ts1 = simulate_gp(OU(1.0,1.0), 10, 1.0)[1]
    ts2 = simulate_gp(CorrelatedOU(1.0,1.0,0.5), 10, 1.0)[1]
    @test ndims(ts1) == 1
    @test length(ts1) == 10
    @test ndims(ts2) == 2
    @test length(ts2) == 10
end
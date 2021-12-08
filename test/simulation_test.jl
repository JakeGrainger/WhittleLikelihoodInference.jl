@testset "simulation" begin
    @test WhittleLikelihoodInference.GaussianProcess(OU(1.0,1.0), 10, 1.0).X isa WhittleLikelihoodInference.Distributions.MvNormal
    @test WhittleLikelihoodInference.GaussianProcess(CorrelatedOU(1.0,1.0,0.5), 10, 1.0).X isa WhittleLikelihoodInference.Distributions.MatrixReshaped
end
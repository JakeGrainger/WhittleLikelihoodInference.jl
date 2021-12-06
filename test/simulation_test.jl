@testset "simulation" begin
    @test FiniteNormal(OU(1.0,1.0), 10, 1.0) isa WhittleLikelihoodInference.Distributions.MvNormal
    @test FiniteNormal(CorrelatedOU(1.0,1.0,0.5), 10, 1.0) isa WhittleLikelihoodInference.Distributions.MatrixReshaped
end
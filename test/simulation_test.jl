@testset "simulation" begin
    @test FiniteNormal(OU, 10, 1.0) isa WhittleLikelihoodInference.Distributions.MvNormal
    @test FiniteNormal(CorrelatedOU, 10, 1.0) isa WhittleLikelihoodInference.Distributions.MatrixReshaped
end
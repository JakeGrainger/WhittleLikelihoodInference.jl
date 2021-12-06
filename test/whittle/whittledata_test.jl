@testset "whittledata" begin
    @test_throws ArgumentError WhittleLikelihoodInference.WhittleData(CorrelatedOU, ones(10,3), 1.0)
    @test_throws ArgumentError WhittleLikelihoodInference.DebiasedWhittleData(CorrelatedOU, ones(10,3), 1.0)
end
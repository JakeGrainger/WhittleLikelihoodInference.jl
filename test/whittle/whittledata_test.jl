@testset "whittledata" begin
    @test_throws ArgumentError WhittleLikelihoodInference.WhittleData(CorrelatedOU, ones(10,3), 1.0)
    @test_throws ArgumentError WhittleLikelihoodInference.DebiasedWhittleData(CorrelatedOU, ones(10,3), 1.0)
    @test_throws ArgumentError WhittleLikelihoodInference.WhittleData(CorrelatedOU, ones(10,2), 1.0, taper = ones(10,2))
    @test_throws ArgumentError WhittleLikelihoodInference.DebiasedWhittleData(CorrelatedOU, ones(10,2), 1.0, taper = ones(10,2))
    @test_throws ArgumentError WhittleLikelihoodInference.WhittleData(CorrelatedOU, ones(10,2), 1.0, taper = ones(ComplexF64,10))
    @test_throws ArgumentError WhittleLikelihoodInference.DebiasedWhittleData(CorrelatedOU, ones(10,2), 1.0, taper = ones(ComplexF64,10))
end
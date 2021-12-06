@testset "debiasedwhittle" begin
    @test_throws ArgumentError DebiasedWhittleLikelihood(OU,ones(10),-1)
    @test_throws ArgumentError DebiasedWhittleLikelihood(OU,ones(10,2),1)
end
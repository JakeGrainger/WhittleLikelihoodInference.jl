@testset "standardwhittle" begin
    @test_throws ArgumentError WhittleLikelihood(OU,ones(10),-1)
    @test_throws ArgumentError WhittleLikelihood(OU,ones(10,2),1)
end
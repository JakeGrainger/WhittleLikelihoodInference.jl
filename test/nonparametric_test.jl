@testset "nonparametric" begin
    @test_throws ArgumentError Periodogram(ones(10), -1)
    @test_throws ArgumentError Periodogram(ones(10,2), -1)
end
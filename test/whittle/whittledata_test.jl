@testset "whittledata" begin
    @test_throws ArgumentError WhittleData(OU, ones(10,2), 1.0)
    @test_throws ArgumentError DebiasedWhittleData(OU, ones(10,2), 1.0)
end
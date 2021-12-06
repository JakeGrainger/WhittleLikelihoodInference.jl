@testset "whittledata" begin
    @test_throws ArgumentError WhittleData(OU(1.0,1.0), ones(10,2), 1.0)
    @test_throws ArgumentError DebiasedWhittleData(OU(1.0,1.0), ones(10,2), 1.0)
end
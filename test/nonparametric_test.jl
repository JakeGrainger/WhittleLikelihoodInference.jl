@testset "nonparametric" begin
    @test_throws ArgumentError Periodogram(ones(1000), -1)
    @test_throws ArgumentError Periodogram(ones(1000,2), -1)
    @test Periodogram(rand(1000),1/2) isa Periodogram
    @test BartlettPeriodogram(rand(1000),1/2) isa BartlettPeriodogram
    @test Periodogram(rand(1000,2),1/2) isa Periodogram
    @test BartlettPeriodogram(rand(1000,2),1/2) isa BartlettPeriodogram
    @test CoherancyEstimate(rand(1000,2),1/2) isa CoherancyEstimate
    @test Periodogram(rand(1000),1/2) + Periodogram(rand(1000),1/2) isa Periodogram
    @test BartlettPeriodogram(rand(1000),1/2) + BartlettPeriodogram(rand(1000),1/2) isa BartlettPeriodogram
    @test Periodogram(rand(1000,2),1/2) + Periodogram(rand(1000,2),1/2) isa Periodogram
    @test BartlettPeriodogram(rand(1000,2),1/2) + BartlettPeriodogram(rand(1000,2),1/2) isa BartlettPeriodogram
    @test CoherancyEstimate(rand(1000,2),1/2) + CoherancyEstimate(rand(1000,2),1/2) isa CoherancyEstimate
end
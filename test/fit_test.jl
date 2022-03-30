@testset "fit" begin
    θ = [1.0,1.0]
    θ2 = [1.0,1.0,1.0,1.0]
    @testset "Error handling" begin
        @test_throws ArgumentError fit(ones(10,2),1,model=OU,x₀=θ)
        @test_throws ArgumentError fit(ones(10,3),1,model=CorrelatedOU,x₀=[1.0,1.0,0.5])
        @test_throws ArgumentError fit(ones(10),1,model=OU,x₀=[θ;1.1])
        @test_throws ArgumentError fit(ones(10),1,model=OU,x₀=θ,lowerΩcutoff = 2, upperΩcutoff = 1)
        @test_throws ArgumentError fit(ones(10),1,model=OU,x₀=θ,x_lowerbounds = ones(3))
        @test_throws ArgumentError fit(ones(10),1,model=OU,x₀=θ,x_upperbounds = ones(3))
        @test_throws ArgumentError fit(ones(10),1,model=OU,x₀=θ,taper="Dpss_4")
    end
    @testset "fitting" begin
        ts = simulate_gp(OU(θ),1000,1.0,1)[1]
        @test fit(ts,model=OU,x₀=θ) isa WhittleLikelihoodInference.Optim.MultivariateOptimizationResults
        @test fit(ts,model=OU,x₀=θ,taper=:dpss_4) isa WhittleLikelihoodInference.Optim.MultivariateOptimizationResults
        @test fit(ts,model=OU,x₀=θ,taper="dpss_4.2",options=WhittleLikelihoodInference.Optim.Options(iterations=4000)) isa WhittleLikelihoodInference.Optim.MultivariateOptimizationResults        
        @test fit(ts.ts,ts.Δ,model=OU,x₀=θ,taper="dpss_4.2") isa WhittleLikelihoodInference.Optim.MultivariateOptimizationResults
        @test lowerbounds(TwoOU) == [lowerbounds(OU);lowerbounds(OU)]
        @test upperbounds(TwoOU) == [upperbounds(OU);upperbounds(OU)]
        ts2 = simulate_gp(TwoOU(θ2),1000,1.0,1)[1]
        @test fit(ts2.ts,ts2.Δ,model=TwoOU,x₀=θ2) isa WhittleLikelihoodInference.Optim.MultivariateOptimizationResults
    end
end
@testset "CorrelatedOU" begin
    θ₀ = [2.0,0.7,0.4]
    ω = 1.2
    τ = 3.0
    @testset "sdf" begin
        @testset "Gradient" begin
            @test approx_gradient(θ -> sdf(CorrelatedOU(θ), ω), θ₀)      ≈ grad_sdf(CorrelatedOU(θ₀), ω)
        end
        @testset "Hessian" begin
            @test approx_hessian( θ -> grad_sdf(CorrelatedOU(θ), ω), θ₀) ≈ hess_sdf(CorrelatedOU(θ₀), ω)
        end
    end
    @testset "acv" begin
        @testset "Gradient" begin
            @test approx_gradient(θ -> acv(CorrelatedOU(θ), τ), θ₀)      ≈ grad_acv(CorrelatedOU(θ₀), τ)
        end
        @testset "Hessian" begin
            @test approx_hessian( θ -> grad_acv(CorrelatedOU(θ), τ), θ₀) ≈ hess_acv(CorrelatedOU(θ₀), τ)
        end
    end
    @testset "Error handling" begin
        @test_throws ArgumentError CorrelatedOU(-1.0,1.0,0.5)
        @test_throws ArgumentError CorrelatedOU(1.0,-1.0,0.5)
        @test_throws ArgumentError CorrelatedOU(1.0,1.0,-0.5)
        @test_throws ArgumentError CorrelatedOU(1.0,1.0, 1.0)
        @test_throws ArgumentError CorrelatedOU(fill(0.5,2))
        @test_throws ArgumentError CorrelatedOU(fill(0.5,4))
    end
end
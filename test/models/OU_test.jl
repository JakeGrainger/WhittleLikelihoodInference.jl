@testset "OU" begin
    θ₀ = [2.0,0.4]
    ω = 1.1
    τ = 2.0
    @testset "sdf" begin
        @testset "Gradient" begin
            @test approx_gradient_uni(θ -> sdf(OU(θ), ω), θ₀)      ≈ grad_sdf(OU(θ₀), ω)
        end
        @testset "Hessian" begin
            @test approx_hessian_uni( θ -> grad_sdf(OU(θ), ω), θ₀) ≈ hess_sdf(OU(θ₀), ω)
        end
    end
    @testset "acv" begin
        @testset "Gradient" begin
            @test approx_gradient_uni(θ -> acv(OU(θ), τ), θ₀)      ≈ grad_acv(OU(θ₀), τ)
        end
        @testset "Hessian" begin
            @test approx_hessian_uni( θ -> grad_acv(OU(θ), τ), θ₀) ≈ hess_acv(OU(θ₀), τ)
        end
    end
    @testset "Error handling" begin
        @test_throws ArgumentError OU(-1.0,1.0)
        @test_throws ArgumentError OU(1.0,-1.0)
        @test_throws ArgumentError OU(ones(1))
        @test_throws ArgumentError OU(ones(3))
    end
end
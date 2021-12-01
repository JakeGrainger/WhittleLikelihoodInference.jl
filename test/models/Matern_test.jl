@testset "Matern" begin
    @testset "Matern2D" begin
        θ₂=[2.0,0.5,0.7,
            3/2,6.3,4.7,
            0.5,2.0,1.2]
        ω₂ = 1.1
        @testset "Gradient" begin
            @test approx_gradient(θ -> sdf(Matern2D(θ), ω₂), θ₂)      ≈ grad_sdf(Matern2D(θ₂), ω₂)
        end
        @testset "Hessian" begin
            @test approx_hessian( θ -> grad_sdf(Matern2D(θ), ω₂), θ₂) ≈ hess_sdf(Matern2D(θ₂), ω₂)
        end
    end
    @testset "Matern3D" begin
        θ₃=[2.0,0.5,0.3,1.7,0.6,3.2,
            3/2,6.3,2/5,4.7,1/2,2.1,
            0.5,2.0,1.2,3.2,1.0,2.5]
        ω₃ = 0.5
        @testset "Gradient" begin
            @test approx_gradient(θ -> sdf(Matern3D(θ), ω₃), θ₃)      ≈ grad_sdf(Matern3D(θ₃), ω₃)
        end
        @testset "Hessian" begin
            @test approx_hessian( θ -> grad_sdf(Matern3D(θ), ω₃), θ₃) ≈ hess_sdf(Matern3D(θ₃), ω₃)
        end
    end
    @testset "Matern4D" begin
        θ₄=[2.0,0.5,0.3,0.4,1.7,0.6,0.1,3.2,0.3,2.0,
            3/2,6.3,2/5,4.7,1/2,2.1,5.2,1.3,3/4,2/7,
            0.5,2.0,1.2,3.2,1.0,2.5,2.1,2.2,0.9,1.0]
        ω₄ = 1.4
        @testset "Gradient" begin
            @test approx_gradient(θ -> sdf(Matern4D(θ), ω₄), θ₄)      ≈ grad_sdf(Matern4D(θ₄), ω₄)
        end
        @testset "Hessian" begin
            @test approx_hessian( θ -> grad_sdf(Matern4D(θ), ω₄), θ₄) ≈ hess_sdf(Matern4D(θ₄), ω₄)
        end
    end
    @testset "Error handling" begin
        @test_throws ArgumentError Matern2D([[1.0,1.2]; fill(0.5,7)])
        @test_throws ArgumentError Matern2D([-1; fill(0.5,8)])
        @test_throws ArgumentError Matern2D(fill(0.5,8))
        @test_throws ErrorException Matern{2,5}(fill(0.5,9))
    end
end
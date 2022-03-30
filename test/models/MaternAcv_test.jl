import WhittleLikelihoodInference: MaternAcv, MaternAcv1D
@testset "MaternAcv" begin
    θ₁ = [2.0, 3/2, 0.5]
    ω₁ = 0.7
    θ₂=[2.0,0.5,0.7,
        3/2,6.3,4.7,
        0.5,2.0,1.2]
    ω₂ = 1.1
    @testset "Function sdf" begin
        @test sdf(MaternAcv1D(θ₁), ω₁)         == sdf(Matern1D(θ₁), ω₁)
        @test sdf(MaternAcv{2,3}(θ₂), ω₂)      == sdf(Matern2D(θ₂), ω₂)
    end
    @testset "Gradient sdf" begin
        @test_throws ErrorException grad_sdf(MaternAcv1D(θ₁), ω₁)
    end
    @testset "Hessian sdf" begin
        @test_throws ErrorException hess_sdf(MaternAcv1D(θ₁), ω₁)
    end
    @testset "Error handling" begin
        @test_throws ArgumentError MaternAcv1D(-1.0,1.0,1.0)
        @test_throws ArgumentError MaternAcv1D(1.0,-1.0,1.0)
        @test_throws ArgumentError MaternAcv1D(1.0,1.0,-1.0)
        @test_throws ArgumentError MaternAcv1D(ones(2))
        @test_throws ArgumentError MaternAcv1D(ones(4))
        
        @test_throws ArgumentError MaternAcv{2,3}([[1.0,1.2]; fill(0.5,7)])
        @test_throws ArgumentError MaternAcv{2,3}([-1; fill(0.5,8)])
        @test_throws ArgumentError MaternAcv{2,3}(fill(0.5,8))
        @test_throws ErrorException MaternAcv{2,5}(fill(0.5,9))
    end
    @testset "parameter bounds" begin
        @test_throws ErrorException lowerbounds(Matern2D)
        @test_throws ErrorException upperbounds(Matern2D)
        @test lowerbounds(Matern1D) == zeros(3)
        @test upperbounds(Matern1D) == fill(Inf,3)
    end
end
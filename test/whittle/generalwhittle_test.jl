@testset "generalwhittle" begin
    @testset "Gradient" begin
        @testset "univariate" begin
            @test_throws ArgumentError WhittleLikelihoodInference.grad_generalwhittle!(ones(2), ones(10), ones(2,10), ones(11))
            @test_throws ArgumentError WhittleLikelihoodInference.grad_generalwhittle!(ones(2), ones(10), ones(2,11), ones(10))
        end
        @testset "multivariate" begin
            S,I = fill([@SMatrix ones(2,2) for i in 1:10],2)
            ∇S = [@SMatrix ones(2,2) for j in 1:2, i in 1:10]
            @test_throws ArgumentError WhittleLikelihoodInference.grad_generalwhittle!(ones(2), S[1:9], ∇S[1:9], I)
            @test_throws ArgumentError WhittleLikelihoodInference.grad_generalwhittle!(ones(2), S[1:9], ∇S, I[1:9])
        end
    end
    @testset "Hessian" begin
        @testset "univariate" begin
            @test_throws ArgumentError WhittleLikelihoodInference.hess_generalwhittle!(ones(2), ones(10), ones(2,10), ones(3,10), ones(11))
            @test_throws ArgumentError WhittleLikelihoodInference.hess_generalwhittle!(ones(2), ones(10), ones(2,11), ones(3,10), ones(10))
            @test_throws ArgumentError WhittleLikelihoodInference.hess_generalwhittle!(ones(2), ones(10), ones(2,10), ones(3,11), ones(10))
        end
        @testset "multivariate" begin
            S,I = fill([@SMatrix ones(2,2) for i in 1:10],2)
            ∇S = [@SMatrix ones(2,2) for j in 1:2, i in 1:10]
            ∇²S = [@SMatrix ones(2,2) for j in 1:3, i in 1:10]
            @test_throws ArgumentError WhittleLikelihoodInference.hess_generalwhittle!(ones(2), S[1:9], ∇S[1:9], ∇²S[1:9], I)
            @test_throws ArgumentError WhittleLikelihoodInference.hess_generalwhittle!(ones(2), S[1:9], ∇S, ∇²S[1:9], I[1:9])
            @test_throws ArgumentError WhittleLikelihoodInference.hess_generalwhittle!(ones(2), S[1:9], ∇S[1:9], ∇²S, I[1:9])
        end
    end
    @testset "Hessian offdiag" begin
        @testset "univariate" begin
            @test_throws ArgumentError WhittleLikelihoodInference.offdiag_add_hess_generalwhittle!(ones(2), ones(10), ones(2,10), ones(2,10), ones(11))
            @test_throws ArgumentError WhittleLikelihoodInference.offdiag_add_hess_generalwhittle!(ones(2), ones(10), ones(2,11), ones(2,10), ones(10))
            @test_throws ArgumentError WhittleLikelihoodInference.offdiag_add_hess_generalwhittle!(ones(2), ones(10), ones(2,10), ones(2,11), ones(10))
        end
        @testset "multivariate" begin
            S,I = fill([@SMatrix ones(2,2) for i in 1:10],2)
            ∇S1,∇S2 = fill([@SMatrix ones(2,2) for j in 1:2, i in 1:10],2)
            @test_throws ArgumentError WhittleLikelihoodInference.offdiag_add_hess_generalwhittle!(ones(2), S[1:9], ∇S1[1:9], ∇S2[1:9], I)
            @test_throws ArgumentError WhittleLikelihoodInference.offdiag_add_hess_generalwhittle!(ones(2), S[1:9], ∇S1, ∇S2[1:9], I[1:9])
            @test_throws ArgumentError WhittleLikelihoodInference.offdiag_add_hess_generalwhittle!(ones(2), S[1:9], ∇S1[1:9], ∇S2, I[1:9])
        end
    end
end
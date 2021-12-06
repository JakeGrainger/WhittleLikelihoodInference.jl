@testset "expectedperiodogram" begin
    store = allocate_memory_EI_FGH(CorrelatedOU,100,1)
    storeuni = allocate_memory_EI_FGH(OU,100,1)
    @testset "Function" begin
        @test_throws ArgumentError WhittleLikelihoodInference._EI!(store.funcmemory,200,1)
        @test_throws ArgumentError WhittleLikelihoodInference._EI!(storeuni.funcmemory,200,1)
        @test length(EI(OU(1.0,1.0),10,1/2)) == 10
    end
    @testset "Gradient" begin
        @test_throws ArgumentError WhittleLikelihoodInference._grad_EI!(store.gradmemory,200,1)
        @test_throws ArgumentError WhittleLikelihoodInference._grad_EI!(storeuni.gradmemory,200,1)
    end
    @testset "Hessian" begin
        @test_throws ArgumentError WhittleLikelihoodInference._hess_EI!(store.hessmemory,200,1)
        @test_throws ArgumentError WhittleLikelihoodInference._hess_EI!(storeuni.hessmemory,200,1)
    end
end
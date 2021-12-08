@testset "expectedperiodogram" begin
    store = allocate_memory_EI_FGH(CorrelatedOU,100,1)
    storeuni = allocate_memory_EI_FGH(OU,100,1)
    badkernel = ones(50) # kernel should be length 200
    @testset "Function" begin
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(store.funcmemory.allocatedarray,badkernel)
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(storeuni.funcmemory.allocatedarray,badkernel)
        @test length(EI(OU(1.0,1.0),10,1/2)) == 10
    end
    @testset "Gradient" begin
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(store.gradmemory.allocatedarray,badkernel)
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(storeuni.gradmemory.allocatedarray,badkernel)
    end
    @testset "Hessian" begin
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(store.hessmemory.allocatedarray,badkernel)
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(storeuni.hessmemory.allocatedarray,badkernel)
    end
end
const TwoCorrelatedOU = CorrelatedOU+CorrelatedOU
const TwoOU = OU+OU
@testset "expectedperiodogram" begin
    store = allocate_memory_EI_FGH(CorrelatedOU,100,1)
    storeadd = allocate_memory_EI_FGH(TwoCorrelatedOU,100,1)
    storeuni = allocate_memory_EI_FGH(OU,100,1)
    storeadduni = allocate_memory_EI_FGH(TwoOU,100,1)
    badkernel = ones(50) # kernel should be length 200
    @testset "Function" begin
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(store.funcmemory.allocatedarray,badkernel)
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(storeuni.funcmemory.allocatedarray,badkernel)
        @test length(EI(OU(1.0,1.0),10,1/2)) == 10
        @test EI!(store,CorrelatedOU(1.0,1.0,0.5)) == nothing
        @test EI!(storeuni,OU(1.0,1.0)) == nothing
        @test EI!(storeadd,TwoCorrelatedOU([1.0,1.0,0.5,1.0,1.0,0.5])) == nothing
        @test EI!(storeadduni,TwoOU(ones(4))) == nothing
    end
    @testset "Gradient" begin
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(store.gradmemory.allocatedarray,badkernel)
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(storeuni.gradmemory.allocatedarray,badkernel)
        @test grad_EI!(store,CorrelatedOU(1.0,1.0,0.5)) == nothing
        @test grad_EI!(storeuni,OU(1.0,1.0)) == nothing
        @test grad_EI!(storeadd,TwoCorrelatedOU([1.0,1.0,0.5,1.0,1.0,0.5])) == nothing
        @test grad_EI!(storeadduni,TwoOU(ones(4))) == nothing
    end
    @testset "Hessian" begin
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(store.hessmemory.allocatedarray,badkernel)
        @test_throws DimensionMismatch WhittleLikelihoodInference.mult_allocated_kernel!(storeuni.hessmemory.allocatedarray,badkernel)
        @test hess_EI!(store,CorrelatedOU(1.0,1.0,0.5)) == nothing
        @test hess_EI!(storeuni,OU(1.0,1.0)) == nothing
        @test hess_EI!(storeadd,TwoCorrelatedOU([1.0,1.0,0.5,1.0,1.0,0.5])) == nothing
        @test hess_EI!(storeadduni,TwoOU(ones(4))) == nothing
    end
end
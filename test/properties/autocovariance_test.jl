struct TestAcvModel <: WhittleLikelihoodInference.TimeSeriesModel{2,Float64} end
struct TestAcvModelUni <: WhittleLikelihoodInference.TimeSeriesModel{1,Float64} end
struct TestUnknownAcv <: WhittleLikelihoodInference.UnknownAcvTimeSeriesModel{2,Float64} end
struct TestUnknownAcvUni <: WhittleLikelihoodInference.UnknownAcvTimeSeriesModel{1,Float64} end
WhittleLikelihoodInference.npars(::Type{TestAcvModel}) = 2
WhittleLikelihoodInference.npars(::Type{TestAcvModelUni}) = 2
WhittleLikelihoodInference.npars(::Type{TestUnknownAcv}) = 2
WhittleLikelihoodInference.npars(::Type{TestUnknownAcvUni}) = 2
WhittleLikelihoodInference.add_sdf!(out,::TestUnknownAcv,ω) = (out .+= 0.2; nothing)
WhittleLikelihoodInference.grad_add_sdf!(out,::TestUnknownAcv,ω) = (out .+= 0.2; nothing)
WhittleLikelihoodInference.hess_add_sdf!(out,::TestUnknownAcv,ω) = (out .+= 0.2; nothing)
WhittleLikelihoodInference.sdf(::TestUnknownAcvUni,ω) = 0.2
WhittleLikelihoodInference.grad_add_sdf!(out,::TestUnknownAcvUni,ω) = (out .+= 0.2; nothing)
WhittleLikelihoodInference.hess_add_sdf!(out,::TestUnknownAcvUni,ω) = (out .+= 0.2; nothing)
struct TestAcv2 <: WhittleLikelihoodInference.TimeSeriesModel{2,Float64} end
struct TestAcvUni2 <: WhittleLikelihoodInference.TimeSeriesModel{1,Float64} end
struct TestAcv3 <: WhittleLikelihoodInference.TimeSeriesModel{2,Float64} end
struct TestAcvUni3 <: WhittleLikelihoodInference.TimeSeriesModel{1,Float64} end
TestAcv2(x) = TestAcv2() # just to allow additive constructor to work
TestAcvUni2(x) = TestAcvUni2()
TestAcv3(x) = TestAcv3() # just to allow additive constructor to work
TestAcvUni3(x) = TestAcvUni3()
WhittleLikelihoodInference.npars(::Type{TestAcv2}) = 4
WhittleLikelihoodInference.npars(::Type{TestAcvUni2}) = 4
WhittleLikelihoodInference.npars(::Type{TestAcv3}) = 4
WhittleLikelihoodInference.npars(::Type{TestAcvUni3}) = 4
WhittleLikelihoodInference.acv!(out,::TestAcv2,τ::Number) = (out.=1.0;nothing)
WhittleLikelihoodInference.grad_acv!(out,::TestAcv2,τ::Number) = (out.=1.0;nothing)
WhittleLikelihoodInference.hess_acv!(out,::TestAcv2,τ::Number) = (out.=1.0;nothing)
WhittleLikelihoodInference.acv(::TestAcvUni2,τ::Number) = 1.0
WhittleLikelihoodInference.grad_acv!(out,::TestAcvUni2,τ::Number) = (out.=1.0;nothing)
WhittleLikelihoodInference.hess_acv!(out,::TestAcvUni2,τ::Number) = (out.=1.0;nothing)
WhittleLikelihoodInference.acv!(out,::TestAcv3,τ::Number) = (out.=2.0;nothing)
WhittleLikelihoodInference.grad_acv!(out,::TestAcv3,τ::Number) = (out.=2.0;nothing)
WhittleLikelihoodInference.hess_acv!(out,::TestAcv3,τ::Number) = (out.=2.0;nothing)
WhittleLikelihoodInference.acv(::TestAcvUni3,τ::Number) = 2.0
WhittleLikelihoodInference.grad_acv!(out,::TestAcvUni3,τ::Number) = (out.=2.0;nothing)
WhittleLikelihoodInference.hess_acv!(out,::TestAcvUni3,τ::Number) = (out.=2.0;nothing)
const AcvDouble = TestAcv2 + TestAcv3
const AcvDoubleUni = TestAcvUni2 + TestAcvUni3
@testset "autocovariance" begin
    @testset "Function" begin
        @testset "Error handling" begin
            @testset "Acv unspecified error" begin
                @test_throws ErrorException WhittleLikelihoodInference.acv!(ones(ComplexF64,3), TestAcvModel(), 1.0)
                @test_throws ErrorException acv(TestAcvModelUni(), 1.0)
            end
            @testset "Lags different size to preallocated memory error" begin
                store1 = allocate_memory_EI_F(TestAcvModel, 1000, 1)
                store2 = allocate_memory_EI_F(TestAcvModel, 10, 1)
                store1uni = allocate_memory_EI_F(TestAcvModelUni, 1000, 1)
                store2uni = allocate_memory_EI_F(TestAcvModelUni, 10, 1)
                @test_throws ArgumentError WhittleLikelihoodInference.acv!(store1.funcmemory, TestAcvModel(), store2.encodedtime) 
                @test_throws ArgumentError WhittleLikelihoodInference.acv!(store1uni.funcmemory, TestAcvModelUni(), store2uni.encodedtime) 
            end
            @testset "Custom lag when acv unknown error" begin
                @test_throws ArgumentError acv(TestUnknownAcv(), 1.0)
                @test_throws ArgumentError acv(TestUnknownAcvUni(), 1.0)
            end
            @testset "UnknownAcv" begin
                store = allocate_memory_EI_FGH(TestUnknownAcv, 100, 1)
                @test WhittleLikelihoodInference.acv!(store, TestUnknownAcv()) == nothing
                @test WhittleLikelihoodInference.grad_acv!(store, TestUnknownAcv()) == nothing
                @test WhittleLikelihoodInference.hess_acv!(store, TestUnknownAcv()) == nothing
                storeuni = allocate_memory_EI_FGH(TestUnknownAcvUni, 100, 1)
                @test WhittleLikelihoodInference.acv!(storeuni, TestUnknownAcvUni()) == nothing
                @test WhittleLikelihoodInference.grad_acv!(storeuni, TestUnknownAcvUni()) == nothing
                @test WhittleLikelihoodInference.hess_acv!(storeuni, TestUnknownAcvUni()) == nothing
            end
        end
        @testset "Value" begin
            @testset "Simple model" begin
                store1 = allocate_memory_EI_F(TestAcv2, 10, 1)
                store1uni = allocate_memory_EI_F(TestAcvUni2, 10, 1)
                WhittleLikelihoodInference.acv!(store1,TestAcv2())
                WhittleLikelihoodInference.acv!(store1uni,TestAcvUni2())
                @test all(store1.funcmemory.allocatedarray.==1.0)
                @test all(store1uni.funcmemory.allocatedarray.==1.0)
            end
            @testset "Additive model" begin
                store2 = allocate_memory_EI_F(AcvDouble, 10, 1)
                store2uni = allocate_memory_EI_F(AcvDoubleUni, 10, 1)
                WhittleLikelihoodInference.acv!(store2,AcvDouble(ones(8)))
                WhittleLikelihoodInference.acv!(store2uni,AcvDoubleUni(ones(8)))
                @test all(store2.store1.funcmemory.allocatedarray.==3.0)
                @test all(store2uni.store1.funcmemory.allocatedarray.==3.0)
            end
        end
    end
    @testset "Gradient" begin
        @testset "Error handling" begin
            @testset "Acv unspecified error" begin
                @test_throws ErrorException WhittleLikelihoodInference.grad_acv!(ones(ComplexF64,3,4), TestAcvModel(), 1.0)
                @test_throws ErrorException WhittleLikelihoodInference.grad_acv!(ones(ComplexF64,4),TestAcvModelUni(), 1.0)
            end
            @testset "Lags different size to preallocated memory error" begin
                store1 = allocate_memory_EI_FG(TestAcvModel, 1000, 1)
                store2 = allocate_memory_EI_FG(TestAcvModel, 10, 1)
                store1uni = allocate_memory_EI_FG(TestAcvModelUni, 1000, 1)
                store2uni = allocate_memory_EI_FG(TestAcvModelUni, 10, 1)
                @test_throws ArgumentError WhittleLikelihoodInference.grad_acv!(store1.gradmemory, TestAcvModel(), store2.encodedtime) 
                @test_throws ArgumentError WhittleLikelihoodInference.grad_acv!(store1uni.gradmemory, TestAcvModelUni(), store2uni.encodedtime) 
            end
            @testset "Custom lag when acv unknown error" begin
                @test_throws ArgumentError grad_acv(TestUnknownAcv(), 1.0)
                @test_throws ArgumentError grad_acv(TestUnknownAcvUni(), 1.0)
            end
        end
        @testset "Value" begin
            @testset "Simple model" begin
                store1 = allocate_memory_EI_FG(TestAcv2, 10, 1)
                store1uni = allocate_memory_EI_FG(TestAcvUni2, 10, 1)
                WhittleLikelihoodInference.grad_acv!(store1,TestAcv2())
                WhittleLikelihoodInference.grad_acv!(store1uni,TestAcvUni2())
                @test all(store1.gradmemory.allocatedarray.==1.0)
                @test all(store1uni.gradmemory.allocatedarray.==1.0)
            end
            @testset "Additive model" begin
                store2 = allocate_memory_EI_FG(AcvDouble, 10, 1)
                store2uni = allocate_memory_EI_FG(AcvDoubleUni, 10, 1)
                WhittleLikelihoodInference.grad_acv!(store2,AcvDouble(ones(8)))
                WhittleLikelihoodInference.grad_acv!(store2uni,AcvDoubleUni(ones(8)))
                @test all(store2.store1.gradmemory.allocatedarray.==1.0)
                @test all(store2.store2.gradmemory.allocatedarray.==2.0)
                @test all(store2uni.store1.gradmemory.allocatedarray.==1.0)
                @test all(store2uni.store2.gradmemory.allocatedarray.==2.0)
            end
        end
    end
    @testset "Hessian" begin
        @testset "Error handling" begin
            @testset "Acv unspecified error" begin
                @test_throws ErrorException WhittleLikelihoodInference.hess_acv!(ones(ComplexF64,3,4), TestAcvModel(), 1.0)
                @test_throws ErrorException WhittleLikelihoodInference.hess_acv!(ones(ComplexF64,4),TestAcvModelUni(), 1.0)
            end
            @testset "Lags different size to preallocated memory error" begin
                store1 = allocate_memory_EI_FGH(TestAcvModel, 1000, 1)
                store2 = allocate_memory_EI_FGH(TestAcvModel, 10, 1)
                store1uni = allocate_memory_EI_FGH(TestAcvModelUni, 1000, 1)
                store2uni = allocate_memory_EI_FGH(TestAcvModelUni, 10, 1)
                @test_throws ArgumentError WhittleLikelihoodInference.hess_acv!(store1.hessmemory, TestAcvModel(), store2.encodedtime) 
                @test_throws ArgumentError WhittleLikelihoodInference.hess_acv!(store1uni.hessmemory, TestAcvModelUni(), store2uni.encodedtime) 
            end
            @testset "Custom lag when acv unknown error" begin
                @test_throws ArgumentError grad_acv(TestUnknownAcv(), 1.0)
                @test_throws ArgumentError grad_acv(TestUnknownAcvUni(), 1.0)
            end
        end
        @testset "Value" begin
            @testset "Simple model" begin
                store1 = allocate_memory_EI_FGH(TestAcv2, 10, 1)
                store1uni = allocate_memory_EI_FGH(TestAcvUni2, 10, 1)
                WhittleLikelihoodInference.hess_acv!(store1,TestAcv2())
                WhittleLikelihoodInference.hess_acv!(store1uni,TestAcvUni2())
                @test all(store1.hessmemory.allocatedarray.==1.0)
                @test all(store1uni.hessmemory.allocatedarray.==1.0)
            end
            @testset "Additive model" begin
                store2 = allocate_memory_EI_FGH(AcvDouble, 10, 1)
                store2uni = allocate_memory_EI_FGH(AcvDoubleUni, 10, 1)
                WhittleLikelihoodInference.hess_acv!(store2,AcvDouble(ones(8)))
                WhittleLikelihoodInference.hess_acv!(store2uni,AcvDoubleUni(ones(8)))
                @test all(store2.store1.hessmemory.allocatedarray.==1.0)
                @test all(store2.store2.hessmemory.allocatedarray.==2.0)
                @test all(store2uni.store1.hessmemory.allocatedarray.==1.0)
                @test all(store2uni.store2.hessmemory.allocatedarray.==2.0)
            end
        end
    end
end
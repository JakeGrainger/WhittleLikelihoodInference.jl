struct TestModel <: WhittleLikelihoodInference.TimeSeriesModel{2,Float64} end
struct TestModelUni <: WhittleLikelihoodInference.TimeSeriesModel{1,Float64} end
struct TestModel2 <: WhittleLikelihoodInference.TimeSeriesModel{2,Float64} end
struct TestModel2C <: WhittleLikelihoodInference.TimeSeriesModel{2,ComplexF64} end
struct TestModelUni2 <: WhittleLikelihoodInference.TimeSeriesModel{1,Float64} end
struct TestModelUni2C <: WhittleLikelihoodInference.TimeSeriesModel{1,ComplexF64} end
struct TestModel3 <: WhittleLikelihoodInference.TimeSeriesModel{2,Float64} end
struct TestModelUni3 <: WhittleLikelihoodInference.TimeSeriesModel{1,Float64} end
TestModel2(x) = TestModel2() # just to allow additive constructor to work
TestModelUni2(x) = TestModelUni2()
TestModel3(x) = TestModel3() # just to allow additive constructor to work
TestModelUni3(x) = TestModelUni3()
WhittleLikelihoodInference.npars(::Union{Type{TestModel2},Type{TestModel2C}}) = 4
WhittleLikelihoodInference.npars(::Union{Type{TestModelUni2},Type{TestModelUni2C}}) = 4
WhittleLikelihoodInference.npars(::Type{TestModel3}) = 4
WhittleLikelihoodInference.npars(::Type{TestModelUni3}) = 4
const TestDouble = TestModel2 + TestModel3
const TestDoubleUni = TestModelUni2 + TestModelUni3
WhittleLikelihoodInference.add_sdf!(out,::Union{TestModel2,TestModel2C},ω) = (out .+= 0.5; nothing)
WhittleLikelihoodInference.grad_add_sdf!(out,::Union{TestModel2,TestModel2C},ω) = (out .+= 0.5; nothing)
WhittleLikelihoodInference.hess_add_sdf!(out,::Union{TestModel2,TestModel2C},ω) = (out .+= 0.5; nothing)
WhittleLikelihoodInference.sdf(::Union{TestModelUni2,TestModelUni2C},ω) = 0.5
WhittleLikelihoodInference.grad_add_sdf!(out,::Union{TestModelUni2,TestModelUni2C},ω) = (out .+= 0.5; nothing)
WhittleLikelihoodInference.hess_add_sdf!(out,::Union{TestModelUni2,TestModelUni2C},ω) = (out .+= 0.5; nothing)
WhittleLikelihoodInference.add_sdf!(out,::TestModel3,ω) = (out .+= 0.2; nothing)
WhittleLikelihoodInference.grad_add_sdf!(out,::TestModel3,ω) = (out .+= 0.2; nothing)
WhittleLikelihoodInference.hess_add_sdf!(out,::TestModel3,ω) = (out .+= 0.2; nothing)
WhittleLikelihoodInference.sdf(::TestModelUni3,ω) = 0.2
WhittleLikelihoodInference.grad_add_sdf!(out,::TestModelUni3,ω) = (out .+= 0.2; nothing)
WhittleLikelihoodInference.hess_add_sdf!(out,::TestModelUni3,ω) = (out .+= 0.2; nothing)
WhittleLikelihoodInference.nalias(::Union{TestModel2,TestModel2C}) = 2
WhittleLikelihoodInference.nalias(::Union{TestModelUni2,TestModelUni2C}) = 2
WhittleLikelihoodInference.nalias(::TestModel3) = 1
WhittleLikelihoodInference.nalias(::TestModelUni3) = 1
@testset "Spectral density" begin
    @testset "Function" begin
        @testset "Error handling" begin
            @test_throws ErrorException WhittleLikelihoodInference.add_sdf!(ones(2),TestModel(),1.0)
            @test_throws ErrorException WhittleLikelihoodInference.sdf(TestModelUni(),1.0)
        end
        @testset "sdf" begin
            @test sdf(TestModel2(),1.0) == SHermitianCompact(@SVector [complex(0.5) for i in 1:3])
            @test sdf(TestModelUni2(),1.0) == 0.5
            @test sdf(TestDouble(zeros(8)),1.0) == SHermitianCompact(@SVector [complex(0.7) for i in 1:3])
            @test sdf(TestDoubleUni(zeros(8)),1.0) == 0.7
        end
        @testset "asdf" begin
            @test asdf(TestModel2(),1.0,1.0) == SHermitianCompact(@SVector [complex(2.5) for i in 1:3]) 
            @test asdf(TestModelUni2(),1.0,1.0) == 2.5 # two lots of aliasing
            @test asdf(TestDouble(zeros(8)),1.0,1.0) ≈ SHermitianCompact(@SVector [complex(3.1) for i in 1:3])
            @test asdf(TestDoubleUni(zeros(8)),1.0,1.0) == 3.1 # two lots of aliasing of 0.5 plus one lot of 0.2 (but has floating point error)
        end
        @testset "Bounds check" begin
            store = allocate_memory_EI_F(TestModel2, 1000, 1)
            storeuni = allocate_memory_EI_F(TestModelUni2, 1000, 1)
            freqbad = FreqAcvEst(10,1)
            @test_throws ArgumentError WhittleLikelihoodInference.asdf!(store.funcmemory,TestModel2(),freqbad)
            @test_throws ArgumentError WhittleLikelihoodInference.asdf!(storeuni.funcmemory,TestModelUni2(),freqbad)
        end
        @testset "Complex vs real" begin
            for n in [100, 101]
                store = allocate_memory_EI_F(TestModel2, n, 1.2)
                store_c = allocate_memory_EI_F(TestModel2C, n, 1.2)
                storeuni = allocate_memory_EI_F(TestModelUni2, n, 1.2)
                storeuni_c = allocate_memory_EI_F(TestModelUni2C, n, 1.2)
                WhittleLikelihoodInference.asdf!(store,TestModel2())
                WhittleLikelihoodInference.asdf!(store_c,TestModel2C())
                WhittleLikelihoodInference.asdf!(storeuni,TestModelUni2())
                WhittleLikelihoodInference.asdf!(storeuni_c,TestModelUni2C())
                @test store.allocatedarray == store_c.funcmemory.allocatedarray
                @test storeuni.allocatedarray == storeuni_c.funcmemory.allocatedarray
            end
        end
        @testset "extract_asdf" begin
            @test_throws MethodError extract_asdf(allocate_memory_EI_F(TestModel2, 1000, 1).funcmemory)
            @test extract_asdf(allocate_memory_sdf_F(TestModel2, 1000, 1).funcmemory) isa Matrix{ComplexF64}
        end
    end
    @testset "Gradient" begin
        @testset "Error handling" begin
            @test_throws ErrorException WhittleLikelihoodInference.add_sdf!(ones(2),TestModel(),1.0)
            @test_throws ErrorException WhittleLikelihoodInference.sdf(TestModelUni(),1.0)
        end
        @testset "sdf" begin
            @test grad_sdf(TestModel2(),1.0) == fill(complex(0.5),3,4)
            @test grad_sdf(TestModelUni2(),1.0) == fill(complex(0.5),4)
            @test grad_sdf(TestDouble(zeros(8)),1.0) == hcat(fill(complex(0.5),3,4),fill(complex(0.2),3,4))
            @test grad_sdf(TestDoubleUni(zeros(8)),1.0) == [fill(complex(0.5),4);fill(complex(0.2),4)]
        end
        @testset "Bounds check" begin
            store = allocate_memory_EI_FG(TestModel2, 1000, 1)
            storeuni = allocate_memory_EI_FG(TestModelUni2, 1000, 1)
            freqbad = FreqAcvEst(10,1)
            @test_throws ArgumentError WhittleLikelihoodInference.grad_asdf!(store.gradmemory,TestModel2(),freqbad)
            @test_throws ArgumentError WhittleLikelihoodInference.grad_asdf!(storeuni.gradmemory,TestModelUni2(),freqbad)
        end
        @testset "Complex vs real" begin
            for n in [100, 101]
                store = allocate_memory_EI_FG(TestModel2, n, 1.2)
                store_c = allocate_memory_EI_FG(TestModel2C, n, 1.2)
                storeuni = allocate_memory_EI_FG(TestModelUni2, n, 1.2)
                storeuni_c = allocate_memory_EI_FG(TestModelUni2C, n, 1.2)
                WhittleLikelihoodInference.grad_asdf!(store,TestModel2())
                WhittleLikelihoodInference.grad_asdf!(store_c,TestModel2C())
                WhittleLikelihoodInference.grad_asdf!(storeuni,TestModelUni2())
                WhittleLikelihoodInference.grad_asdf!(storeuni_c,TestModelUni2C())
                @test store.allocatedarray == store_c.gradmemory.allocatedarray
                @test storeuni.allocatedarray == storeuni_c.gradmemory.allocatedarray
            end
        end
    end
    @testset "Hessian" begin
        @testset "Error handling" begin
            @test_throws ErrorException WhittleLikelihoodInference.add_sdf!(ones(2),TestModel(),1.0)
            @test_throws ErrorException WhittleLikelihoodInference.sdf(TestModelUni(),1.0)
        end
        @testset "sdf" begin
            @test hess_sdf(TestModel2(),1.0) == fill(complex(0.5),3,10)
            @test hess_sdf(TestModelUni2(),1.0) == fill(complex(0.5),10)
            @test hess_sdf(TestDouble(zeros(8)),1.0) == hcat(fill(complex(0.5),3,10),fill(complex(0.2),3,10))
            @test hess_sdf(TestDoubleUni(zeros(8)),1.0) == [fill(complex(0.5),10);fill(complex(0.2),10)]
        end
        @testset "Bounds check" begin
            store = allocate_memory_EI_FGH(TestModel2, 1000, 1)
            storeuni = allocate_memory_EI_FGH(TestModelUni2, 1000, 1)
            freqbad = FreqAcvEst(10,1)
            @test_throws ArgumentError WhittleLikelihoodInference.hess_asdf!(store.hessmemory,TestModel2(),freqbad)
            @test_throws ArgumentError WhittleLikelihoodInference.hess_asdf!(storeuni.hessmemory,TestModelUni2(),freqbad)
        end
        @testset "Complex vs real" begin
            for n in [100, 101]
                store = allocate_memory_EI_FGH(TestModel2, n, 1.2)
                store_c = allocate_memory_EI_FGH(TestModel2C, n, 1.2)
                storeuni = allocate_memory_EI_FGH(TestModelUni2, n, 1.2)
                storeuni_c = allocate_memory_EI_FGH(TestModelUni2C, n, 1.2)
                WhittleLikelihoodInference.hess_asdf!(store,TestModel2())
                WhittleLikelihoodInference.hess_asdf!(store_c,TestModel2C())
                WhittleLikelihoodInference.hess_asdf!(storeuni,TestModelUni2())
                WhittleLikelihoodInference.hess_asdf!(storeuni_c,TestModelUni2C())
                @test store.allocatedarray == store_c.hessmemory.allocatedarray
                @test storeuni.allocatedarray == storeuni_c.hessmemory.allocatedarray
            end
        end
    end
end
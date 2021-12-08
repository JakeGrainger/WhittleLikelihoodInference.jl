const CorrelatedOUDouble = CorrelatedOU+CorrelatedOU
n = 10
Δ = 0.5
@testset "memoryallocation" begin
    @testset "Acv2EIStorage" begin
        x = randn(ComplexF64,3,n); y = copy(x)
        acvstore = Acv2EIStorage(y,reinterpret(SHermitianCompact{2,eltype(y),3}, vec(y)),FFTW.ESTIMATE)
        @test fft(x,2) == acvstore.planned_fft*acvstore.allocatedarray
    end
    @testset "Sdf2AcvStorage" begin
        x = randn(ComplexF64,3,n); y = copy(x)
        sdfstore = Sdf2AcvStorage(y,FFTW.ESTIMATE)
        @test ifft(x,2) == sdfstore.planned_ifft*sdfstore.allocatedarray
    end
    @testset "Acv2EIStorageUni" begin
        x = randn(ComplexF64,n); y = copy(x)
        acvstore = Acv2EIStorageUni(y,FFTW.ESTIMATE)
        @test fft(x,1) == acvstore.planned_fft*acvstore.allocatedarray
    end
    @testset "Sdf2AcvStorageUni" begin
        x = randn(ComplexF64,n); y = copy(x)
        sdfstore = Sdf2AcvStorageUni(y,FFTW.ESTIMATE)
        @test ifft(x,1) == sdfstore.planned_ifft*sdfstore.allocatedarray
    end
    @testset "EIstorage_function" begin
        @test EIstorage_function(CorrelatedOU,n) isa Acv2EIStorage
        @test EIstorage_function(OU,n)           isa Acv2EIStorageUni
        @test EIstorage_function(Matern2D,n)     isa Sdf2EIStorage
        @test EIstorage_function(OUUnknown{1},n)  isa Sdf2EIStorageUni
    end
    @testset "EIstorage_gradient" begin
        @test EIstorage_gradient(CorrelatedOU,n) isa Acv2EIStorage
        @test EIstorage_gradient(OU,n)           isa Acv2EIStorageUni
        @test EIstorage_gradient(Matern2D,n)     isa Sdf2EIStorage
        @test EIstorage_gradient(OUUnknown{1},n) isa Sdf2EIStorageUni
    end
    @testset "EIstorage_hessian" begin
        @test EIstorage_hessian(CorrelatedOU,n) isa Acv2EIStorage
        @test EIstorage_hessian(OU,n)           isa Acv2EIStorageUni
        @test EIstorage_hessian(Matern2D,n)     isa Sdf2EIStorage
        @test EIstorage_hessian(OUUnknown{1},n) isa Sdf2EIStorageUni
    end
    @testset "FreqAcvEst" begin
        freq = FreqAcvEst(3,Δ)
        @test freq.Ωₘ ≈ [0.0,4.1887902047863905,-4.1887902047863905]
        @test freq.n == 3
        @test freq.Δ == Δ
        @test length(FreqAcvEst(Matern2D,n,Δ).Ωₘ) > n
    end
    @testset "LagsEI" begin
        @test LagsEI(3,Δ).lags ≈ [0.0,Δ,2Δ,-3Δ,-2Δ,-Δ]
    end
    @testset "encodetimescale" begin
        @test encodetimescale(OU,n,Δ) isa LagsEI
        @test encodetimescale(Matern2D,n,Δ) isa FreqAcvEst
    end
    @testset "extract_S" begin
        @test WhittleLikelihoodInference.extract_S(allocate_memory_EI_F(CorrelatedOUDouble,n,Δ)) isa Acv2EIStorage
        @test WhittleLikelihoodInference.extract_S(allocate_memory_EI_F(CorrelatedOUUnknown{1},n,Δ)) isa Sdf2EIStorage
    end
    @testset "allocate_memory_EI_F" begin
        @test allocate_memory_EI_F(CorrelatedOU,n,Δ) isa PreallocatedEI
        add_alloc_func = allocate_memory_EI_F(CorrelatedOUDouble,n,Δ)
        @test add_alloc_func isa AdditiveStorage
        @test add_alloc_func.store1 isa PreallocatedEI
        @test add_alloc_func.store2 isa PreallocatedEI
        @test add_alloc_func.npar1 == 3
    end
    @testset "allocate_memory_EI_FG" begin
        @test allocate_memory_EI_FG(CorrelatedOU,n,Δ) isa PreallocatedEIGradient
        add_alloc_grad = allocate_memory_EI_FG(CorrelatedOUDouble,n,Δ)
        @test add_alloc_grad isa AdditiveStorage
        @test add_alloc_grad.store1 isa PreallocatedEIGradient
        @test add_alloc_grad.store2 isa PreallocatedEIGradient
        @test add_alloc_grad.npar1 == 3
    end
    @testset "allocate_memory_EI_FGH" begin
        @test allocate_memory_EI_FGH(CorrelatedOU,n,Δ) isa PreallocatedEIHessian
        add_alloc_hess = allocate_memory_EI_FGH(CorrelatedOUDouble,n,Δ)
        @test add_alloc_hess isa AdditiveStorage
        @test add_alloc_hess.store1 isa PreallocatedEIHessian
        @test add_alloc_hess.store2 isa PreallocatedEIHessian
        @test add_alloc_hess.npar1 == 3
    end
    @testset "sdfstorage_function" begin
        @test sdfstorage_function(OU,n) isa SdfStorageUni
        @test sdfstorage_function(CorrelatedOU,n) isa SdfStorage
    end
    @testset "sdfstorage_gradient" begin
        @test sdfstorage_gradient(OU,n) isa SdfStorageUni
        @test sdfstorage_gradient(CorrelatedOU,n) isa SdfStorage
    end
    @testset "sdfstorage_hessian" begin
        @test sdfstorage_hessian(OU,n) isa SdfStorageUni
        @test sdfstorage_hessian(CorrelatedOU,n) isa SdfStorage
    end
    @testset "allocate_memory_sdf_F" begin
        @test allocate_memory_sdf_F(CorrelatedOU,n,Δ) isa PreallocatedSdf
        add_alloc_func = allocate_memory_sdf_F(CorrelatedOUDouble,n,Δ)
        @test add_alloc_func isa AdditiveStorage
        @test add_alloc_func.store1 isa PreallocatedSdf
        @test add_alloc_func.store2 isa PreallocatedSdf
        @test add_alloc_func.npar1 == 3
    end
    @testset "allocate_memory_sdf_FG" begin
        @test allocate_memory_sdf_FG(CorrelatedOU,n,Δ) isa PreallocatedSdfGradient
        add_alloc_grad = allocate_memory_sdf_FG(CorrelatedOUDouble,n,Δ)
        @test add_alloc_grad isa AdditiveStorage
        @test add_alloc_grad.store1 isa PreallocatedSdfGradient
        @test add_alloc_grad.store2 isa PreallocatedSdfGradient
        @test add_alloc_grad.npar1 == 3
    end
    @testset "allocate_memory_sdf_FGH" begin
        @test allocate_memory_sdf_FGH(CorrelatedOU,n,Δ) isa PreallocatedSdfHessian
        add_alloc_hess = allocate_memory_sdf_FGH(CorrelatedOUDouble,n,Δ)
        @test add_alloc_hess isa AdditiveStorage
        @test add_alloc_hess.store1 isa PreallocatedSdfHessian
        @test add_alloc_hess.store2 isa PreallocatedSdfHessian
        @test add_alloc_hess.npar1 == 3
    end
    @testset "unpack" begin
        x = randn(ComplexF64,3,n)
        y1,y2,y3,y4,y5,y6 = fill(copy(x),6)
        h1 = reinterpret(SHermitianCompact{2,eltype(y1),3}, vec(y1))
        h2 = reinterpret(SHermitianCompact{2,eltype(y2),3}, vec(y2))
        h3 = reinterpret(SHermitianCompact{2,eltype(y3),3}, vec(y3))
        acvstore    = Acv2EIStorage(y1,h1,FFTW.ESTIMATE)
        sdfstore    = Sdf2EIStorage(y2,y2,h2,FFTW.ESTIMATE)
        sdfonly     = SdfStorage(y3,h3)
        acvstoreuni = Acv2EIStorageUni(y4,FFTW.ESTIMATE)
        sdfstoreuni = Sdf2EIStorageUni(y5,y5,FFTW.ESTIMATE)
        sdfonlyuni  = SdfStorageUni(y6)
        @test WhittleLikelihoodInference.unpack(acvstore)    === h1
        @test WhittleLikelihoodInference.unpack(sdfstore)    === h2
        @test WhittleLikelihoodInference.unpack(sdfonly )    === h3
        @test WhittleLikelihoodInference.unpack(acvstoreuni) === y4
        @test WhittleLikelihoodInference.unpack(sdfstoreuni) === y5
        @test WhittleLikelihoodInference.unpack(sdfonlyuni)  === y6
    end
    @testset "taper2timedomainkernel" begin
        @test WhittleLikelihoodInference.taper2timedomainkernel(ones(4)./2,4) == WhittleLikelihoodInference.taper2timedomainkernel(nothing,4)
    end
end
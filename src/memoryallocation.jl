## Multivariate ##

"""
`TimeSeriesModelStorage` preallocated storage for model manipulation including preplanned ffts.
"""
abstract type TimeSeriesModelStorage end

struct Acv2EIStorage{N,M,D,L,P} <: TimeSeriesModelStorage
    allocatedarray::Array{ComplexF64,M}
    hermitianarray::Base.ReinterpretArray{
        SHermitianCompact{D,ComplexF64,L},
        N,
        ComplexF64,
        Array{ComplexF64,N},
    }
    planned_fft::P
    function Acv2EIStorage(
        allocatedarray::Array{ComplexF64,M},
        hermitianarray::Base.ReinterpretArray{
            SHermitianCompact{D,ComplexF64,L},
            N,
            ComplexF64,
            Array{ComplexF64,N},
        },
        fft_flags,
    ) where {N,M,D,L}
        planned_fft = plan_fft!(allocatedarray, M, flags = fft_flags)
        new{N,M,D,L,typeof(planned_fft)}(allocatedarray, hermitianarray, planned_fft)
    end
end

struct Sdf2AcvStorage{M,Q} <: TimeSeriesModelStorage
    allocatedarray::Array{ComplexF64,M}
    planned_ifft::Q
    function Sdf2AcvStorage(allocatedarray::Array{ComplexF64,M}, fft_flags) where {M}
        planned_ifft = plan_ifft!(allocatedarray, M, flags = fft_flags)
        new{M,typeof(planned_ifft)}(allocatedarray, planned_ifft)
    end
end

"Return the type of the fftplan from Acv2EIStorage and Sdf2AcvStorage types."
fft_plan_type(::Acv2EIStorage{N,M,D,L,P}) where {N,M,D,L,P} = P
fft_plan_type(::Sdf2AcvStorage{M,Q}) where {M,Q} = Q

"Struct to store required storage for computing EI when acv is not specified."
struct Sdf2EIStorage{N,M,D,L,Q,P} <: TimeSeriesModelStorage
    sdf2acv::Sdf2AcvStorage{M,Q}
    acv2EI::Acv2EIStorage{N,M,D,L,P}
    function Sdf2EIStorage(
        sdf_array::Array{ComplexF64,M},
        acv_array::Array{ComplexF64,M},
        hermitianarray::Base.ReinterpretArray{
            SHermitianCompact{D,ComplexF64,L},
            N,
            ComplexF64,
            Array{ComplexF64,N},
        },
        fft_flags,
    ) where {N,M,D,L}
        sdf2acv = Sdf2AcvStorage(sdf_array, fft_flags)
        acv2EI = Acv2EIStorage(acv_array, hermitianarray, fft_flags)
        new{N,M,D,L,fft_plan_type(sdf2acv),fft_plan_type(acv2EI)}(sdf2acv, acv2EI)
    end
end

"Generate required storage for computing the EI for a general time series model."
function EIstorage_function(model::Type{<:TimeSeriesModel{D}}, n, fft_flags = FFTW.MEASURE) where {D}
    acv_array = zeros(ComplexF64, nlowertriangle_dimension(model), 2n)
    hermitianarray = reinterpret(SHermitianCompact{D,eltype(acv_array),nlowertriangle_dimension(model)}, vec(acv_array))
    return Acv2EIStorage(acv_array, hermitianarray, fft_flags)
end
function EIstorage_function(model::Type{<:UnknownAcvTimeSeriesModel{D}}, n, fft_flags = FFTW.MEASURE) where {D}
    sdf_array = zeros(ComplexF64, nlowertriangle_dimension(model), max(2n, minbins(model)))
    acv_array = zeros(ComplexF64, nlowertriangle_dimension(model), 2n)
    hermitianarray = reinterpret(SHermitianCompact{D,eltype(acv_array),nlowertriangle_dimension(model)}, vec(acv_array))
    return Sdf2EIStorage(sdf_array, acv_array, hermitianarray, fft_flags)
end

"Generate required storage for computing the gradient of EI for a general time series model."
function EIstorage_gradient(model::Type{<:TimeSeriesModel{D}}, n, fft_flags = FFTW.MEASURE) where {D}
    acv_array = zeros(ComplexF64, nlowertriangle_dimension(model), npars(model), 2n)
    hermitianarray = reinterpret(SHermitianCompact{D,eltype(acv_array),nlowertriangle_dimension(model)}, reshape(acv_array, (:, 2n)))
    return Acv2EIStorage(acv_array, hermitianarray, fft_flags)
end
function EIstorage_gradient(model::Type{<:UnknownAcvTimeSeriesModel{D}}, n, fft_flags = FFTW.MEASURE) where {D}
    sdf_array = zeros(ComplexF64, nlowertriangle_dimension(model), npars(model), max(2n, minbins(model)))
    acv_array = zeros(ComplexF64, nlowertriangle_dimension(model), npars(model), 2n)
    hermitianarray = reinterpret(SHermitianCompact{D,eltype(acv_array),nlowertriangle_dimension(model)}, reshape(acv_array, (:, 2n)))
    return Sdf2EIStorage(sdf_array, acv_array, hermitianarray, fft_flags)
end

"Generate required storage for computing the hessian of EI for a general time series model."
function EIstorage_hessian(model::Type{<:TimeSeriesModel{D}}, n, fft_flags = FFTW.MEASURE) where {D}
    acv_array = zeros(ComplexF64, nlowertriangle_dimension(model), nlowertriangle_parameter(model), 2n)
    hermitianarray = reinterpret(SHermitianCompact{D,eltype(acv_array),nlowertriangle_dimension(model)}, reshape(acv_array, (:, 2n)))
    return Acv2EIStorage(acv_array, hermitianarray, fft_flags)
end
function EIstorage_hessian(model::Type{<:UnknownAcvTimeSeriesModel{D}}, n, fft_flags = FFTW.MEASURE) where {D}
    sdf_array = zeros(ComplexF64, nlowertriangle_dimension(model), nlowertriangle_parameter(model), max(2n, minbins(model)))
    acv_array = zeros(ComplexF64, nlowertriangle_dimension(model), nlowertriangle_parameter(model), 2n)
    hermitianarray = reinterpret(SHermitianCompact{D,eltype(acv_array),nlowertriangle_dimension(model)}, reshape(acv_array, (:, 2n)))
    return Sdf2EIStorage(sdf_array, acv_array, hermitianarray, fft_flags)
end

"Storage for useful quanties associated with the EI memory when estimating the acv, namely n, Δ and Ωₘ."
struct FreqAcvEst
    n::Int
    Δ::Float64
    Ωₘ::Frequencies{Float64}
    function FreqAcvEst(model::Type{<:UnknownAcvTimeSeriesModel},n,Δ)
        Ωₘ = fftfreq(max(2n, minbins(model)), 2π/Δ)
        new(n,Δ,Ωₘ)
    end
    function FreqAcvEst(n,Δ)
        Ωₘ = fftfreq(n, 2π/Δ)
        new(n,Δ,Ωₘ)
    end
end

"Storage for useful quanties associated with the EI memory when the acv is known, namely n, Δ and Ωₘ."
struct LagsEI
    n::Int
    Δ::Float64
    lags::Frequencies{Float64}
    function LagsEI(n,Δ)
        lags = fftfreq(2n, 2n)
        new(n,Δ,lags)
    end
end

"Function to choose additional storage depending on whether the acv is known or not."
encodetimescale(model::Type{<:TimeSeriesModel},n,Δ) = LagsEI(n,Δ)
encodetimescale(model::Type{<:UnknownAcvTimeSeriesModel},n,Δ) = FreqAcvEst(model,n,Δ)

abstract type TimeSeriesModelStorageFunction <: TimeSeriesModelStorage end
abstract type TimeSeriesModelStorageGradient <: TimeSeriesModelStorageFunction end
abstract type TimeSeriesModelStorageHessian  <: TimeSeriesModelStorageGradient end

"Structure to store memory if only the EI is required."
struct PreallocatedEI{T,F} <: TimeSeriesModelStorageFunction
    encodedtime::T
    funcmemory::F
    function PreallocatedEI(model::Type{<:TimeSeriesModel}, n, Δ; fft_flags = FFTW.MEASURE)
        time = encodetimescale(model, n, Δ)
        func = EIstorage_function(model, n, fft_flags)
        new{typeof(time),typeof(func)}(time,func)
    end
end

"Structure to store memory if the EI and its gradient is required."
struct PreallocatedEIGradient{T,F,G} <: TimeSeriesModelStorageGradient
    encodedtime::T
    funcmemory::F
    gradmemory::G
    function PreallocatedEIGradient(model::Type{<:TimeSeriesModel}, n, Δ; fft_flags = FFTW.MEASURE)
        time = encodetimescale(model, n, Δ)
        func = EIstorage_function(model, n, fft_flags)
        grad = EIstorage_gradient(model, n, fft_flags)
        new{typeof(time),typeof(func),typeof(grad)}(time, func, grad)
    end
end

"Structure to store memory if the EI, its gradient and its hessian are required."
struct PreallocatedEIHessian{T,F,G,H} <: TimeSeriesModelStorageHessian
    encodedtime::T
    funcmemory::F
    gradmemory::G
    hessmemory::H
    function PreallocatedEIHessian(model::Type{<:TimeSeriesModel}, n, Δ; fft_flags = FFTW.MEASURE)
        time = encodetimescale(model, n, Δ)
        func = EIstorage_function(model, n, fft_flags)
        grad = EIstorage_gradient(model, n, fft_flags)
        hess = EIstorage_hessian(model, n, fft_flags)
        new{typeof(time),typeof(func),typeof(grad),typeof(hess)}(time, func, grad, hess)
    end
end

"Storage for additive models."
struct AdditiveStorage{S₁<:TimeSeriesModelStorage,S₂<:TimeSeriesModelStorage} <: TimeSeriesModelStorage
    store1::S₁
    store2::S₂
    npar1::Int
end

"Helper function for extracting the function value from additive storage (which is only stored in the left most storage)."
extract_S(store::TimeSeriesModelStorageFunction) = store.funcmemory
extract_S(store::AdditiveStorage) = extract_S(store.store1) # returns the left hand store if additive storage

"Interface level function for allocating the memory for EI computation."
function allocate_memory_EI_F(model::Type{<:TimeSeriesModel}, n, Δ; fft_flags = FFTW.MEASURE)
    return PreallocatedEI(model, n, Δ; fft_flags = fft_flags)
end
function allocate_memory_EI_F(model::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, n, Δ; fft_flags = FFTW.MEASURE) where {M₁,M₂,D}
    return AdditiveStorage( allocate_memory_EI_F(M₁, n, Δ; fft_flags = fft_flags),
                            allocate_memory_EI_F(M₂, n, Δ; fft_flags = fft_flags),
                            npars(M₁))
end

"Interface level function for allocating the memory for EI computation with gradient."
function allocate_memory_EI_FG(model::Type{<:TimeSeriesModel}, n, Δ; fft_flags = FFTW.MEASURE)
    return PreallocatedEIGradient(model, n, Δ; fft_flags = fft_flags)
end
function allocate_memory_EI_FG(model::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, n, Δ; fft_flags = FFTW.MEASURE) where {M₁,M₂,D}
    return AdditiveStorage( allocate_memory_EI_FG(M₁, n, Δ; fft_flags = fft_flags),
                            allocate_memory_EI_FG(M₂, n, Δ; fft_flags = fft_flags),
                            npars(M₁))
end

"Interface level function for allocating the memory for EI computation with gradient and hessian."
function allocate_memory_EI_FGH(model::Type{<:TimeSeriesModel}, n, Δ; fft_flags = FFTW.MEASURE)
    return PreallocatedEIHessian(model, n, Δ; fft_flags = fft_flags)
end
function allocate_memory_EI_FGH(model::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, n, Δ; fft_flags = FFTW.MEASURE) where {M₁,M₂,D}
    return AdditiveStorage( allocate_memory_EI_FGH(M₁, n, Δ; fft_flags = fft_flags),
                            allocate_memory_EI_FGH(M₂, n, Δ; fft_flags = fft_flags),
                            npars(M₁))
end

## sdf only storage

struct SdfStorage{N,M,D,L} <: TimeSeriesModelStorage
    allocatedarray::Array{ComplexF64,M}
    hermitianarray::Base.ReinterpretArray{
        SHermitianCompact{D,ComplexF64,L},
        N,
        ComplexF64,
        Array{ComplexF64,N},
    }
    function SdfStorage(allocatedarray::Array{ComplexF64,M}, hermitianarray::Base.ReinterpretArray{
            SHermitianCompact{D,ComplexF64,L},
            N,
            ComplexF64,
            Array{ComplexF64,N},
        }) where {N,M,D,L}
        new{N,M,D,L}(allocatedarray,hermitianarray)
    end
end
"Function to allocate storage for the function part of sdf calculations."
function sdfstorage_function(model::Type{<:TimeSeriesModel{D}}, n) where {D}
    allocatedarray = zeros(ComplexF64, nlowertriangle_dimension(model), n)
    hermitianarray = reinterpret(SHermitianCompact{D,eltype(allocatedarray),nlowertriangle_dimension(model)}, vec(allocatedarray))
    return SdfStorage(allocatedarray,hermitianarray)
end
"Function to allocate storage for the gradient part of sdf calculations."
function sdfstorage_gradient(model::Type{<:TimeSeriesModel{D}}, n) where {D}
    allocatedarray = zeros(ComplexF64, nlowertriangle_dimension(model), npars(model), n)
    hermitianarray = reinterpret(SHermitianCompact{D,eltype(allocatedarray),nlowertriangle_dimension(model)}, reshape(allocatedarray, (:, n)))
    return SdfStorage(allocatedarray,hermitianarray)
end
"Function to allocate storage for the hessian part of sdf calculations."
function sdfstorage_hessian(model::Type{<:TimeSeriesModel{D}}, n) where {D}
    allocatedarray = zeros(ComplexF64, nlowertriangle_dimension(model), nlowertriangle_parameter(model), n)
    hermitianarray = reinterpret(SHermitianCompact{D,eltype(allocatedarray),nlowertriangle_dimension(model)}, reshape(allocatedarray, (:, n)))
    return SdfStorage(allocatedarray,hermitianarray)
end

"Structure to store memory if the sdf and its gradient is required."
struct PreallocatedSdf{F} <: TimeSeriesModelStorageFunction
    encodedtime::FreqAcvEst
    funcmemory::F
    function PreallocatedSdf(model::Type{<:TimeSeriesModel}, n, Δ)
        time = FreqAcvEst(n, Δ)
        func = sdfstorage_function(model, n)
        new{typeof(func)}(time, func)
    end
end

"Structure to store memory if the sdf and its gradient is required."
struct PreallocatedSdfGradient{F,G} <: TimeSeriesModelStorageGradient
    encodedtime::FreqAcvEst
    funcmemory::F
    gradmemory::G
    function PreallocatedSdfGradient(model::Type{<:TimeSeriesModel}, n, Δ)
        time = FreqAcvEst(n, Δ)
        func = sdfstorage_function(model, n)
        grad = sdfstorage_gradient(model, n)
        new{typeof(func),typeof(grad)}(time, func, grad)
    end
end

"Structure to store memory if the sdf, its gradient and its hessian are required."
struct PreallocatedSdfHessian{F,G,H} <: TimeSeriesModelStorageHessian
    encodedtime::FreqAcvEst
    funcmemory::F
    gradmemory::G
    hessmemory::H
    function PreallocatedSdfHessian(model::Type{<:TimeSeriesModel}, n, Δ)
        time = FreqAcvEst(n, Δ)
        func = sdfstorage_function(model, n)
        grad = sdfstorage_gradient(model, n)
        hess = sdfstorage_hessian(model, n)
        new{typeof(func),typeof(grad),typeof(hess)}(time, func, grad, hess)
    end
end

"Interface level function for allocating the memory for EI computation."
function allocate_memory_sdf_F(model::Type{<:TimeSeriesModel}, n, Δ)
    return PreallocatedSdf(model, n, Δ)
end
function allocate_memory_sdf_F(model::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, n, Δ) where {M₁,M₂,D}
    return AdditiveStorage( allocate_memory_sdf_F(M₁, n, Δ),
                            allocate_memory_sdf_F(M₂, n, Δ),
                            npars(M₁))
end

"Interface level function for allocating the memory for sdf computation with gradient."
function allocate_memory_sdf_FG(model::Type{<:TimeSeriesModel}, n, Δ)
    return PreallocatedSdfGradient(model, n, Δ)
end
function allocate_memory_sdf_FG(model::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, n, Δ) where {M₁,M₂,D}
    return AdditiveStorage( allocate_memory_sdf_FG(M₁, n, Δ),
                            allocate_memory_sdf_FG(M₂, n, Δ),
                            npars(M₁))
end

"Interface level function for allocating the memory for sdf computation with gradient and hessian."
function allocate_memory_sdf_FGH(model::Type{<:TimeSeriesModel}, n, Δ)
    return PreallocatedSdfHessian(model, n, Δ)
end
function allocate_memory_sdf_FGH(model::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, n, Δ) where {M₁,M₂,D}
    return AdditiveStorage( allocate_memory_sdf_FGH(M₁, n, Δ),
                            allocate_memory_sdf_FGH(M₂, n, Δ),
                            npars(M₁))
end


## Univariate ##

struct Acv2EIStorageUni{M,P} <: TimeSeriesModelStorage
    allocatedarray::Array{ComplexF64,M}
    planned_fft::P
    function Acv2EIStorageUni(
        allocatedarray::Array{ComplexF64,M},
        fft_flags,
    ) where {M}
        planned_fft = plan_fft!(allocatedarray, M, flags = fft_flags)
        new{M,typeof(planned_fft)}(allocatedarray, planned_fft)
    end
end

struct Sdf2AcvStorageUni{M,Q} <: TimeSeriesModelStorage
    allocatedarray::Array{ComplexF64,M}
    planned_ifft::Q
    function Sdf2AcvStorageUni(allocatedarray::Array{ComplexF64,M}, fft_flags) where {M}
        planned_ifft = plan_ifft!(allocatedarray, M, flags = fft_flags)
        new{M,typeof(planned_ifft)}(allocatedarray, planned_ifft)
    end
end

"Return the type of the fftplan from Acv2EIStorageUni and Sdf2AcvStorageUni types."
fft_plan_type(::Acv2EIStorageUni{M,P}) where {M,P} = P
fft_plan_type(::Sdf2AcvStorageUni{M,Q}) where {M,Q} = Q

"Struct to store required storage for computing EI when acv is not specified."
struct Sdf2EIStorageUni{M,Q,P} <: TimeSeriesModelStorage
    sdf2acv::Sdf2AcvStorageUni{M,Q}
    acv2EI::Acv2EIStorageUni{M,P}
    function Sdf2EIStorageUni(
        sdf_array::Array{ComplexF64,M},
        acv_array::Array{ComplexF64,M},
        fft_flags,
    ) where {M}
        sdf2acv = Sdf2AcvStorageUni(sdf_array, fft_flags)
        acv2EI = Acv2EIStorageUni(acv_array, fft_flags)
        new{M,fft_plan_type(sdf2acv),fft_plan_type(acv2EI)}(sdf2acv, acv2EI)
    end
end

"Generate required storage for computing the EI for a univariate time series model."
function EIstorage_function(model::Type{<:TimeSeriesModel{1}}, n, fft_flags = FFTW.MEASURE)
    acv_array = zeros(ComplexF64, 2n)
    return Acv2EIStorageUni(acv_array, fft_flags)
end
function EIstorage_function(model::Type{<:UnknownAcvTimeSeriesModel{1}}, n, fft_flags = FFTW.MEASURE)
    sdf_array = zeros(ComplexF64, max(2n, minbins(model)))
    acv_array = zeros(ComplexF64, 2n)
    return Sdf2EIStorageUni(sdf_array, acv_array, fft_flags)
end

"Generate required storage for computing the gradient of EI for a univariate time series model."
function EIstorage_gradient(model::Type{<:TimeSeriesModel{1}}, n, fft_flags = FFTW.MEASURE)
    acv_array = zeros(ComplexF64, npars(model), 2n)
    return Acv2EIStorageUni(acv_array, fft_flags)
end
function EIstorage_gradient(model::Type{<:UnknownAcvTimeSeriesModel{1}}, n, fft_flags = FFTW.MEASURE)
    sdf_array = zeros(ComplexF64, npars(model), max(2n, minbins(model)))
    acv_array = zeros(ComplexF64, npars(model), 2n)
    return Sdf2EIStorageUni(sdf_array, acv_array, fft_flags)
end

"Generate required storage for computing the hessian of EI for a univariate time series model."
function EIstorage_hessian(model::Type{<:TimeSeriesModel{1}}, n, fft_flags = FFTW.MEASURE)
    acv_array = zeros(ComplexF64, nlowertriangle_parameter(model), 2n)
    return Acv2EIStorageUni(acv_array, fft_flags)
end
function EIstorage_hessian(model::Type{<:UnknownAcvTimeSeriesModel{1}}, n, fft_flags = FFTW.MEASURE)
    sdf_array = zeros(ComplexF64, nlowertriangle_parameter(model), max(2n, minbins(model)))
    acv_array = zeros(ComplexF64, nlowertriangle_parameter(model), 2n)
    return Sdf2EIStorageUni(sdf_array, acv_array, fft_flags)
end

## sdf only store
struct SdfStorageUni{M} <: TimeSeriesModelStorage
    allocatedarray::Array{ComplexF64,M}
    function SdfStorageUni(allocatedarray::Array{ComplexF64,M}) where {M}
        new{M}(allocatedarray)
    end
end

"Function to allocate storage for the function part of sdf calculations for univariate case."
function sdfstorage_function(model::Type{<:TimeSeriesModel{1}}, n)
    allocatedarray = zeros(ComplexF64, n)
    return SdfStorageUni(allocatedarray)
end
"Function to allocate storage for the gradient part of sdf calculations for univariate case."
function sdfstorage_gradient(model::Type{<:TimeSeriesModel{1}}, n)
    allocatedarray = zeros(ComplexF64, npars(model), n)
    return SdfStorageUni(allocatedarray)
end
"Function to allocate storage for the hessian part of sdf calculations for univariate case."
function sdfstorage_hessian(model::Type{<:TimeSeriesModel{1}}, n)
    allocatedarray = zeros(ComplexF64, nlowertriangle_parameter(model), n)
    return SdfStorageUni(allocatedarray)
end

"Function to extract array from storage for univariate."
extract_array(store::Sdf2EIStorageUni) = extract_array(store.acv2EI)
extract_array(store::Acv2EIStorageUni) = store.allocatedarray

## Unitily functions for pulling out desired vector from storage for Whittle
unpack(store::Sdf2EIStorage)    = store.acv2EI.hermitianarray
unpack(store::Acv2EIStorage)    = store.hermitianarray
unpack(store::SdfStorage)       = store.hermitianarray
unpack(store::Sdf2EIStorageUni) = store.acv2EI.allocatedarray
unpack(store::Acv2EIStorageUni) = store.allocatedarray
unpack(store::SdfStorageUni)    = store.allocatedarray

## Whittle data ##

"General structure for storing timeseries for Whittle type computation."
abstract type GenWhittleData end

"Structure to store timeseries data in a format which is useful for Whittle methods."
struct WhittleData{T} <: GenWhittleData
    I::Vector{T}
    N::Int64
    Δ::Float64
    Ω_used_index::Vector{Int64}
    function WhittleData(::Type{<:TimeSeriesModel{D}}, timeseries::Matrix{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf) where {D,S<:Real}
        size(timeseries, 2) == D || error("Time series has $(size(timeseries, 2)) dimensions, but model has $D dimension.")
        N = size(timeseries, 1)
        Ω = fftfreq(N, 2π/Δ)
        Ω_used_index = (1:N)[lowerΩcutoff .< abs.(Ω) .< upperΩcutoff]
        J = fft(timeseries, 1)
        I = [SMatrix{D,D}(J[ii, :]*J[ii, :]'.*(Δ / (2π * N))) for ii = Ω_used_index]
        new{eltype(I)}(I, N, Δ, Ω_used_index)
    end
    function WhittleData(::Type{<:TimeSeriesModel{1}}, timeseries::Vector{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf) where {S<:Real}
        N = size(timeseries, 1)
        Ω = fftfreq(N, 2π/Δ)
        Ω_used_index = (1:N)[lowerΩcutoff .< abs.(Ω) .< upperΩcutoff]
        J = fft(timeseries, 1)
        I = abs.(J).^2 .* (Δ / (2π * N))
        new{eltype(I)}(I[Ω_used_index], N, Δ, Ω_used_index)
    end
end

"Structure to store timeseries data in a format which is useful for debiased Whittle methods."
struct DebiasedWhittleData{T} <: GenWhittleData
    I::Vector{T}
    N::Int64
    Δ::Float64
    Ω_used_index::Vector{Int64}
    function DebiasedWhittleData(::Type{<:TimeSeriesModel{D}}, timeseries::Matrix{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf) where {D,S<:Real}
        size(timeseries, 2) == D || error("Time series has $(size(timeseries, 2)) dimensions, but model has $D dimension.")
        N = size(timeseries, 1)
        Ω = fftfreq(N, 2π/Δ)
        Ω_used_index = (1:N)[lowerΩcutoff .< abs.(Ω) .< upperΩcutoff]
        J = fft(timeseries, 1)
        I = [SMatrix{D,D}(J[ii, :]*J[ii, :]'.*(Δ / (2π * N))) for ii = Ω_used_index]
        new{eltype(I)}(I, N, Δ, 2 .* Ω_used_index .- 1) # accounts for double resolution in EI computations
    end
    function DebiasedWhittleData(::Type{<:TimeSeriesModel{1}}, timeseries::Vector{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf) where {S<:Real}
        N = size(timeseries, 1)
        Ω = fftfreq(N, 2π/Δ)
        Ω_used_index = (1:N)[lowerΩcutoff .< abs.(Ω) .< upperΩcutoff]
        J = fft(timeseries, 1)
        I = abs.(J).^2 .* (Δ / (2π * N))
        new{eltype(I)}(I[Ω_used_index], N, Δ, 2 .* Ω_used_index .- 1) # accounts for double resolution in EI computations
    end
end
Base.show(io::IO, W::GenWhittleData) = print(io, "Precomputed periodogram for $(size(W.I[1],1)) dimensional timeseries of length $(W.N) with a sampling rate of $(W.Δ).")

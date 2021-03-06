## Autocovariance ##

### Multivariate ###

"""
    acv!(store::Sdf2EIStorage, model::UnknownAcvTimeSeriesModel, encodedtime::FreqAcvEst)

Approximate the acv and allocate to storage when acv is unknown.
"""
function acv!(store::Sdf2EIStorage, model::UnknownAcvTimeSeriesModel, encodedtime::FreqAcvEst)
    asdf!(store.sdf2acv, model, encodedtime)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:, 1:encodedtime.n] .= store.sdf2acv.allocatedarray[:, 1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:, encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:, end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

"""
    acv!(out, model::TimeSeriesModel, τ)

Compute the acv for a single lag and when acv is known.
"""
function acv!(out, model::TimeSeriesModel, τ) # default acv returns error
    error("acv! not yet defined for model of type $(typeof(model)).")
end

"""
    acv!(store::Acv2EIStorage, model::TimeSeriesModel, encodedtime::LagsEI)

Compute the acv at many lags and allocate to storage when acv is known.
"""
function acv!(store::Acv2EIStorage, model::TimeSeriesModel, encodedtime::LagsEI)
    size(store.allocatedarray, 2) == length(encodedtime.lags) || throw(ArgumentError("size(store.allocatedarray,2) !== length(encodedtime.lags)"))
    @inbounds for (i,τ) ∈ enumerate(encodedtime.lags)
        @views acv!(store.allocatedarray[:, i], model, τ)
    end
    return nothing
end

"""
    acv!(store::TimeSeriesModelStorageFunction, model::TimeSeriesModel)

Unwrap storage and pass to lower level `acv!` call.
"""
function acv!(store::TimeSeriesModelStorageFunction, model::TimeSeriesModel)
    acv!(store.funcmemory, model, store.encodedtime)
    return nothing
end

"""
    extract_acv(store::Sdf2EIStorage)
    extract_acv(store::Acv2EIStorage)
    extract_acv(store::Sdf2EIStorageUni)
    extract_acv(store::Acv2EIStorageUni)

Extract the acv from general storage.
"""
extract_acv(store) = extract_acv(extract_S(store))
extract_acv(store::Sdf2EIStorage) = extract_acv(store.acv2EI)
extract_acv(store::Acv2EIStorage) = store.allocatedarray

"""
    acv(model::TimeSeriesModel, n, Δ)

Compute acv at lags `-(n-1)*Δ:Δ:(n-1)*Δ`.
"""
function acv(model::TimeSeriesModel, n, Δ)
    store = allocate_memory_EI_F(typeof(model), n , Δ)
    acv!(store, model) # lags -n to n-1 in extract_acv(store)
    A = real.(fftshift(extract_acv(store),2)[:, 2:end]) # remove imaginary floating point error
    return [[i >= j ? A[indexLT(i,j,ndims(model)), iτ] : A[indexLT(i,j,ndims(model)), size(A, 2)-iτ+1] for i = 1:ndims(model), j = 1:ndims(model)] for iτ = 1:size(A, 2)]
end

"""
    acv(model::TimeSeriesModel, τ)

Compute acv at τ, provided acv is known.
"""
function acv(model::TimeSeriesModel, τ)
    !(model isa UnknownAcvTimeSeriesModel) || throw(ArgumentError("Custom lag only possible if model has known acv."))
    out = zeros(ComplexF64, nlowertriangle_dimension(model))
    acv!(out, model, τ)
    return SHermitianCompact(SVector{nlowertriangle_dimension(model)}(out))
end

### Univariate ###

"""
    acv!(store::Sdf2EIStorageUni, model::UnknownAcvTimeSeriesModel{1,T}, encodedtime::FreqAcvEst)

Approximate the acv and allocate to storage when acv is known in the univariate case.
"""
function acv!(store::Sdf2EIStorageUni, model::UnknownAcvTimeSeriesModel{1,T}, encodedtime::FreqAcvEst) where {T}
    asdf!(store.sdf2acv, model, encodedtime)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[1:encodedtime.n] .= store.sdf2acv.allocatedarray[1:encodedtime.n]
    @views store.acv2EI.allocatedarray[encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function acv(model::TimeSeriesModel{1,T}, τ) where {T} # default acv returns error
    error("acv not yet defined for model of type $(typeof(model)).")
end

function acv(model::UnknownAcvTimeSeriesModel{1,T}, τ) where {T} # default acv returns error
    throw(ArgumentError("Custom lag only possible if model has known acv."))
end

"""
    acv!(store::Acv2EIStorageUni, model::TimeSeriesModel{1,T}, encodedtime::LagsEI)

Compute the acv at many lags and allocate to storage when acv is known in the univariate case.
"""
function acv!(store::Acv2EIStorageUni, model::TimeSeriesModel{1,T}, encodedtime::LagsEI) where {T}
    length(store.allocatedarray) == length(encodedtime.lags) || throw(ArgumentError("length(store.allocatedarray) !== length(encodedtime.lags)"))
    @inbounds for (i,τ) ∈ enumerate(encodedtime.lags)
        store.allocatedarray[i] = @views acv(model, τ)
    end
    return nothing
end

extract_acv(store::Sdf2EIStorageUni) = extract_acv(store.acv2EI)
extract_acv(store::Acv2EIStorageUni) = store.allocatedarray

function acv(model::TimeSeriesModel{1,T}, n, Δ) where {T}
    store = allocate_memory_EI_F(typeof(model), n , Δ)
    acv!(store, model) # lags -n to n-1 in extract_acv(store)
    return real.(fftshift(extract_acv(store))[2:end]) # remove imaginary floating point error
end

### Additive ###
"""
    acv!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)

Compute the acv for an additive model.

Processes storage and model, and computes acv for each separately, then combines and stores in the leftmost storage.
In other words, the sum of both autocovariances is stores in store 1.
This is preferable as for some models the acv may be known.
"""
function acv!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    acv!(store.store1, model.model1)
    acv!(store.store2, model.model2)
    extract_acv(store.store1) .+= extract_acv(store.store2) ## in this case, a call to extract_S will recover the acv as EI!() function has not yet been called.
    return nothing
end

## Gradient of autocovariance ##

### Multivariate ###

"""
    grad_acv!

Compute the gradient of the acv and allocate to storage as appropriate.
"""
function grad_acv!(store::Sdf2EIStorage, model::UnknownAcvTimeSeriesModel, encodedtime::FreqAcvEst)
    grad_asdf!(store.sdf2acv, model, encodedtime)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:, :, 1:encodedtime.n] .= store.sdf2acv.allocatedarray[:, :, 1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:, :, encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:, :, end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function grad_acv!(out, model::TimeSeriesModel, τ) # default acv returns error
    error("grad_acv! not yet defined for model of type $(typeof(model)). Maybe define as an UnknownAcvTimeSeriesModel.")
end

function grad_acv!(store::Acv2EIStorage, model::TimeSeriesModel, encodedtime::LagsEI)
    size(store.allocatedarray, 3) == length(encodedtime.lags) || throw(ArgumentError("size(store.allocatedarray, 3) !== length(encodedtime.lags)"))
    @inbounds for (i,τ) ∈ enumerate(encodedtime.lags)
        @views grad_acv!(store.allocatedarray[:, :, i], model, τ)
    end
    
    return nothing
end

function grad_acv!(store::TimeSeriesModelStorageGradient, model::TimeSeriesModel)
    grad_acv!(store.gradmemory, model, store.encodedtime)
    return nothing
end

"""
    grad_acv(model::TimeSeriesModel, τ)

Compute the gradient of the acv at `τ`.
"""
function grad_acv(model::TimeSeriesModel, τ)
    !(model isa UnknownAcvTimeSeriesModel) || throw(ArgumentError("grad_acv only defined when acv is known."))
    G = zeros(ComplexF64,nlowertriangle_dimension(model),npars(model))
    grad_acv!(G, model, τ)
    return G
end

### Univariate ###

function grad_acv!(store::Sdf2EIStorageUni, model::UnknownAcvTimeSeriesModel{1,T}, encodedtime::FreqAcvEst) where {T}
    grad_asdf!(store.sdf2acv, model, encodedtime)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:,1:encodedtime.n] .= store.sdf2acv.allocatedarray[:,1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:,encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:,end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function grad_acv!(store::Acv2EIStorageUni, model::TimeSeriesModel{1,T}, encodedtime::LagsEI) where {T}
    size(store.allocatedarray, 2) == length(encodedtime.lags) || throw(ArgumentError("size(store.allocatedarray, 2) !== length(encodedtime.lags)"))
    @inbounds for (i,τ) ∈ enumerate(encodedtime.lags)
        @views grad_acv!(store.allocatedarray[:, i], model, τ)
    end
    return nothing
end

function grad_acv(model::TimeSeriesModel{1,T}, τ) where {T}
    !(model isa UnknownAcvTimeSeriesModel) || throw(ArgumentError("grad_acv only defined when acv is known."))
    G = zeros(ComplexF64,npars(model))
    grad_acv!(G, model, τ)
    return G
end

### Additive

function grad_acv!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    grad_acv!(store.store1, model.model1)
    grad_acv!(store.store2, model.model2)
    return nothing
end

function grad_acv(model::AdditiveTimeSeriesModel, τ)
    return hcat(grad_acv(model.model1, τ), grad_acv(model.model2, τ))
end
function grad_acv(model::AdditiveTimeSeriesModel{M₁,M₂,1,T}, τ) where {M₁<:TimeSeriesModel{1,T},M₂<:TimeSeriesModel{1,T}} where {T}
    return vcat(grad_acv(model.model1, τ), grad_acv(model.model2, τ))
end

## Hessian of autocovariance ##

### Multivariate ###

"""
    hess_acv!

Compute the Hessian of the acv and allocate to storage as appropriate.
"""
function hess_acv!(store::Sdf2EIStorage, model::UnknownAcvTimeSeriesModel, encodedtime::FreqAcvEst)
    hess_asdf!(store.sdf2acv, model, encodedtime)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:, :, 1:encodedtime.n] .= store.sdf2acv.allocatedarray[:, :, 1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:, :, encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:, :, end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function hess_acv!(out, model::TimeSeriesModel, τ) # default acv returns error
    error("hess_acv! not yet defined for model of type $(typeof(model)).")
end

function hess_acv!(store::Acv2EIStorage, model::TimeSeriesModel, encodedtime::LagsEI)
    size(store.allocatedarray, 3) == length(encodedtime.lags) || throw(ArgumentError("size(store.allocatedarray, 3) !== length(encodedtime.lags)"))
    @inbounds for (i,τ) ∈ enumerate(encodedtime.lags)
        @views hess_acv!(store.allocatedarray[:, :, i], model, τ)
    end
    return nothing
end

function hess_acv!(store::TimeSeriesModelStorageHessian, model::TimeSeriesModel)
    hess_acv!(store.hessmemory, model, store.encodedtime)
    return nothing
end

function hess_acv(model::TimeSeriesModel, τ)
    !(model isa UnknownAcvTimeSeriesModel) || throw(ArgumentError("grad_acv only defined when acv is known."))
    H = zeros(ComplexF64,nlowertriangle_dimension(model),nlowertriangle_parameter(model))
    hess_acv!(H, model, τ)
    return H
end

### Univariate ###

function hess_acv!(store::Sdf2EIStorageUni, model::UnknownAcvTimeSeriesModel{1,T}, encodedtime::FreqAcvEst) where {T}
    hess_asdf!(store.sdf2acv, model, encodedtime)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:,1:encodedtime.n] .= store.sdf2acv.allocatedarray[:,1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:,encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:,end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function hess_acv!(store::Acv2EIStorageUni, model::TimeSeriesModel{1,T}, encodedtime::LagsEI) where {T}
    size(store.allocatedarray, 2) == length(encodedtime.lags) || throw(ArgumentError("size(store.allocatedarray, 2) !== length(encodedtime.lags)"))
    @inbounds for (i,τ) ∈ enumerate(encodedtime.lags)
        @views hess_acv!(store.allocatedarray[:, i], model, τ)
    end
    return nothing
end

function hess_acv(model::TimeSeriesModel{1,T}, τ) where {T}
    !(model isa UnknownAcvTimeSeriesModel) || throw(ArgumentError("grad_acv only defined when acv is known."))
    H = zeros(ComplexF64,nlowertriangle_parameter(model))
    hess_acv!(H, model, τ)
    return H
end

### Additive ###

function hess_acv!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    @views hess_acv!(store.store1, model.model1)
    @views hess_acv!(store.store2, model.model2)
    return nothing
end

function hess_acv(model::AdditiveTimeSeriesModel, τ)
    return hcat(hess_acv(model.model1, τ), hess_acv(model.model2, τ))
end
function hess_acv(model::AdditiveTimeSeriesModel{M₁,M₂,1,T}, τ) where {M₁<:TimeSeriesModel{1,T},M₂<:TimeSeriesModel{1,T}} where {T}
    return vcat(hess_acv(model.model1, τ), hess_acv(model.model2, τ))
end
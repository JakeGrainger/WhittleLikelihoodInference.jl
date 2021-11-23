## Expected periodogram ##

### Multivariate ###

"""
    EI!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageFunction, θ)

Compute the expected periodogram and assign to appropriate place in memory (note this computes at twice the desired resolution).
"""
function EI!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageFunction, θ)
    acv!(model, store, θ)
    _EI!(store.funcmemory, store.encodedtime.n, store.encodedtime.Δ)
    return nothing
end

"""
    _EI!(store, n, Δ)

Interior function for `EI!`.
"""
function _EI!(store::Sdf2EIStorage, n, Δ) # dispatch to alternate version to pull out correct storage
    _EI!(store.acv2EI, n, Δ)
    return nothing
end
function _EI!(store::Acv2EIStorage, n, Δ)
    
    for i ∈ 1:n
        @views store.allocatedarray[:, i] .*= (1 - (i - 1) / n)
    end
    for i ∈ n+1:size(store.allocatedarray, 2)
        @views store.allocatedarray[:, i] .*= (1 - (2n - i + 1) / n)
    end
    store.planned_fft * store.allocatedarray # essentially fft!(acv, 2)
    store.allocatedarray .*= Δ / (2π)
    
    return nothing
end

"""
    extract_EI(store::Sdf2EIStorage)
    extract_EI(store::Acv2EIStorage)
    extract_EI(store::Sdf2EIStorageUni)
    extract_EI(store::Acv2EIStorageUni)

Extract the EI from general storage.
"""
extract_EI(store) = extract_EI(extract_S(store))
extract_EI(store::Sdf2EIStorage) = extract_EI(store.acv2EI)
extract_EI(store::Acv2EIStorage) = store.hermitianarray

"""
    EI(model::Type{<:TimeSeriesModel}, n, Δ, θ)
    EI(model::TimeSeriesModel, n, Δ)

Compute EI at Fourier frequencies `fftshift(fftfreq(n,Δ,θ))`.

Note internal computation provides values at twice the resolution, this function returns at the desired resolution.
"""
function EI(model::Type{<:TimeSeriesModel}, n, Δ, θ)
    store = allocate_memory_EI_F(model, n , Δ)
    EI!(model, store, θ)
    return fftshift(copy(extract_EI(store)))[1:2:end]
end
EI(model::TimeSeriesModel, n, Δ) = EI(typeof(model), n, Δ, parameter(model))

### Univariate ###

function _EI!(store::Sdf2EIStorageUni, n, Δ) # dispatch to alternate version to pull out correct storage
    _EI!(store.acv2EI, n, Δ)
    return nothing
end
function _EI!(store::Acv2EIStorageUni, n, Δ)
    
    for i ∈ 1:n
        @views store.allocatedarray[i] *= (1 - (i - 1) / n)
    end
    for i ∈ n+1:size(store.allocatedarray, 2)
        @views store.allocatedarray[i] *= (1 - (2n - i + 1) / n)
    end
    store.planned_fft * store.allocatedarray # essentially fft!(acv, 1)
    store.allocatedarray .*= Δ / (2π)
    
    return nothing
end

extract_EI(store::Sdf2EIStorageUni) = extract_EI(store.acv2EI)
extract_EI(store::Acv2EIStorageUni) = store.allocatedarray

### Additive ###

"""
EI!(model::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) 

Compute the expected periodogram for additive models.
"""
function EI!(model::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    acv!(model, store, θ)
    _EI!(store.store1)
end

_EI!(store::AdditiveStorage) = _EI!(store.store1)
_EI!(store::TimeSeriesModelStorageFunction) = _EI!(store.funcmemory, store.encodedtime.n, store.encodedtime.Δ)



## Gradient of expected periodogram ##

### Multivariate ###

"""
grad_EI!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageGradient, θ)

Compute the gradient of the expected periodogram and assign to appropriate place in memory (note this computes at twice the desired resolution).
"""
function grad_EI!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageGradient, θ)
    grad_acv!(model, store, θ)
    _grad_EI!(store.gradmemory, store.encodedtime.n, store.encodedtime.Δ)
    return nothing
end

"""
    _grad_EI!(store, n, Δ)

Interior function for `grad_EI`.
"""
function _grad_EI!(store::Sdf2EIStorage, n, Δ) # dispatch to alternate version to pull out correct storage
    _grad_EI!(store.acv2EI, n, Δ)
    return nothing
end
function _grad_EI!(store::Acv2EIStorage, n, Δ)
    
    for i ∈ 1:n
        @views store.allocatedarray[:, :, i] .*= (1 - (i - 1) / n)
    end
    for i ∈ n+1:size(store.allocatedarray, 3)
        @views store.allocatedarray[:, :, i] .*= (1 - (2n - i + 1) / n)
    end
    store.planned_fft * store.allocatedarray # essentially fft!(acv, 2)
    store.allocatedarray .*= Δ / (2π)
    
    return nothing
end

### Univariate ###

function _grad_EI!(store::Sdf2EIStorageUni, n, Δ) # dispatch to alternate version to pull out correct storage
    _grad_EI!(store.acv2EI, n, Δ)
    return nothing
end
function _grad_EI!(store::Acv2EIStorageUni, n, Δ)
    
    for i ∈ 1:n
        @views store.allocatedarray[:, i] .*= (1 - (i - 1) / n)
    end
    for i ∈ n+1:size(store.allocatedarray, 2)
        @views store.allocatedarray[:, i] .*= (1 - (2n - i + 1) / n)
    end
    store.planned_fft * store.allocatedarray # essentially fft!(acv, 2)
    store.allocatedarray .*= Δ / (2π)
    
    return nothing
end

### Additive ###

function grad_EI!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views grad_EI!(M₁, store.store1, θ[1:npars(M₁)])
    @views grad_EI!(M₂, store.store2, θ[npars(M₁)+1:end])
    return nothing
end


## Hessian of expected periodogram ##

### Multivariate ###
"""
    hess_EI!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageHessian, θ)

Compute the Hessian of the expected periodogram and assign to appropriate place in memory (note this computes at twice the desired resolution).
"""
function hess_EI!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageHessian, θ)
    hess_acv!(model, store, θ)
    _hess_EI!(store.hessmemory, store.encodedtime.n, store.encodedtime.Δ)
    return nothing
end

"""
    _hess_EI!(store, n, Δ)

Interior function for `hess_EI`.
"""
function _hess_EI!(store::Sdf2EIStorage, n, Δ) # dispatch to alternate version to pull out correct storage
    _hess_EI!(store.acv2EI, n, Δ)
    return nothing
end
function _hess_EI!(store::Acv2EIStorage, n, Δ)
    
    for i ∈ 1:n
        @views store.allocatedarray[:, :, i] .*= (1 - (i - 1) / n)
    end
    for i ∈ n+1:size(store.allocatedarray, 3)
        @views store.allocatedarray[:, :, i] .*= (1 - (2n - i + 1) / n)
    end
    store.planned_fft * store.allocatedarray # essentially fft!(acv, 2)
    store.allocatedarray .*= Δ / (2π)
    
    return nothing
end

### Univariate ###

function _hess_EI!(store::Sdf2EIStorageUni, n, Δ) # dispatch to alternate version to pull out correct storage
    _hess_EI!(store.acv2EI, n, Δ)
    return nothing
end
function _hess_EI!(store::Acv2EIStorageUni, n, Δ)
    
    for i ∈ 1:n
        @views store.allocatedarray[:, i] .*= (1 - (i - 1) / n)
    end
    for i ∈ n+1:size(store.allocatedarray, 2)
        @views store.allocatedarray[:, i] .*= (1 - (2n - i + 1) / n)
    end
    store.planned_fft * store.allocatedarray # essentially fft!(acv, 2)
    store.allocatedarray .*= Δ / (2π)
    
    return nothing
end

### Additive ###

function hess_EI!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views hess_EI!(M₁, store.store1, θ[1:npars(M₁)])
    @views hess_EI!(M₂, store.store2, θ[npars(M₁)+1:end])
    return nothing
end
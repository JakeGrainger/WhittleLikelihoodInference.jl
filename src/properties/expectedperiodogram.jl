## Expected periodogram ##

### Multivariate ###

"""
    EI!(store::TimeSeriesModelStorageFunction, model::TimeSeriesModel)

Compute the expected periodogram and assign to appropriate place in memory (note this computes at twice the desired resolution).
"""
function EI!(store::TimeSeriesModelStorageFunction, model::TimeSeriesModel)
    acv!(store, model)
    _EI!(store)
    return nothing
end

"""
    _EI!(store, n, Δ)

Interior function for `EI!`.
"""
_EI!(store::TimeSeriesModelStorageFunction) = _EI!(store.funcmemory, store.encodedtime.n, store.encodedtime.Δ)
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
    EI(model::TimeSeriesModel, n, Δ)

Compute EI at Fourier frequencies `fftshift(fftfreq(n,2π/Δ))`.

Note internal computation provides values at twice the resolution, this function returns at the desired resolution.
"""
function EI(model::TimeSeriesModel, n, Δ)
    store = allocate_memory_EI_F(typeof(model), n , Δ)
    EI!(store, model)
    return fftshift(copy(extract_EI(store)))[1:2:end]
end

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
function EI!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    acv!(store, model)
    _EI!(store.store1)
end
_EI!(store::AdditiveStorage) = _EI!(store.store1) # store could contain additive store so need this for recursion.



## Gradient of expected periodogram ##

### Multivariate ###

"""
    grad_EI!(store::TimeSeriesModelStorageGradient, model::TimeSeriesModel)

Compute the gradient of the expected periodogram and assign to appropriate place in memory (note this computes at twice the desired resolution).
"""
function grad_EI!(store::TimeSeriesModelStorageGradient, model::TimeSeriesModel)
    grad_acv!(store, model)
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

function grad_EI!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    @views grad_EI!(store.store1, model.model1)
    @views grad_EI!(store.store2, model.model2)
    return nothing
end


## Hessian of expected periodogram ##

### Multivariate ###

"""
    hess_EI!(store::TimeSeriesModelStorageHessian, model::TimeSeriesModel)

Compute the Hessian of the expected periodogram and assign to appropriate place in memory (note this computes at twice the desired resolution).
"""
function hess_EI!(store::TimeSeriesModelStorageHessian, model::TimeSeriesModel)
    hess_acv!(store, model)
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

function hess_EI!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    @views hess_EI!(store.store1, model.model1)
    @views hess_EI!(store.store2, model.model2)
    return nothing
end
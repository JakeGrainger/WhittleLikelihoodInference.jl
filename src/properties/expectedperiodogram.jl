## Expected periodogram ##

### Multivariate ###

"""
    EI!(store, model::TimeSeriesModel)

Compute the expected periodogram and assign to appropriate place in memory (note this computes at twice the desired resolution).
"""
function EI!(store::Union{TimeSeriesModelStorageFunction,AdditiveStorage}, model::TimeSeriesModel)
    acv!(store, model)
    _func_EI!(store)
    return nothing
end
_func_EI!(store::AdditiveStorage) = _func_EI!(store.store1)
_func_EI!(store::TimeSeriesModelStorageFunction) = _EI!(store.funcmemory, store.encodedtime.Δ)

"""
    grad_EI!(store::TimeSeriesModelStorageGradient, model::TimeSeriesModel)

Compute the gradient of the expected periodogram and assign to appropriate place in memory (note this computes at twice the desired resolution).
"""
function grad_EI!(store::TimeSeriesModelStorageGradient, model::TimeSeriesModel)
    grad_acv!(store, model)
    _EI!(store.gradmemory, store.encodedtime.Δ)
    return nothing
end
function grad_EI!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    grad_EI!(store.store1, model.model1)
    grad_EI!(store.store2, model.model2)
    return nothing
end

"""
    hess_EI!(store::TimeSeriesModelStorageHessian, model::TimeSeriesModel)

Compute the Hessian of the expected periodogram and assign to appropriate place in memory (note this computes at twice the desired resolution).
"""
function hess_EI!(store::TimeSeriesModelStorageHessian, model::TimeSeriesModel)
    hess_acv!(store, model)
    _EI!(store.hessmemory, store.encodedtime.n, store.encodedtime.Δ)
    return nothing
end
function hess_EI!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    @views hess_EI!(store.store1, model.model1)
    @views hess_EI!(store.store2, model.model2)
    return nothing
end

_EI!(store::Sdf2EIStorage, Δ) = __EI!(store.acv2EI, Δ)
_EI!(store::Acv2EIStorage, Δ) = __EI!(store, Δ)
_EI!(store::Sdf2EIStorageUni, Δ) = __EI!(store.acv2EI, Δ)
_EI!(store::Acv2EIStorageUni, Δ) = __EI!(store, Δ)

function __EI!(store, Δ)
    mult_allocated_kernel!(store.allocatedarray, store.kernel)
    store.planned_fft * store.allocatedarray # essentially fft!(acv, ndims(store.allocatedarray))
    store.allocatedarray .*= Δ / (2π)
    return nothing
end

function mult_allocated_kernel!(array::Vector{ComplexF64},kernel)
    array .*= kernel
    nothing
end
function mult_allocated_kernel!(array::Matrix{ComplexF64},kernel)
    size(array, 2) == length(kernel) || throw(DimensionMismatch("size of second dimension of array should equal length of kernel."))
    @inbounds for i ∈ 1:size(array, 2)
        @views array[:, i] .*= kernel[i]
    end
    nothing
end
function mult_allocated_kernel!(array::Array{ComplexF64,3},kernel)
    size(array, 3) == length(kernel) || throw(DimensionMismatch("size of third dimension of array should equal length of kernel."))
    @inbounds for i ∈ 1:size(array, 3)
        @views array[:, :, i] .*= kernel[i]
    end
    nothing
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
extract_EI(store::Sdf2EIStorageUni) = extract_EI(store.acv2EI)
extract_EI(store::Acv2EIStorageUni) = store.allocatedarray

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
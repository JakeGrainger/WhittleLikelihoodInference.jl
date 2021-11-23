## Autocovariance ##

### Multivariate ###

"""
    acv!(model::Type{<:UnknownAcvTimeSeriesModel}, store::Sdf2EIStorage, encodedtime::FreqAcvEst, θ)

Approximate the acv and allocate to storage when acv is unknown.
"""
function acv!(model::Type{<:UnknownAcvTimeSeriesModel}, store::Sdf2EIStorage, encodedtime::FreqAcvEst, θ)
    asdf!(model, store.sdf2acv, encodedtime.Ωₘ, encodedtime.Δ, θ)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:, 1:encodedtime.n] .= store.sdf2acv.allocatedarray[:, 1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:, encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:, end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

"""
    acv!(model::Type{<:TimeSeriesModel}, out, τ::Number, θ)

Compute the acv for a single lag and when acv is known.
"""
function acv!(model::Type{<:TimeSeriesModel}, out, τ::Number, θ) # default acv returns error
    error("acv not yet defined for model of type $model.")
end

"""
    acv!(model::Type{<:TimeSeriesModel}, store::Acv2EIStorage, encodedtime::LagsEI, θ)

Compute the acv at many lags and allocate to storage when acv is known.
"""
function acv!(model::Type{<:TimeSeriesModel}, store::Acv2EIStorage, encodedtime::LagsEI, θ)
    for i ∈ 1:size(store.allocatedarray, 2)
        @views acv!(model, store.allocatedarray[:, i], encodedtime.lags[i], θ)
    end
    return nothing
end

"""
    acv!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageFunction, θ)

Unwrap storage and pass to lower level `acv!` call.
"""
function acv!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageFunction, θ)
    acv!(model, store.funcmemory, store.encodedtime, θ)
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
    acv(model::Type{<:TimeSeriesModel}, n, Δ, θ)
    acv(model::TimeSeriesModel, n, Δ)

Compute acv at lags `-(n-1)*Δ:Δ:(n-1)*Δ`.
"""
function acv(model::Type{<:TimeSeriesModel}, n, Δ, θ)
    store = allocate_memory_EI_F(model, n , Δ)
    acv!(model, store, θ) # lags -n to n-1 in extract_acv(store)
    A = real.(fftshift(extract_acv(store),2)[:, 2:end]) # remove imaginary floating point error
    return [[i >= j ? A[indexLT(i,j,ndims(model)), iτ] : A[indexLT(i,j,ndims(model)), size(A, 2)-iτ+1] for i = 1:ndims(model), j = 1:ndims(model)] for iτ = 1:size(A, 2)]
end
acv(model::TimeSeriesModel, n::Number, Δ::Number) = acv(typeof(model), n, Δ, parameter(model))

"""
    acv(model::Type{<:TimeSeriesModel}, lags, θ)
    acv(model::TimeSeriesModel, lags)

Compute acv at the specified lags, provided acv is known.
"""
function acv(model::Type{<:TimeSeriesModel}, lags, θ)
    !(model <: UnknownAcvTimeSeriesModel) || error("Custom lag vector only possible if model has known acv.")
    A = ones(nlowertriangle_dimension(model), 1:length(lags))
    for i ∈ 1:size(store.allocatedarray, 2)
        @views acv!(model, store[:, i], lags[i], θ)
    end
    return [[i >= j ? A[indexLT(i,j,ndims(model)), iτ] : A[indexLT(i,j,ndims(model)), size(A, 2)-iτ+1] for i = 1:ndims(model), j = 1:ndims(model)] for iτ = 1:size(A, 2)]
end
acv(model::TimeSeriesModel, lags::AbstractVector{T}) where {T<:Number} = acv(typeof(model), lags, parameter(model))

### Univariate ###

"""
    acv!(model::Type{<:UnknownAcvTimeSeriesModel{1}}, store::Sdf2EIStorageUni, encodedtime::FreqAcvEst, θ)

Approximate the acv and allocate to storage when acv is known in the univariate case.
"""
function acv!(model::Type{<:UnknownAcvTimeSeriesModel{1}}, store::Sdf2EIStorageUni, encodedtime::FreqAcvEst, θ)
    asdf!(model, store.sdf2acv, encodedtime.Ωₘ, encodedtime.Δ, θ)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[1:encodedtime.n] .= store.sdf2acv.allocatedarray[1:encodedtime.n]
    @views store.acv2EI.allocatedarray[encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function acv(model::Type{<:TimeSeriesModel{1}}, τ::Number, θ) # default acv returns error
    error("acv not yet defined for model of type $model.")
end

"""
    acv!(model::Type{<:TimeSeriesModel}, store::Acv2EIStorage, encodedtime::LagsEI, θ)

Compute the acv at many lags and allocate to storage when acv is known in the univariate case.
"""
function acv!(model::Type{<:TimeSeriesModel{1}}, store::Acv2EIStorageUni, encodedtime::LagsEI, θ)
    
    for i ∈ 1:size(store.allocatedarray, 2)
        store.allocatedarray[i] = @views acv(model, encodedtime.T[i], θ)
    end
    
    return nothing
end

extract_acv(store::Sdf2EIStorageUni) = extract_acv(store.acv2EI)
extract_acv(store::Acv2EIStorageUni) = store.allocatedarray

function acv(model::Type{<:TimeSeriesModel{1}}, n, Δ, θ)
    store = allocate_memory_EI_F(model, n , Δ)
    acv!(model, store, θ) # lags -n to n-1 in extract_acv(store)
    return real.(fftshift(extract_acv(store))[2:end]) # remove imaginary floating point error
end

function acv(model::Type{<:TimeSeriesModel{1}}, lags, θ)
    !(model <: UnknownAcvTimeSeriesModel) || error("Custom lag vector only possible if model has known acv.")
    store = ones(1:length(lags))
    for i ∈ 1:size(store.allocatedarray, 2)
        @views acv!(model, store[i], lags[i], θ)
    end
    return real.(fftshift(store,2)[2:end]) # remove imaginary floating point error
end

### Additive ###
"""
    acv!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ)

Compute the acv for an additive model.

Processes storage and model, and computes acv for each separately, then combines and stores in the leftmost storage.
In other words, the sum of both autocovariances is stores in store 1.
This is preferable as for some models the acv may be known.
"""
function acv!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views acv!(M₁, store.store1, θ[1:npars(M₁)])
    @views acv!(M₂, store.store2, θ[npars(M₁)+1:end])
    extract_acv(store.store1) .+= extract_acv(store.store2) ## in this case, a call to extract_S will recover the acv as EI!() function has not yet been called.
    return nothing
end

## Gradient of autocovariance ##

### Multivariate ###

"""
    grad_acv!

Compute the gradient of the acv and allocate to storage as appropriate.
"""
function grad_acv!(model::Type{<:UnknownAcvTimeSeriesModel}, store::Sdf2EIStorage, encodedtime::FreqAcvEst, θ)
    grad_asdf!(model, store.sdf2acv, encodedtime.Ωₘ, encodedtime.Δ, θ)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:, :, 1:encodedtime.n] .= store.sdf2acv.allocatedarray[:, :, 1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:, :, encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:, :, end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function grad_acv!(model::Type{<:TimeSeriesModel}, out, τ::Number, θ) # default acv returns error
    error("acv not yet defined for model of type $model.")
end

function grad_acv!(model::Type{<:TimeSeriesModel}, store::Acv2EIStorage, encodedtime::LagsEI, θ)
    
    for i ∈ 1:size(store.allocatedarray, 3)
        @views grad_acv!(model, store.allocatedarray[:, :, i], encodedtime.T[i], θ)
    end
    
    return nothing
end

function grad_acv!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageGradient, θ)
    grad_acv!(model, store.gradmemory, store.encodedtime, θ)
    return nothing
end

### Univariate ###

function grad_acv!(model::Type{<:UnknownAcvTimeSeriesModel{1}}, store::Sdf2EIStorageUni, encodedtime::FreqAcvEst, θ)
    grad_asdf!(model, store.sdf2acv, encodedtime.Ωₘ, encodedtime.Δ, θ)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:,1:encodedtime.n] .= store.sdf2acv.allocatedarray[:,1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:,encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:,end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function grad_acv!(model::Type{<:TimeSeriesModel{1}}, store::Acv2EIStorageUni, encodedtime::LagsEI, θ)
    
    for i ∈ 1:size(store.allocatedarray, 2)
        @views grad_acv!(model, store.allocatedarray[:, i], encodedtime.T[i], θ)
    end
    
    return nothing
end

### Additive

function grad_acv!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views grad_acv!(M₁, store.store1, θ[1:npars(M₁)])
    @views grad_acv!(M₂, store.store2, θ[npars(M₁)+1:end])
    return nothing
end


## Hessian of autocovariance ##

### Multivariate ###

"""
    hess_acv!

Compute the Hessian of the acv and allocate to storage as appropriate.
"""
function hess_acv!(model::Type{<:UnknownAcvTimeSeriesModel}, store::Sdf2EIStorage, encodedtime::FreqAcvEst, θ)
    hess_asdf!(model, store.sdf2acv, encodedtime.Ωₘ, encodedtime.Δ, θ)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:, :, 1:encodedtime.n] .= store.sdf2acv.allocatedarray[:, :, 1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:, :, encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:, :, end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function hess_acv!(model::Type{<:TimeSeriesModel}, out, τ::Number, θ) # default acv returns error
    error("acv not yet defined for model of type $model.")
end

function hess_acv!(model::Type{<:TimeSeriesModel}, store::Acv2EIStorage, encodedtime::LagsEI, θ)
    
    for i ∈ 1:size(store.allocatedarray, 3)
        @views hess_acv!(model, store.allocatedarray[:, :, i], encodedtime.T[i], θ)
    end
    
    return nothing
end

function hess_acv!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageHessian, θ)
    hess_acv!(model, store.hessmemory, store.encodedtime, θ)
    return nothing
end

### Univariate ###

function hess_acv!(model::Type{<:UnknownAcvTimeSeriesModel{1}}, store::Sdf2EIStorageUni, encodedtime::FreqAcvEst, θ)
    hess_asdf!(model, store.sdf2acv, encodedtime.Ωₘ, encodedtime.Δ, θ)
    
    store.sdf2acv.planned_ifft * store.sdf2acv.allocatedarray # essentially ifft!(asdf, 2)

    @views store.acv2EI.allocatedarray[:,1:encodedtime.n] .= store.sdf2acv.allocatedarray[:,1:encodedtime.n]
    @views store.acv2EI.allocatedarray[:,encodedtime.n+1:end] .= store.sdf2acv.allocatedarray[:,end-encodedtime.n+1:end]
    
    store.acv2EI.allocatedarray .*= (2π/encodedtime.Δ) # faster than doing before pulling out indicies
    return nothing
end

function hess_acv!(model::Type{<:TimeSeriesModel{1}}, store::Acv2EIStorageUni, encodedtime::LagsEI, θ)
    
    for i ∈ 1:size(store.allocatedarray, 2)
        @views hess_acv!(model, store.allocatedarray[:, i], encodedtime.T[i], θ)
    end
    
    return nothing
end

### Additive ###

function hess_acv!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views hess_acv!(M₁, store.store1, θ[1:npars(M₁)])
    @views hess_acv!(M₂, store.store2, θ[npars(M₁)+1:end])
    return nothing
end
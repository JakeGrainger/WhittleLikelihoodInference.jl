## Spectral density ##

### Multivariate ###

"""
    add_sdf!(out, model::TimeSeriesModel, ω)

Add the sdf of a model to a storage vector.
"""
function add_sdf!(out, model::TimeSeriesModel, ω) # default sdf returns error
    error("sdf not yet defined for model of type $(typeof(model)).")
end

"""
    add_asdf!(out, model::TimeSeriesModel, ω, Δ)

Add the aliased sdf of a model to a storage vector.
"""
function add_asdf!(out, model::TimeSeriesModel, ω, Δ) # default method approximates the asdf (overload if known)
    for k ∈ -nalias(model):nalias(model)
        add_sdf!(out, model, ω + k * 2π / Δ)
    end
    return nothing
end

"""
    asdf!(out, model::TimeSeriesModel, ω, Δ)

Computes the asdf and overwrites a storage vector.
"""
function asdf!(out, model::TimeSeriesModel, ω, Δ)
    out .= zero(eltype(out))
    add_asdf!(out, model, ω, Δ)
    return nothing
end

"""
    asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel, freq::FreqAcvEst)

Compute the asdf for all frequencies and allocates to appropriate location in storage.
"""
function asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel, freq::FreqAcvEst)
    size(store.allocatedarray,2) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,2) != length(Ω)."))
    @inbounds for i ∈ 1:length(freq.Ωₘ)
        @views asdf!(store.allocatedarray[:, i], model, freq.Ωₘ[i], freq.Δ)
    end
    return nothing
end
function asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel{D,T}, freq::FreqAcvEst) where {D,T<:Real} ## real values
    size(store.allocatedarray,2) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,2) != length(Ω)."))
    m = length(freq.Ωₘ)
    startcopyind = m÷2+2
    @inbounds for i ∈ 1:startcopyind-1
        @views asdf!(store.allocatedarray[:, i], model, freq.Ωₘ[i], freq.Δ)
    end
    @inbounds for i in startcopyind:length(freq.Ωₘ) 
        @views store.allocatedarray[:, i] .= conj.(store.allocatedarray[:, m-i+2])
    end
    return nothing
end
function asdf!(store::TimeSeriesModelStorageFunction, model::TimeSeriesModel)
    asdf!(store.funcmemory, model, store.encodedtime)
    return nothing
end

"""
    sdf(model::TimeSeriesModel, ω)

Compute sdf at `ω`.
"""
function sdf(model::TimeSeriesModel, ω)
    out = zeros(ComplexF64, nlowertriangle_dimension(model))
    add_sdf!(out, model, ω)
    return SHermitianCompact(SVector{nlowertriangle_dimension(model)}(out))
end

"""
    asdf(model::TimeSeriesModel, ω, Δ)

Compute asdf with sampling rate `Δ` at `ω`.
"""
function asdf(model::TimeSeriesModel, ω, Δ)
    out = zeros(ComplexF64, nlowertriangle_dimension(model))
    asdf!(out, model, ω, Δ)
    return SHermitianCompact(SVector{nlowertriangle_dimension(model)}(out))
end

### Univariate ###
function sdf(model::TimeSeriesModel{1,T}, ω) where {T} # default sdf returns error
    error("sdf not yet defined for model of type $(typeof(model)).")
end
function asdf(model::TimeSeriesModel{1,T}, ω, Δ) where {T} # default method approximates the asdf (overload if known)
    val = zero(ComplexF64)
    for k ∈ -nalias(model):nalias(model)
        val += sdf(model, ω + k * 2π / Δ)
    end
    return val
end

function asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel{1,T}, freq::FreqAcvEst) where {T}
    length(store.allocatedarray) == length(freq.Ωₘ) || throw(ArgumentError("length(store.allocatedarray) != length(Ω)."))
    @inbounds for i ∈ 1:length(freq.Ωₘ)
        store.allocatedarray[i] = asdf(model, freq.Ωₘ[i], freq.Δ)
    end
    return nothing
end
function asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel{1,T}, freq::FreqAcvEst) where {T<:Real} ## real values
    length(store.allocatedarray) == length(freq.Ωₘ) || throw(ArgumentError("length(store.allocatedarray) != length(Ω)."))
    m = length(freq.Ωₘ)
    startcopyind = m÷2+2
    @inbounds for i ∈ 1:startcopyind-1
        store.allocatedarray[i] = asdf(model, freq.Ωₘ[i], freq.Δ)
    end
    @inbounds for i ∈ startcopyind:length(freq.Ωₘ) 
        store.allocatedarray[i] = store.allocatedarray[m-i+2]
    end
    return nothing
end

### Additive ###
function add_sdf!(out, model::AdditiveTimeSeriesModel, ω)
    add_sdf!(out, model.model1, ω)
    add_sdf!(out, model.model2, ω)
    return nothing
end
function add_asdf!(out, model::AdditiveTimeSeriesModel, ω, Δ)
    add_asdf!(out, model.model1, ω, Δ)
    add_asdf!(out, model.model2, ω, Δ)
    return nothing
end
function sdf(model::AdditiveTimeSeriesModel{M₁,M₂,1,T}, ω) where {M₁<:TimeSeriesModel{1,T},M₂<:TimeSeriesModel{1,T}} where {T}
    return sdf(model.model1,ω)+sdf(model.model2,ω)
end
function asdf(model::AdditiveTimeSeriesModel{M₁,M₂,1,T}, ω, Δ) where {M₁<:TimeSeriesModel{1,T},M₂<:TimeSeriesModel{1,T}} where {T}
    return asdf(model.model1,ω,Δ)+asdf(model.model2,ω,Δ)
end

extract_asdf(store::SdfStorage) = store.allocatedarray
extract_asdf(store::SdfStorageUni) = store.allocatedarray

function asdf!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    @views asdf!(store.store1, model.model1)
    @views asdf!(store.store2, model.model2)
    extract_asdf(extract_S(store.store1)) .+= extract_asdf(extract_S(store.store2))
    return nothing
end

## Gradient of spectral density

### Multivariate ###

"""
    grad_add_sdf!(out, model::TimeSeriesModel, ω)

Add the gradient of the sdf of a model to an input storage vector.
"""
function grad_add_sdf!(out, model::TimeSeriesModel, ω) # default sdf returns error
    error("grad_add_sdf! not yet defined for model of type $(typeof(model)).")
end

"""
    grad_add_asdf!(out, model::TimeSeriesModel, ω, Δ)

Add the gradient of the aliased sdf of a model to an input storage vector.
"""
function grad_add_asdf!(out, model::TimeSeriesModel, ω, Δ) # default method approximates the asdf (overload if known)
    for k ∈ -nalias(model):nalias(model)
        grad_add_sdf!(out, model, ω + k * 2π / Δ)
    end
    return nothing
end

"""
    grad_asdf!(out, model::TimeSeriesModel, ω, Δ)

Compute the gradient of the aliased sdf of a model and store in storage vector.
"""
function grad_asdf!(out, model::TimeSeriesModel, ω, Δ)
    out .= zero(eltype(out))
    grad_add_asdf!(out, model, ω, Δ)
    return nothing
end

"""
    grad_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel, freq::FreqAcvEst)
    grad_asdf!(store::TimeSeriesModelStorageGradient, model::TimeSeriesModel)

Compute the gradient of the asdf for all frequencies and allocate to appropriate location in storage.
"""
function grad_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel, freq::FreqAcvEst)
    size(store.allocatedarray,3) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,3) != length(freq.Ωₘ)."))
    @inbounds for i ∈ 1:length(freq.Ωₘ)
        @views grad_asdf!(store.allocatedarray[:, :, i], model, freq.Ωₘ[i], freq.Δ)
    end
    return nothing
end
function grad_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel{D,T}, freq::FreqAcvEst) where {D,T<:Real} # real case
    size(store.allocatedarray,3) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,3) != length(freq.Ωₘ)."))
    m = length(freq.Ωₘ)
    startcopyind = m÷2+2
    @inbounds for i ∈ 1:startcopyind-1
        @views grad_asdf!(store.allocatedarray[:, :, i], model, freq.Ωₘ[i], freq.Δ)
    end
    @inbounds for i in startcopyind:length(freq.Ωₘ) 
        @views store.allocatedarray[:, :, i] .= conj.(store.allocatedarray[:, :, m-i+2])
    end
    return nothing
end
function grad_asdf!(store::TimeSeriesModelStorageGradient, model::TimeSeriesModel)
    grad_asdf!(store.gradmemory, model, store.encodedtime)
    return nothing
end

"""
    grad_sdf(model::TimeSeriesModel, ω)

Compute the gradient of the sdf at `ω`.
"""
function grad_sdf(model::TimeSeriesModel, ω)
    G = zeros(ComplexF64,nlowertriangle_dimension(model),npars(model))
    grad_add_sdf!(G, model, ω)
    return G
end

### Univariate ###

function grad_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel{1,T}, freq::FreqAcvEst) where {T}
    size(store.allocatedarray,2) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,2) != length(freq.Ωₘ)."))
    @inbounds for i ∈ 1:length(freq.Ωₘ)
        @views grad_asdf!(store.allocatedarray[:, i], model, freq.Ωₘ[i], freq.Δ)
    end
    return nothing
end
function grad_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel{1,T}, freq::FreqAcvEst) where {T<:Real} # real case
    size(store.allocatedarray,2) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,2) != length(freq.Ωₘ)."))
    m = length(freq.Ωₘ)
    startcopyind = m÷2+2
    @inbounds for i ∈ 1:startcopyind-1
        @views grad_asdf!(store.allocatedarray[:, i], model, freq.Ωₘ[i], freq.Δ)
    end
    @inbounds for i in startcopyind:length(freq.Ωₘ) 
        @views store.allocatedarray[:, i] .= store.allocatedarray[:, m-i+2]
    end
    return nothing
end
function grad_sdf(model::TimeSeriesModel{1,T}, ω) where {T}
    G = zeros(ComplexF64,npars(model))
    grad_add_sdf!(G, model, ω)
    return G
end

### Additive ###

function grad_add_sdf!(out, model::AdditiveTimeSeriesModel, ω)
    @views grad_add_sdf!(out[1:npars(M₁)], model.model1, ω)
    @views grad_add_sdf!(out[npars(M₁)+1:end], model.model2, ω)
    return nothing
end

function grad_add_asdf!(out, model::AdditiveTimeSeriesModel, ω, Δ)
    @views grad_add_asdf!(out[1:npars(M₁)], model.model1, ω, Δ)
    @views grad_add_asdf!(out[npars(M₁)+1:end], model.model2, ω, Δ)
    return nothing
end

function grad_asdf!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    @views grad_asdf!(store.store1, model.model1)
    @views grad_asdf!(store.store2, model.model2)
    return nothing
end

function grad_sdf(model::AdditiveTimeSeriesModel, ω)
    return hcat(grad_sdf(model.model1, ω), grad_sdf(model.model2, ω)) # not very efficient implementation but not designed to be used.
end
function grad_sdf(model::AdditiveTimeSeriesModel{M₁,M₂,1,T}, ω) where {M₁<:TimeSeriesModel{1,T},M₂<:TimeSeriesModel{1,T}} where {T}
    return vcat(grad_sdf(model.model1, ω), grad_sdf(model.model2, ω)) # not very efficient implementation but not designed to be used.
end

## Hessian of spectral density

"""
    hess_add_sdf!(out, model::TimeSeriesModel, ω)

Add the Hessian of the sdf of a model to an input storage vector.
"""
function hess_add_sdf!(out, model::TimeSeriesModel, ω) # default sdf returns error
    error("hess_add_sdf! not yet defined for model of type $(typeof(model)).")
end

"""
    hess_add_asdf!(out, model::Type{<:TimeSeriesModel}, ω, Δ)

Add the Hessian of the aliased sdf of a model to an input storage vector.
"""
function hess_add_asdf!(out, model::TimeSeriesModel, ω, Δ) # default method approximates the asdf (overload if known)
    for k ∈ -nalias(model):nalias(model)
        hess_add_sdf!(out, model, ω + k * 2π / Δ)
    end
    return nothing
end

"""
    hess_asdf!(out, model::TimeSeriesModel, ω, Δ)

Compute the Hessian of the aliased sdf of a model and store in storage vector.
"""
function hess_asdf!(out, model::TimeSeriesModel, ω, Δ)
    out .= zero(eltype(out))
    hess_add_asdf!(out, model, ω, Δ)
    return nothing
end

"""
    hess_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel, freq::FreqAcvEst)
    hess_asdf!(store::TimeSeriesModelStorageHessian, model::TimeSeriesModel)

Compute the Hessian of the asdf for all frequencies and allocate to appropriate location in storage.
"""
function hess_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel, freq::FreqAcvEst)
    size(store.allocatedarray,3) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,3) != length(freq.Ωₘ)."))
    @inbounds for i ∈ 1:length(freq.Ωₘ)
        @views hess_asdf!(store.allocatedarray[:, :, i], model, freq.Ωₘ[i], freq.Δ)
    end
    return nothing
end
function hess_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel{D,T}, freq::FreqAcvEst) where {D,T<:Real} # real case
    size(store.allocatedarray,3) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,3) != length(freq.Ωₘ)."))
    m = length(freq.Ωₘ)
    startcopyind = m÷2+2
    @inbounds for i ∈ 1:startcopyind-1
        @views grad_asdf!(store.allocatedarray[:, :, i], model, freq.Ωₘ[i], freq.Δ)
    end
    @inbounds for i in startcopyind:length(freq.Ωₘ) 
        @views store.allocatedarray[:, :, i] .= conj.(store.allocatedarray[:, :, m-i+2])
    end
    return nothing
end
function hess_asdf!(store::TimeSeriesModelStorageHessian, model::TimeSeriesModel)
    hess_asdf!(store.hessmemory, model, store.encodedtime)
    return nothing
end

"""
    hess_sdf(model::TimeSeriesModel, ω)

Compute the Hessian of the sdf at `ω`.
"""
function hess_sdf(model::TimeSeriesModel, ω)
    H = zeros(ComplexF64,nlowertriangle_dimension(model),nlowertriangle_parameter(model))
    hess_add_sdf!(H, model, ω)
    return H
end

### Univariate ###

function hess_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel{1,T}, freq::FreqAcvEst) where {T}
    size(store.allocatedarray,2) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,2) != length(freq.Ωₘ)."))
    @inbounds for i ∈ 1:length(freq.Ωₘ)
        @views hess_asdf!(store.allocatedarray[:, i], model, freq.Ωₘ[i], freq.Δ)
    end
    return nothing
end
function hess_asdf!(store::TimeSeriesModelStorage, model::TimeSeriesModel{1,T}, freq::FreqAcvEst) where {T<:Real} # real case
    size(store.allocatedarray,2) == length(freq.Ωₘ) || throw(ArgumentError("size(store.allocatedarray,2) != length(freq.Ωₘ)."))
    m = length(freq.Ωₘ)
    startcopyind = m÷2+2
    @inbounds for i ∈ 1:startcopyind-1
        @views grad_asdf!(store.allocatedarray[:, i], model, freq.Ωₘ[i], freq.Δ)
    end
    @inbounds for i in startcopyind:length(freq.Ωₘ) 
        @views store.allocatedarray[:, i] .= store.allocatedarray[:, m-i+2]
    end
    return nothing
end
function hess_sdf(model::TimeSeriesModel{1,T}, ω) where {T}
    H = zeros(ComplexF64,nlowertriangle_parameter(model))
    hess_add_sdf!(H, model, ω)
    return H
end

### Additive ###

function hess_asdf!(store::AdditiveStorage, model::AdditiveTimeSeriesModel)
    @views hess_asdf!(store.store1, model.model1)
    @views hess_asdf!(store.store2, model.model2)
    return nothing
end

function hess_sdf(model::AdditiveTimeSeriesModel, ω)
    return hcat(hess_sdf(model.model1, ω), hess_sdf(model.model2, ω)) # not very efficient implementation but not designed to be used.
end
function hess_sdf(model::AdditiveTimeSeriesModel{M₁,M₂,1,T}, ω) where {M₁<:TimeSeriesModel{1,T},M₂<:TimeSeriesModel{1,T}} where {T}
    return vcat(hess_sdf(model.model1, ω), hess_sdf(model.model2, ω)) # not very efficient implementation but not designed to be used.
end
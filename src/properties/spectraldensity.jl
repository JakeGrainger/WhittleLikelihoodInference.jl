## Spectral density ##

### Multivariate ###

"""
    add_sdf!(model::Type{<:TimeSeriesModel}, out, ω, θ)

Add the sdf of a model to a storage vector.
"""
function add_sdf!(model::Type{<:TimeSeriesModel}, out, ω, θ) # default sdf returns error
    error("sdf not yet defined for model of type $model.")
end

"""
    add_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ)

Add the aliased sdf of a model to a storage vector.
"""
function add_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ) # default method approximates the asdf (overload if known)
    for k ∈ -nalias(model):nalias(model)
        add_sdf!(model, out, ω + k * 2π / Δ, θ)
    end
    return nothing
end

"""
    asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ)

Computes the asdf and overwrites a storage vector.
"""
function asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ)
    out .= zero(eltype(out))
    add_asdf!(model, out, ω, Δ, θ)
    return nothing
end

"""
    asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorage, Ω, Δ, θ)

Compute the asdf for all frequencies and allocates to appropriate location in storage."
"""
function asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorage, Ω, Δ, θ)
    for i ∈ 1:size(store.allocatedarray, 2)
        @views asdf!(model, store.allocatedarray[:, i], Ω[i], Δ, θ)
    end
    return nothing
end
function asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageFunction, θ)
    asdf!(model, store.funcmemory, store.encodedtime.Ωₘ, store.encodedtime.Δ, θ)
    return nothing
end

"""
    sdf(model::Type{<:TimeSeriesModel}, ω, θ)
    sdf(model::TimeSeriesModel, ω)

Compute sdf at `ω`.
"""
function sdf(model::Type{<:TimeSeriesModel}, ω, θ)
    out = zeros(ComplexF64, nlowertriangle_dimension(model))
    add_sdf!(model, out, ω, θ)
    return SHermitianCompact(SVector{nlowertriangle_dimension(model)}(out))
end
sdf(model::TimeSeriesModel, ω) = sdf(typeof(model), ω, parameter(model))

"""
    asdf(model::Type{<:TimeSeriesModel}, ω, Δ, θ)
    asdf(model::TimeSeriesModel, ω, Δ)

Compute asdf with sampling rate `Δ` at `ω`.
"""
function asdf(model::Type{<:TimeSeriesModel}, ω, Δ, θ)
    out = zeros(ComplexF64, nlowertriangle_dimension(model))
    asdf!(model, out, ω, Δ, θ)
    return SHermitianCompact(SVector{nlowertriangle_dimension(model)}(out))
end
asdf(model::TimeSeriesModel, ω, Δ) = asdf(typeof(model), ω, Δ, parameter(model))

### Univariate ###

function asdf(model::Type{<:TimeSeriesModel{1}}, ω, Δ, θ) # default method approximates the asdf (overload if known)
    val = zero(ComplexF64)
    for k ∈ -nalias(model):nalias(model)
        val += sdf(model, ω + k * 2π / Δ, θ) 
    end
    return val
end

function asdf!(model::Type{<:TimeSeriesModel{1}}, store::TimeSeriesModelStorage, Ω, Δ, θ)
    for i ∈ 1:length(store.allocatedarray)
        store.allocatedarray[i] = @views asdf(model, Ω[i], Δ, θ)
    end
    return nothing
end

### Additive ###

function add_sdf!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, out, ω, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views add_sdf!(M₁, out, ω, θ[1:npars(M₁)])
    @views add_sdf!(M₂, out, ω, θ[npars(M₁)+1:end])
    return nothing
end

function add_asdf!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, out, ω, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views add_asdf!(M₁, out, ω, Δ, θ[1:npars(M₁)])
    @views add_asdf!(M₂, out, ω, Δ, θ[npars(M₁)+1:end])
    return nothing
end

extract_asdf(store) = extract_asdf(extract_S(store))
extract_asdf(store::SdfStorage) = store.allocatedarray

function asdf!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views asdf!(M₁, store.store1, θ[1:npars(M₁)])
    @views asdf!(M₂, store.store2, θ[npars(M₁)+1:end])
    extract_asdf(store.store1) .+= extract_asdf(store.store2)
    return nothing
end

## Gradient of spectral density

### Multivariate ###

"""
    grad_add_sdf!(model::Type{<:TimeSeriesModel}, out, ω, θ)

Add the gradient of the sdf of a model to an input storage vector.
"""
function grad_add_sdf!(model::Type{<:TimeSeriesModel}, out, ω, θ) # default sdf returns error
    error("sdf not yet defined for model of type $model.")
end

"""
    grad_add_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ)

Add the gradient of the aliased sdf of a model to an input storage vector.
"""
function grad_add_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ) # default method approximates the asdf (overload if known)
    for k ∈ -nalias(model):nalias(model)
        grad_add_sdf!(model, out, ω + k * 2π / Δ, θ)
    end
    return nothing
end

"""
    grad_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ)

Compute the gradient of the aliased sdf of a model and store in storage vector.
"""
function grad_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ)
    out .= zero(eltype(out))
    grad_add_asdf!(model, out, ω, Δ, θ)
    return nothing
end

"""
    grad_asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorage, Ω, Δ, θ)
    grad_asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageGradient, θ)

Compute the gradient of the asdf for all frequencies and allocate to appropriate location in storage.
"""
function grad_asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorage, Ω, Δ, θ)
    for i ∈ 1:size(store.allocatedarray, 3)
        @views grad_asdf!(model, store.allocatedarray[:, :, i], Ω[i], Δ, θ)
    end
    return nothing
end
function grad_asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageGradient, θ)
    grad_asdf!(model, store.gradmemory, store.encodedtime.Ωₘ, store.encodedtime.Δ, θ)
    return nothing
end

"""
    grad_sdf(model::Type{<:TimeSeriesModel}, ω, θ)

Compute the gradient of the sdf at `ω`.
"""
function grad_sdf(model::Type{<:TimeSeriesModel}, ω, θ)
    G = zeros(ComplexF64,nlowertriangle_dimension(model),length(θ))
    grad_add_sdf!(model, G, ω, θ)
    return G
end

### Univariate ###

function grad_asdf!(model::Type{<:TimeSeriesModel{1}}, store::TimeSeriesModelStorage, Ω, Δ, θ)
    for i ∈ 1:size(store.allocatedarray, 2)
        @views grad_asdf!(model, store.allocatedarray[:, i], Ω[i], Δ, θ)
    end
    return nothing
end

### Additive ###

function add_grad_sdf!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, out, ω, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views add_grad_sdf!(M₁, out[1:npars(M₁)], ω, θ[1:npars(M₁)])
    @views add_grad_sdf!(M₂, out[npars(M₁)+1:end], ω, θ[npars(M₁)+1:end])
    return nothing
end

function add_grad_asdf!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, out, ω, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views add_grad_asdf!(M₁, out[1:npars(M₁)], ω, Δ, θ[1:npars(M₁)])
    @views add_grad_asdf!(M₂, out[npars(M₁)+1:end], ω, Δ, θ[npars(M₁)+1:end])
    return nothing
end

function grad_asdf!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views grad_asdf!(M₁, store.store1, θ[1:npars(M₁)])
    @views grad_asdf!(M₂, store.store2, θ[npars(M₁)+1:end])
    return nothing
end

## Hessian of spectral density

"""
    hess_add_sdf!(model::Type{<:TimeSeriesModel}, out, ω, θ)

Add the Hessian of the sdf of a model to an input storage vector.
"""
function hess_add_sdf!(model::Type{<:TimeSeriesModel}, out, ω, θ) # default sdf returns error
    error("sdf not yet defined for model of type $model.")
end

"""
    hess_add_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ)

Add the Hessian of the aliased sdf of a model to an input storage vector.
"""
function hess_add_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ) # default method approximates the asdf (overload if known)
    for k ∈ -nalias(model):nalias(model)
        hess_add_sdf!(model, out, ω + k * 2π / Δ, θ)
    end
    return nothing
end

"""
    hess_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ)

Compute the Hessian of the aliased sdf of a model and store in storage vector.
"""
function hess_asdf!(model::Type{<:TimeSeriesModel}, out, ω, Δ, θ)
    out .= zero(eltype(out))
    hess_add_asdf!(model, out, ω, Δ, θ)
    return nothing
end

"""
    hess_asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorage, Ω, Δ, θ)
    hess_asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageHessian, θ)

Compute the Hessian of the asdf for all frequencies and allocate to appropriate location in storage.
"""
function hess_asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorage, Ω, Δ, θ)
    for i ∈ 1:size(store.allocatedarray, 3)
        @views hess_asdf!(model, store.allocatedarray[:, :, i], Ω[i], Δ, θ)
    end
    return nothing
end
function hess_asdf!(model::Type{<:TimeSeriesModel}, store::TimeSeriesModelStorageHessian, θ)
    hess_asdf!(model, store.hessmemory, store.encodedtime.Ωₘ, store.encodedtime.Δ, θ)
    return nothing
end

"""
    hess_sdf(model::Type{<:TimeSeriesModel}, ω, θ)

Compute the Hessian of the sdf at `ω`.
"""
function hess_sdf(model, ω, θ)
    H = zeros(ComplexF64,nlowertriangle_dimension(model),nlowertriangle_parameter(model))
    hess_add_sdf!(model, H, ω, θ)
    return H
end

### Univariate ###

function hess_asdf!(model::Type{<:TimeSeriesModel{1}}, store::TimeSeriesModelStorage, Ω, Δ, θ)
    for i ∈ 1:size(store.allocatedarray, 3)
        @views hess_asdf!(model, store.allocatedarray[:, i], Ω[i], Δ, θ)
    end
    return nothing
end

### Additive ###

function hess_asdf!(::Type{AdditiveTimeSeriesModel{M₁,M₂,D}}, store::AdditiveStorage, θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
    @views hess_asdf!(M₁, store.store1, θ[1:npars(M₁)])
    @views hess_asdf!(M₂, store.store2, θ[npars(M₁)+1:end])
    return nothing
end
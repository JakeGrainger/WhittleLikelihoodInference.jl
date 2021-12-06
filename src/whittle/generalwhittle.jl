## Be careful, some of this functionality depends on the storage type provided. 
## Use higher level function calls for Whittle and de-biased Whitle if possible!

"Low level function to compute the Whittle summand given a spectral quantity S (the sdf or EI) and periodogram ordinate I."
generalwhittle_summand(S::SHermitianCompact, I) = real(log(det(S)) + tr(I*inv(S)))
generalwhittle_summand(S::Matrix{T}, I) where {T} = real(log(det(S)) + tr(I/S))
generalwhittle_summand(S::T, I) where {T<:Number} = real(log(S) + I/S)

"Function to compute the Whittle likelihood from the Periodogram and some spectral quantity (sdf or EI)."
generalwhittle(S, I::Vector{T}) where {T} = sum(generalwhittle_summand(Sᵢ,Iᵢ) for (Sᵢ,Iᵢ) ∈ zip(S,I))
generalwhittle(funcmemory, data::GenWhittleData) = @views generalwhittle(unpack(funcmemory)[data.Ω_used_index], data.I)
generalwhittle(store::TimeSeriesModelStorageFunction, data::GenWhittleData) = generalwhittle(store.funcmemory, data)
generalwhittle(store::AdditiveStorage, data::GenWhittleData) = generalwhittle(store.store1, data) # function value will always be stored in leftmost storage.

## gradient
"Function to compute the gradient of the Whittle likelihood from the Periodogram and some spectral quantity (sdf or EI)."
function grad_generalwhittle!(∇ℓ, S, ∇S, I::Vector{T}) where {T<:Number} # univariate case
    length(S) == length(I) == size(∇S,2) || throw(ArgumentError("S and I should be same length as second dim of ∇S."))
    ∇ℓ .= zero(eltype(∇ℓ))
    @inbounds for iω = 1:length(S)
        Ipart = (1.0/real(S[iω]) - I[iω]/(real(S[iω])^2))
        for jpar = 1:size(∇S, 1)
            ∇ℓ[jpar] += Ipart * real(∇S[jpar, iω])
        end
    end
    return nothing
end
function grad_generalwhittle!(∇ℓ, S, ∇S, I::Vector{SMatrix{D,D,T,n}}) where {D,T,n} # multivariate case
    length(S) == length(I) == size(∇S,2) || throw(ArgumentError("S and I should be same length as second dim of ∇S."))
    ∇ℓ .= zero(eltype(∇ℓ))
    Id = diagm(@SVector ones(D))
    @inbounds for iω = 1:length(S)
        invS = inv(S[iω])
        Ipart = (Id - I[iω] * invS)
        for jpar = 1:size(∇S, 1)
            ∇ℓ[jpar] += real(tr(Ipart * ∇S[jpar, iω] * invS))
        end
    end
    return nothing
end
grad_generalwhittle!(∇ℓ, funcmemory, gradmemory, data::GenWhittleData) = @views grad_generalwhittle!(∇ℓ, unpack(funcmemory)[data.Ω_used_index], unpack(gradmemory)[:,data.Ω_used_index], data.I)
grad_generalwhittle!(∇ℓ, store::TimeSeriesModelStorageGradient, data::GenWhittleData) = grad_generalwhittle!(∇ℓ, store.funcmemory, store.gradmemory, data)
grad_generalwhittle!(∇ℓ, funcmemory, store::TimeSeriesModelStorageGradient, data::GenWhittleData) = grad_generalwhittle!(∇ℓ, funcmemory, store.gradmemory, data)
function grad_generalwhittle!(∇ℓ, objS, store::AdditiveStorage, data::GenWhittleData)
    @views grad_generalwhittle!(∇ℓ[1:store.npar1], objS, store.store1, data)
    @views grad_generalwhittle!(∇ℓ[store.npar1+1:end], objS, store.store2, data)
end
grad_generalwhittle!(∇ℓ, store::AdditiveStorage, data::GenWhittleData) = grad_generalwhittle!(∇ℓ, extract_S(store), store::AdditiveStorage, data::GenWhittleData)



## hessian
"Function to compute the hessian of the Whittle likelihood from the Periodogram and some spectral quantity (sdf or EI)."
function hess_generalwhittle!(∇²ℓ, S, ∇S, ∇²S, I::Vector{T}) where {T <: Number} # univariate
    length(S) == length(I) == size(∇S,2) == size(∇²S,2) || throw(ArgumentError("S and I should be same length as second dim of ∇S and ∇²S."))
    ∇²ℓ .= zero(eltype(∇²ℓ))
    @inbounds for iω = 1:length(S) # fill lower triangle
        invS = inv(S[iω])
        otherIpart = (I[iω] * invS)
        Ipart = (1 - otherIpart)
        for jpar = 1:size(∇S, 1)
            ∂SⱼinvS = ∇S[jpar, iω] * invS
            for kpar = 1:jpar
                ∂SₖinvS = ∇S[kpar, iω] * invS
                ∇²ℓ[jpar, kpar] += real((otherIpart * ∂SⱼinvS) * ∂SₖinvS + Ipart * (∇²S[indexLT(jpar,kpar,size(∇S, 1)), iω] * invS - ∂SⱼinvS * ∂SₖinvS) )
            end
        end
    end
    for jpar = 1:size(∇S, 1)-1, kpar = jpar+1:size(∇S, 1) # fill upper triangle from symmetry
        @views ∇²ℓ[jpar, kpar] = ∇²ℓ[kpar, jpar]
    end
    return nothing
end
function hess_generalwhittle!(∇²ℓ, S, ∇S, ∇²S, I::Vector{SMatrix{D,D,T,n}}) where {D,T,n}
    length(S) == length(I) == size(∇S,2) == size(∇²S,2) || throw(ArgumentError("S and I should be same length as second dim of ∇S and ∇²S."))
    ∇²ℓ .= zero(eltype(∇²ℓ))
    Id = diagm(@SVector ones(D))
    @inbounds for iω = 1:length(S) # fill lower triangle
        invS = inv(S[iω])
        otherIpart = (I[iω] * invS)
        Ipart = (Id - otherIpart)
        for jpar = 1:size(∇S, 1)
            ∂SⱼinvS = ∇S[jpar, iω] * invS
            for kpar = 1:jpar
                ∂SₖinvS = ∇S[kpar, iω] * invS
                ∇²ℓ[jpar, kpar] += real(tr( (otherIpart * ∂SⱼinvS) * ∂SₖinvS + Ipart * (∇²S[indexLT(jpar,kpar,size(∇S, 1)), iω] * invS - ∂SⱼinvS * ∂SₖinvS) ))
            end
        end
    end
    @inbounds for jpar = 1:size(∇²ℓ, 1)-1, kpar = jpar+1:size(∇²ℓ, 1) # fill upper triangle from symmetry
        @views ∇²ℓ[jpar, kpar] = ∇²ℓ[kpar, jpar]
    end
    return nothing
end
hess_generalwhittle!(∇²ℓ, funcmemory, gradmemory, hessmemory, data::GenWhittleData) = @views hess_generalwhittle!(∇²ℓ, unpack(funcmemory)[data.Ω_used_index], unpack(gradmemory)[:,data.Ω_used_index], unpack(hessmemory)[:,data.Ω_used_index], data.I)
hess_generalwhittle!(∇²ℓ, store::TimeSeriesModelStorageHessian, data::GenWhittleData) = hess_generalwhittle!(∇²ℓ, store.funcmemory, store.gradmemory, store.hessmemory, data)
hess_generalwhittle!(∇²ℓ, funcmemory, store::TimeSeriesModelStorageHessian, data::GenWhittleData) = hess_generalwhittle!(∇²ℓ, funcmemory, store.gradmemory, store.hessmemory, data)
function hess_generalwhittle!(∇²ℓ, objS, store::AdditiveStorage, data)
    @views hess_generalwhittle!(∇²ℓ[1:store.npar1,1:store.npar1], objS, store.store1, data)
    @views hess_generalwhittle!(∇²ℓ[store.npar1+1:end,store.npar1+1:end], objS, store.store2, data)
    @views offdiag_add_hess_generalwhittle!(∇²ℓ[1:store.npar1,store.npar1+1:end], objS, store.store1, store.store2, data)
    for i = store.npar1+1:size(∇²ℓ,2), j = 1:store.npar1
        @views ∇²ℓ[i,j] = ∇²ℓ[j,i]
    end
    return nothing
end
hess_generalwhittle!(∇ℓ, store::AdditiveStorage, data::GenWhittleData) = hess_generalwhittle!(∇ℓ, extract_S(store), store, data)

"Function to compute the offdiagonal parts of the hessian when the model is additive."
function offdiag_add_hess_generalwhittle!(∇²ℓ, S, ∇S1, ∇S2, I::Vector{T}) where {T<:Number} # univariate
    length(S) == length(I) == size(∇S1,2) == size(∇S2,2) || throw(ArgumentError("S and I should be same length as second dim of ∇S1 and ∇S2."))
    ∇²ℓ .= zero(eltype(∇²ℓ))
    @inbounds for iω = 1:length(S) # fill lower triangle
        invS = inv(S[iω])
        otherIpart = (I[iω] * invS)
        Ipart = (1 - otherIpart)
        for jpar = 1:size(∇S1, 1)
            ∂SⱼinvS = ∇S1[jpar, iω] * invS
            for kpar = 1:size(∇S2, 1)
                ∂SₖinvS = ∇S2[kpar, iω] * invS
                ∇²ℓ[jpar, kpar] += real( (otherIpart * ∂SⱼinvS)*∂SₖinvS - Ipart*(∂SⱼinvS * ∂SₖinvS) )
            end
        end
    end
    return nothing
end
function offdiag_add_hess_generalwhittle!(∇²ℓ, S, ∇S1, ∇S2, I::Vector{SMatrix{D,D,T,n}}) where {D,T,n}
    length(S) == length(I) == size(∇S1,2) == size(∇S2,2) || throw(ArgumentError("S and I should be same length as second dim of ∇S1 and ∇S2."))
    ∇²ℓ .= zero(eltype(∇²ℓ))
    Id = diagm(@SVector ones(D))
    for iω = 1:length(S) # fill lower triangle
        invS = inv(S[iω])
        otherIpart = (I[iω] * invS)
        Ipart = (Id - otherIpart)
        for jpar = 1:size(∇S1, 1)
            ∂SⱼinvS = ∇S1[jpar, iω] * invS
            for kpar = 1:size(∇S2, 1)
                ∂SₖinvS = ∇S2[kpar, iω] * invS
                ∇²ℓ[jpar, kpar] += real(tr( (otherIpart * ∂SⱼinvS)*∂SₖinvS - Ipart*(∂SⱼinvS * ∂SₖinvS) ))
            end
        end
    end
    return nothing
end
offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory, gradmemory1, gradmemory2, data::GenWhittleData) = @views offdiag_add_hess_generalwhittle!(∇²ℓ, unpack(funcmemory)[data.Ω_used_index], unpack(gradmemory1)[:,data.Ω_used_index], unpack(gradmemory2)[:,data.Ω_used_index], data.I)
offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory, store1::TimeSeriesModelStorageHessian, store2::TimeSeriesModelStorageHessian, data::GenWhittleData) = offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory, store1.gradmemory, store2.gradmemory, data)

function offdiag_add_hess_generalwhittle!(∇²ℓ, objS, store1::AdditiveStorage, store2::AdditiveStorage, data::GenWhittleData)
    @views offdiag_add_hess_generalwhittle!(∇²ℓ[1:store1.npar1,:], objS, store1.store1, store2, data)
    @views offdiag_add_hess_generalwhittle!(∇²ℓ[store1.npar1+1:end,:], objS, store1.store2, store2, data)
end

function offdiag_add_hess_generalwhittle!(∇²ℓ, objS, store1::AdditiveStorage, store2, data::GenWhittleData)
    @views offdiag_add_hess_generalwhittle!(∇²ℓ[1:store1.npar1,:], objS, store1.store1, store2, data)
    @views offdiag_add_hess_generalwhittle!(∇²ℓ[store1.npar1+1:end,:], objS, store1.store2, store2, data)
end

function offdiag_add_hess_generalwhittle!(∇²ℓ, objS, store1, store2::AdditiveStorage, data::GenWhittleData)
    @views offdiag_add_hess_generalwhittle!(∇²ℓ[:,1:store1.npar1], objS, store1, store2.store1, data)
    @views offdiag_add_hess_generalwhittle!(∇²ℓ[:,store1.npar1+1:end], objS, store1, store2.store2, data)
end

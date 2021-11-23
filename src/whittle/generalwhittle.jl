## Be careful, some of this functionality depends on the storage type provided. 
## Use higher level function calls for Whittle and de-biased Whitle if possible!

"Low level function to compute the Whittle summand given a spectral quantity S (the sdf or EI) and periodogram ordinate I."
generalwhittle_summand(S::SHermitianCompact, I) = real(log(det(S)) + tr(I*inv(S)))
generalwhittle_summand(S::Matrix{T}, I) where {T} = real(log(det(S)) + tr(I/S))
generalwhittle_summand(S::T, I) where {T<:Number} = real(log(S) + I/S)

"Function to compute the Whittle likelihood from the Periodogram and some spectral quantity (sdf or EI)."
generalwhittle(S, I::Vector{T}) where {T} = sum(generalwhittle_summand(Sᵢ,Iᵢ) for (Sᵢ,Iᵢ) ∈ zip(S,I))
generalwhittle(store::SdfStorage, data::GenWhittleData) = @views generalwhittle(store.hermitianarray[data.Ω_used_index], data.I)
generalwhittle(store::Acv2EIStorage, data::GenWhittleData) = @views generalwhittle(store.hermitianarray[data.Ω_used_index], data.I)
generalwhittle(store::Sdf2EIStorage, data::GenWhittleData) = generalwhittle(store.acv2EI, data)
generalwhittle(store::TimeSeriesModelStorageFunction, data::GenWhittleData) = generalwhittle(store.funcmemory, data)
generalwhittle(store::AdditiveStorage, data::GenWhittleData) = generalwhittle(store.store1, data) # function value will always be stored in leftmost storage.

## gradient
"Function to compute the gradient of the Whittle likelihood from the Periodogram and some spectral quantity (sdf or EI)."
function grad_generalwhittle!(∇ℓ, S, ∇S, I::Vector{T}) where {T<:Number} # univariate case
    ∇ℓ .= zero(eltype(∇ℓ))
    @inbounds for iω = 1:length(S)
        @views Ipart = (1.0/real(S[iω][1]) - I[iω]/(real(S[iω][1])^2))
        for jpar = 1:size(∇S, 1)
            @views ∇ℓ[jpar] += Ipart * real(∇S[jpar, iω][1])
        end
    end
    return nothing
end
function grad_generalwhittle!(∇ℓ, S, ∇S, I::Vector{SMatrix{D,D,T,n}}) where {D,T,n} # multivariate case
    ∇ℓ .= zero(eltype(∇ℓ))
    Id = diagm(@SVector ones(D))
    @inbounds for iω = 1:length(S)
        @views invS = inv(S[iω])
        @views Ipart = (Id - I[iω] * invS)
        for jpar = 1:size(∇S, 1)
            @views ∇ℓ[jpar] += real(tr(Ipart * ∇S[jpar, iω] * invS))
        end
    end
    return nothing
end
grad_generalwhittle!(∇ℓ, funcmemory::SdfStorage,    gradmemory::SdfStorage,    data::GenWhittleData) = @views grad_generalwhittle!(∇ℓ, funcmemory.hermitianarray[data.Ω_used_index], gradmemory.hermitianarray[:,data.Ω_used_index], data.I)
grad_generalwhittle!(∇ℓ, funcmemory::Acv2EIStorage, gradmemory::Acv2EIStorage, data::GenWhittleData) = @views grad_generalwhittle!(∇ℓ, funcmemory.hermitianarray[data.Ω_used_index], gradmemory.hermitianarray[:,data.Ω_used_index], data.I)
grad_generalwhittle!(∇ℓ, funcmemory::Sdf2EIStorage, gradmemory::Acv2EIStorage, data::GenWhittleData) = grad_generalwhittle!(∇ℓ, funcmemory.acv2EI, gradmemory,        data)
grad_generalwhittle!(∇ℓ, funcmemory,                gradmemory::Sdf2EIStorage, data::GenWhittleData) = grad_generalwhittle!(∇ℓ, funcmemory,        gradmemory.acv2EI, data)
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
    ∇²ℓ .= zero(eltype(∇²ℓ))
    @views for iω = 1:length(S) # fill lower triangle
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
    ∇²ℓ .= zero(eltype(∇²ℓ))
    Id = diagm(@SVector ones(D))
    @views for iω = 1:length(S) # fill lower triangle
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
    for jpar = 1:size(∇S, 1)-1, kpar = jpar+1:size(∇S, 1) # fill upper triangle from symmetry
        @views ∇²ℓ[jpar, kpar] = ∇²ℓ[kpar, jpar]
    end
    return nothing
end
hess_generalwhittle!(∇²ℓ, funcmemory::SdfStorage,    gradmemory::SdfStorage,    hessmemory::SdfStorage, data::GenWhittleData) = @views hess_generalwhittle!(∇²ℓ, funcmemory.hermitianarray[data.Ω_used_index], gradmemory.hermitianarray[:,data.Ω_used_index], hessmemory.hermitianarray[:,data.Ω_used_index], data.I)
hess_generalwhittle!(∇²ℓ, funcmemory::Acv2EIStorage, gradmemory::Acv2EIStorage, hessmemory::Acv2EIStorage, data::GenWhittleData) = @views hess_generalwhittle!(∇²ℓ, funcmemory.hermitianarray[data.Ω_used_index], gradmemory.hermitianarray[:,data.Ω_used_index], hessmemory.hermitianarray[:,data.Ω_used_index], data.I)
hess_generalwhittle!(∇²ℓ, funcmemory::Sdf2EIStorage, gradmemory::Acv2EIStorage, hessmemory::Acv2EIStorage, data::GenWhittleData) = hess_generalwhittle!(∇²ℓ, funcmemory.acv2EI, gradmemory, hessmemory, data)
hess_generalwhittle!(∇²ℓ, funcmemory               , gradmemory::Sdf2EIStorage, hessmemory::Sdf2EIStorage, data::GenWhittleData) = hess_generalwhittle!(∇²ℓ, funcmemory, gradmemory.acv2EI, hessmemory.acv2EI, data)
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
    ∇²ℓ .= zero(eltype(∇²ℓ))
    @views for iω = 1:length(S) # fill lower triangle
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
    ∇²ℓ .= zero(eltype(∇²ℓ))
    Id = diagm(@SVector ones(D))
    @views for iω = 1:length(S) # fill lower triangle
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
offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory::SdfStorage, gradmemory1::SdfStorage, gradmemory2::SdfStorage, data::GenWhittleData) = @views offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory.hermitianarray[data.Ω_used_index], gradmemory1.hermitianarray[:,data.Ω_used_index], gradmemory2.hermitianarray[:,data.Ω_used_index], data.I)
offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory::Acv2EIStorage, gradmemory1::Acv2EIStorage, gradmemory2::Acv2EIStorage, data::GenWhittleData) = @views offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory.hermitianarray[data.Ω_used_index], gradmemory1.hermitianarray[:,data.Ω_used_index], gradmemory2.hermitianarray[:,data.Ω_used_index], data.I)
offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory::Sdf2EIStorage, gradmemory1::Acv2EIStorage, gradmemory2::Acv2EIStorage, data::GenWhittleData) = offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory.acv2EI, gradmemory1, gradmemory2, data)
offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory               , gradmemory1::Sdf2EIStorage, gradmemory2::Acv2EIStorage, data::GenWhittleData) = offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory, gradmemory1.acv2EI, gradmemory2, data)
offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory               , gradmemory1               , gradmemory2::Sdf2EIStorage, data::GenWhittleData) = offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory, gradmemory1, gradmemory2.acv2EI, data)
offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory, store1::TimeSeriesModelStorageHessian, store2::TimeSeriesModelStorageHessian, data::GenWhittleData) = @views offdiag_add_hess_generalwhittle!(∇²ℓ, funcmemory, store1.gradmemory, store2.gradmemory, data)

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

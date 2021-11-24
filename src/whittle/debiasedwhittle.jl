"Functor which generates appropriate memory for the debiased Whittle likelihood."
struct DebiasedWhittleLikelihood{T,S<:TimeSeriesModelStorage,M}
    data::DebiasedWhittleData{T}
    memory::S
    model::M
    function DebiasedWhittleLikelihood(
        model::Type{<:TimeSeriesModel{D}}, ts, Δ;
        lowerΩcutoff = 0, upperΩcutoff = Inf) where {D}

        D == size(ts,2) || error("timeseries is $(size(ts,2)) dimensional, but model is $D dimensional.")
        wdata = DebiasedWhittleData(model, ts, Δ, lowerΩcutoff = lowerΩcutoff, upperΩcutoff = upperΩcutoff)
        mem = allocate_memory_EI_FG(model, size(ts,1), Δ)
        new{eltype(wdata.I),typeof(mem),typeof(model)}(wdata,mem,model)
    end
end
function (f::DebiasedWhittleLikelihood)(θ)
    debiasedwhittle!(f.model,f.data,f.memory,θ)
end
function (f::DebiasedWhittleLikelihood)(F,G,EH,θ)
    debiasedwhittle_Fisher(f.model,F,G,EH,f.data,f.memory,θ)
end
Base.show(io::IO, W::DebiasedWhittleLikelihood) = print(io, "Debiased Whittle likelihood for the $(W.model) model.")

## Interior functions 

##
"Function to compute the debiased Whittle likelihood using a preallocated store."
function debiasedwhittle!(store::TimeSeriesModelStorage, model::TimeSeriesModel, data::GenWhittleData)
    EI!(store, model)
    return generalwhittle(store, data)
end

"""
Function to compute the debiased Whittle likelihood and its gradient using a preallocated store.
First argument is the model (a type). F and G are then provided to signify which quantities are required.
This is in keeping with Optims interface.
"""
function debiasedwhittle_FG!!(F, G, store, model::TimeSeriesModel ,data::GenWhittleData)
    if F !== nothing || G !== nothing
        EI!(store, model)
    end
    if G !== nothing
        grad_EI!(store, model)
        grad_generalwhittle!(G, store, data)
    end
    if F !== nothing
        return generalwhittle(store, data)
    end
    return nothing
end

"""
Function to compute the debiased Whittle likelihood and its gradient and hessian using a preallocated store.
First argument is the model (a type). F, G and H are then provided to signify which quantities are required.
This is in keeping with Optims interface.
"""
function debiasedwhittle_FGH(F, G, H, store, model::TimeSeriesModel, data::GenWhittleData)
    if F !== nothing || G !== nothing || H !== nothing
        EI!(store, model)
    end
    if G !== nothing || H !== nothing
        grad_EI!(store, model)
    end
    if H !== nothing
        hess_EI!(store, model)
        hess_generalwhittle!(H, store, data)
    end
    if G !== nothing
        grad_generalwhittle!(G, store, data)
    end
    if F !== nothing
        return generalwhittle(store, data)
    end
    return nothing
end

"Function to extract all of the derivatives from a generic store."
function getallderiv(store::AdditiveStorage)
    Vcat(getallderiv(store.store1), getallderiv(store.store2))
end
getallderiv(store::TimeSeriesModelStorageGradient) = extract_array(store.gradmemory)

"Function to extract array from storage."
extract_array(store::Sdf2EIStorage) = extract_array(store.acv2EI)
extract_array(store::Acv2EIStorage) = store.hermitianarray

"Function to compute the expected hessian of the de-biased Whittle likelihood."
function debiasedwhittle_Ehess!(EH, store, data::GenWhittleData)
    EH .= zero(eltype(EH))
    ∇S = getallderiv(store)
    S = extract_array(extract_S(store))
    for iω = data.Ω_used_index
        @views invS = inv(S[iω]) # account for double frequency resolution
        for jpar = 1:size(EH, 1), kpar = 1:jpar
            @views EH[jpar,kpar] += real(tr(∇S[jpar, iω] * invS*∇S[kpar, iω] * invS))
        end
    end
    for jpar = 1:size(EH, 1)-1, kpar = jpar+1:size(EH, 1) # fill upper triangle from symmetry
        @views EH[jpar, kpar] = EH[kpar, jpar]
    end
    return nothing
end

"Function to compute the debiased Whittle likelihood, its gradient and fisher information."
function debiasedwhittle_Fisher(F, G, H, store, model::TimeSeriesModel, data::GenWhittleData)
    if F !== nothing || G !== nothing || H !== nothing
        EI!(store, model)
    end
    if G !== nothing || H !== nothing
        grad_EI!(store, model)
    end
    if H !== nothing
        debiasedwhittle_Ehess!(H, store, data)
    end
    if G !== nothing
        grad_generalwhittle!(G, store, data)
    end
    if F !== nothing
        return generalwhittle(store, data)
    end
    return nothing
end
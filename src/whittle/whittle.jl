"""
    WhittleLikelihood(model::Type{<:TimeSeriesModel}, ts, Δ; lowerΩcutoff, upperΩcutoff)

Generate a function to evaluate the Whittle likelihood and it's first and second derivatives.

Create a callable struct which prealloctes memory appropriately.
Aruguments are:
- `model`: the model for the process.
- `ts`: the timeseries in the form of an n by d matrix (where d is the dimension of the time series model).
- `Δ`: the sampling rate of the time series.
- `lowerΩcutoff`: the lower bound of the frequency range included in the likelihood.
- `upperΩcutoff`: the upper bound of the frequency range included in the likelihood.
"""
struct WhittleLikelihood{T,S<:TimeSeriesModelStorage,M}
    data::WhittleData{T}
    memory::S
    model::M
    function WhittleLikelihood(
        model::Type{<:TimeSeriesModel{D}}, ts, Δ;
        lowerΩcutoff = 0, upperΩcutoff = Inf) where {D}

        D == size(ts,2) || error("timeseries is $(size(ts,2)) dimensional, but model is $D dimensional.")
        wdata = WhittleData(model, ts, Δ, lowerΩcutoff = lowerΩcutoff, upperΩcutoff = upperΩcutoff)
        mem = allocate_memory_sdf_FGH(model, size(ts,1), Δ)
        new{eltype(wdata.I),typeof(mem),typeof(model)}(wdata,mem,model)
    end
end
function (f::WhittleLikelihood)(θ)
    whittle!(f.memory,f.model(θ),f.data)
end
function (f::WhittleLikelihood)(F,G,H,θ)
    whittle_FGH!(F,G,H,f.memory,f.model(θ),f.data)
end
Base.show(io::IO, W::WhittleLikelihood) = print(io, "Whittle likelihood for the $(W.model) model.")

## internal functions

"Function to compute the Whittle likelihood using a preallocated store."
function whittle!(store, model::TimeSeriesModel, data::GenWhittleData)
    asdf!(store, model)
    return @views generalwhittle(store, data)
end

"""
Function to compute the Whittle likelihood and its gradient using a preallocated store.
First argument is the model (a type). F and G are then provided to signify which quantities are required.
This is in keeping with Optims interface.
"""
function whittle_FG!(F, G, store, model::TimeSeriesModel, data::GenWhittleData)
    if F !== nothing || G !== nothing
        asdf!(model, store)
    end
    if G !== nothing
        grad_asdf!(model, store)
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
function whittle_FGH!(F, G, H, store, model::TimeSeriesModel, data::GenWhittleData)
    if F !== nothing || G !== nothing || H !== nothing
        asdf!(store, model)
    end
    if G !== nothing || H !== nothing
        grad_asdf!(store, model)
    end
    if H !== nothing
        hess_asdf!(store, model)
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
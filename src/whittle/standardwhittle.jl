"""
    WhittleLikelihood(model::Type{<:TimeSeriesModel}, ts, Δ; lowerΩcutoff, upperΩcutoff)

Generate a function to evaluate the Whittle likelihood it's gradient and Hessian.

Create a callable struct which prealloctes memory appropriately.

    (f::WhittleLikelihood)(θ)

Evaluates the Whittle likelihood at θ.

    (f::WhittleLikelihood)(F,G,H,θ)

Evaluates the Whittle likelihood at θ and stores the gradient and Hessian in G and H respectively.
If F, G or H equal nothing, then the function, gradient or Hessian are not evaluated repsectively.

# Aruguments
- `model`: the model for the process. Should be of type TimeSeriesModel, so `OU` and not `OU(1,1)`.
- `ts`: the timeseries in the form of an n by d matrix (where d is the dimension of the time series model).
- `Δ`: the sampling rate of the time series.
- `lowerΩcutoff`: the lower bound of the frequency range included in the likelihood.
- `upperΩcutoff`: the upper bound of the frequency range included in the likelihood.

Note that to use the gradient the model must have `grad_add_sdf!` specified.
Similarly, to use the Hessian, the model must have `hess_add_sdf!` specified.

# Examples
```julia-repl
julia> obj = WhittleLikelihood(OU, ones(1000), 1)
Whittle likelihood for the OU model.

julia> obj([1.0, 1.0])
-2006.7870804551364

julia> F, G, H = 0.0, zeros(2), zeros(2,2)
(0.0, [0.0, 0.0], [0.0 0.0; 0.0 0.0])

julia> obj(F, G, H, [1.0, 1.0])
-2006.7870804551364

julia> G
2-element Vector{Float64}:
   2.777354965282642
 -17.45063591068618

julia> H
2×2 Matrix{Float64}:
 -0.00967179   0.0607696
  0.0607696   -0.381827
```

"""
struct WhittleLikelihood{T,S<:TimeSeriesModelStorage,M}
    data::WhittleData{T}
    memory::S
    model::M
    function WhittleLikelihood(
        model::Type{<:TimeSeriesModel{D}}, ts, Δ;
        lowerΩcutoff = 0, upperΩcutoff = Inf, taper = nothing) where {D}
        
        Δ > 0 || throw(ArgumentError("Δ should be a positive."))
        D == size(ts,2) || throw(ArgumentError("timeseries is $(size(ts,2)) dimensional, but model is $D dimensional."))
        wdata = WhittleData(model, ts, Δ, lowerΩcutoff = lowerΩcutoff, upperΩcutoff = upperΩcutoff, taper = taper)
        mem = allocate_memory_sdf_FGH(model, size(ts,1), Δ)
        new{eltype(wdata.I),typeof(mem),typeof(model)}(wdata,mem,model)
    end
end
(f::WhittleLikelihood)(θ) = whittle!(f.memory,f.model(θ),f.data)
(f::WhittleLikelihood)(F,G,H,θ) = whittle_FGH!(F,G,H,f.memory,f.model(θ),f.data)

Base.show(io::IO, W::WhittleLikelihood) = print(io, "Whittle likelihood for the $(W.model) model.")

## internal functions

"""
    whittle!(store, model::TimeSeriesModel, data::GenWhittleData)

Compute the Whittle likelihood using a preallocated store.
"""
function whittle!(store, model::TimeSeriesModel, data::GenWhittleData)
    asdf!(store, model)
    return @views generalwhittle(store, data)
end

"""
    whittle_FG!(F, G, store, model::TimeSeriesModel, data::GenWhittleData)

Compute the Whittle likelihood and its gradient using a preallocated store.
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
    whittle_FGH!(F, G, H, store, model::TimeSeriesModel, data::GenWhittleData)

Compute the debiased Whittle likelihood and its gradient and hessian using a preallocated store.
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
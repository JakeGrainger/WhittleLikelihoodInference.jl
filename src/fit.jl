"""
    fit(ts,Δ;model::Type{<:TimeSeriesModel},x₀,lowerΩcutoff,upperΩcutoff,x_lowerbounds,x_upperbounds,method,taper)
    fit(timeseries::TimeSeries;model::Type{<:TimeSeriesModel},x₀,lowerΩcutoff,upperΩcutoff,x_lowerbounds,x_upperbounds,method,taper)

Fit a time series model using the `IPNewton` method from Optim.jl.

# Arguments
- `ts`: `n` by `D` matrix containing the timeseries (or vector if `D=1`), where `n` is the number of observations and `D` is the number of series.
- `Δ`: The sampling rate, which should be a positive real number.
- `timeseries`: Can be provided in place of `ts` and `Δ`.
- `model`: The model which will be fitted. Should be a type (not a realisation of the model) e.g. JONSWAP{k} not JONSWAP{K}(x).
- `x₀`: The initial parameter guess.
- `lowerΩcutoff`: The lower cutoff for the frequency range to be used in fitting. Default is `0`.
- `upperΩcutoff`: The upper cutoff for the frequency range to be used in fitting. Default is `Inf`.
- `x_lowerbounds`: The lower bounds on the parameter space. If `nothing` is provided (the default) then these are set to default values based on the model.
- `x_upperbounds`: The upper bounds on the parameter space. If `nothing` is provided (the default) then these are set to default values based on the model.
- `method`: Either `:Whittle` or `:debiasedWhittle`.
- `taper`: The choice of tapering to be used. This should be `nothing` (in which case no taper is used) or `dpss_nw` where `nw` time-bandwith product (see DSP.dpss for more details).
- `options`: Options for optimisation of type `Optim.Options`.
"""
function fit(ts::VecOrMat{Float64},Δ::Real;model::Type{<:TimeSeriesModel},x₀,lowerΩcutoff::Real=0.0,upperΩcutoff::Real=Inf,
            x_lowerbounds=nothing,x_upperbounds=nothing,method=:debiasedWhittle,taper=nothing,options::Optim.Options=Optim.Options(iterations=1000))
    Δ > 0 || throw(ArgumentError("Δ should be a positive number."))
    model(x₀) # to check that the model can be constructed from the initial vector provided.
    size(ts,2) == ndims(model) || throw(ArgumentError("Dimension of the TimeSeries ($(size(ts,2))) does not match the dimension of the TimeSeriesModel ($(ndims(model)))."))
    lowerΩcutoff < upperΩcutoff || throw(ArgumentError("lower frequency cutoff should be below upper frequency cutoff."))
    if x_lowerbounds === nothing
        x_lowerbounds = lowerbounds(model)
    else
        length(x_lowerbounds) == npars(model) || throw(ArgumentError("provided vector of lower bounds is not the same length as the number of parameters"))
    end
    if x_upperbounds === nothing
        x_upperbounds = upperbounds(model)
    else
        length(x_upperbounds) == npars(model) || throw(ArgumentError("provided vector of upper bounds is not the same length as the number of parameters"))
    end
    if method == :Whittle
        objective = WhittleLikelihood(model, ts, Δ; lowerΩcutoff = lowerΩcutoff, upperΩcutoff = upperΩcutoff, taper = maketaper(taper,size(ts,1)))
    elseif method == :debiasedWhittle
        objective = DebiasedWhittleLikelihood(model, ts, Δ; lowerΩcutoff = lowerΩcutoff, upperΩcutoff = upperΩcutoff, taper = maketaper(taper,size(ts,1)))
    else
        throw(ArgumentError("Method not recognised. Choose from :Whittle or :debiasedWhittle"))
    end
    constraints = Optim.TwiceDifferentiableConstraints(x_lowerbounds,x_upperbounds)
    obj = TwiceDifferentiable(Optim.only_fgh!(objective),x₀)
    res = optimize(obj, constraints, x₀, IPNewton(), options)
    Optim.converged(res) || @warn "Optimisation did not converge!"
    return res
end
function fit(ts::TimeSeries;model::Type{<:TimeSeriesModel},x₀,lowerΩcutoff::Real=0.0,upperΩcutoff::Real=Inf,
            x_lowerbounds=nothing,x_upperbounds=nothing,method=:debiasedWhittle,taper=nothing,options::Optim.Options=Optim.Options(iterations=1000))
    fit(ts.ts,ts.Δ,model=model,x₀=x₀,lowerΩcutoff=lowerΩcutoff,upperΩcutoff=upperΩcutoff,
    x_lowerbounds=x_lowerbounds,x_upperbounds=x_upperbounds,method=method,taper=taper,options=options)
end
maketaper(taper::Nothing,n) = nothing
maketaper(taper::Symbol,n) = maketaper(String(taper),n)
function maketaper(taper::String,n)
    tapername, nw = split(taper,'_')
    tapername == "dpss" || throw(ArgumentError("Taper should be nothing or dpss_nw where nw is the value of nw used for the taper. Other kinds of tapering are not yet supported."))
    return vec(dpss(n, parse(Float64,nw), 1))
end
lowerbounds(::Type{<:AdditiveTimeSeriesModel{M₁,M₂,D,T}}) where {M₁,M₂,D,T} = vcat(lowerbounds(M₁),lowerbounds(M₂))
upperbounds(::Type{<:AdditiveTimeSeriesModel{M₁,M₂,D,T}}) where {M₁,M₂,D,T} = vcat(upperbounds(M₁),upperbounds(M₂))
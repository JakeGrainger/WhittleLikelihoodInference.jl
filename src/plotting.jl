struct HermitianPlot{A,B}
    x::A
    y::B
    function HermitianPlot(x::AbstractVector{T},y::AbstractVector{R}) where {R<:AbstractMatrix{S}} where {S<:Number,T<:Number}
        new{typeof(x),typeof(y)}(x,y)
    end
end
@recipe function f(h::HermitianPlot)
    x,y = h.x,h.y
    y_new = Vector{Matrix{Float64}}(undef, length(y))
    for i = 1:length(y)
        y_temp = zeros(size(y[i]))
        for j = 1:size(y[i],1), k = 1:size(y[i],2)
            y_temp[j,k] = j >= k ? real(y[i][j,k]) : imag(y[i][j,k])
        end
        y_new[i] = y_temp
    end
    MatrixPlot(x,y_new)
end

struct MatrixPlot{A,B}
    x::A
    y::B
    function MatrixPlot(x::AbstractVector{T},y::AbstractVector{R}) where {R<:AbstractMatrix{S}} where {S<:Number,T<:Number}
        all(size(yi)==size(y[1]) for yi in y) || error("Matricies given to MatrixPlot are not all the same size.")
        new{typeof(x),typeof(y)}(x,y)
    end
end
@recipe function f(m::MatrixPlot)
    x,y = m.x,m.y
    layout --> size(y[1])
    link --> :all
    label --> false
    count = 1
    for i = 1:size(y[1],1), j = 1:size(y[1],2)
        @series begin
            subplot --> count
            count += 1
            seriestype := :line
            x, getindex.(y,i,j)
        end
    end
end

@userplot PlotSdf
@recipe function f(p::PlotSdf)
    if length(p.args) == 1
        p.args[1] isa TimeSeriesModel || throw(ArgumentError("If only one argument is provided to plotsdf, it must be a TimeSeriesModel."))
        Ω = range(-π,π,length=200)
    elseif length(p.args) == 2
        p.args[1] isa TimeSeriesModel || throw(ArgumentError("First argument to plotsdf must be a TimeSeriesModel."))
        p.args[2] isa AbstractVector{T} where {T<:Number} || throw(ArgumentError("Second argument to plotasdf should be the frequencies, a vector of numbers."))
        Ω = p.args[2]
    else
        throw(ArgumentError("A maximum of two numbers should be passed to plotsdf."))
    end
    model = p.args[1]
    y = sdf.(model, Ω)
    if ndims(model) > 1
        return HermitianPlot(Ω, y)
    else
        return Ω, real.(y)
    end
end

"""
    plotsdf(model)
    plotsdf!(model)
    plotsdf(model, Ω)
    plotsdf!(model, Ω)

Plot the spectral density function of a model at the frequencies `Ω`. 
By default, `Ω = range(-π,π,length=200)` and `Δ = 1`.
"""
plotsdf


"""
    checkposreal(x)

Checks if x is a positive real number."""
checkposreal(x) = x isa Real && x > 0


@userplot PlotAsdf
@recipe function f(p::PlotAsdf)
    if length(p.args) == 1
        p.args[1] isa TimeSeriesModel || throw(ArgumentError("If only one argument is provided to plotasdf, it must be a TimeSeriesModel."))
        Ω = range(-π,π,length=200)
        Δ = 1
    elseif length(p.args) == 3
        p.args[1] isa TimeSeriesModel || throw(ArgumentError("First argument to plotasdf must be a TimeSeriesModel."))
        p.args[2] isa AbstractVector{T} where {T<:Number} || throw(ArgumentError("Second argument to plotasdf should be the frequencies, a vector of numbers."))
        checkposreal(p.args[3]) || throw(ArgumentError("Third argument to plotasdf should be the sampling rate, a positive real number."))
        Ω = p.args[2]
        Δ = p.args[3]
    else
        throw(ArgumentError("Either only a model should be provided to plotasdf or a model, vector of frequencies and sampling rate Δ."))
    end
    model = p.args[1]
    y = asdf.(model, Ω, Δ)
    if ndims(model) > 1
        return HermitianPlot(Ω, y)
    else
        return Ω, real.(y)
    end
end

"""
    plotasdf(model)
    plotasdf!(model)
    plotasdf(model, Ω, Δ)
    plotasdf!(model, Ω, Δ)

Plot the aliased spectral density function of a model at the frequencies Ω. 
By default, `Ω = range(-π,π,length=200)` and `Δ = 1`.
"""
plotasdf

@userplot PlotAcv
@recipe function f(p::PlotAcv)
    if length(p.args) == 1
        p.args[1] isa TimeSeriesModel || throw(ArgumentError("If only one argument is provided to plotacv, it must be a TimeSeriesModel."))
        n = 30
        Δ = 1
        τ = (-n+1:n-1).*Δ
    elseif length(p.args) == 2
        p.args[1] isa TimeSeriesModel || throw(ArgumentError("First argument to plotsdf must be a TimeSeriesModel."))
        p.args[2] isa AbstractVector{T} where {T<:Number} || throw(ArgumentError("If two arguments provided to the second must be a vector of time lags."))
        !(p.args[1] isa UnknownAcvTimeSeriesModel) || throw(ArgumentError("Custom lag vector only possible if acv of model is known, pass the number of observations and sampling rate instead."))
        τ = p.args[2]
    elseif length(p.args) == 3
        p.args[1] isa TimeSeriesModel || throw(ArgumentError("First argument to plotsdf must be a TimeSeriesModel."))
        checkposreal(p.args[2]) || throw(ArgumentError("Second argument to plotasdf should be the number of lags, a positive real number."))
        checkposreal(p.args[3]) || throw(ArgumentError("Third argument to plotasdf should be the sampling rate, a positive real number."))
        n = p.args[2]
        Δ = p.args[3]
        τ = (-n+1:n-1).*Δ
    else
        throw(ArgumentError("Either only a model should be provided to plotacv or a model, number of observations and sampling rate Δ (or vector of lags if appropriate)."))
    end
    model = p.args[1]
    y = length(p.args)==2 ? acv(model, τ) : acv(model, n, Δ)
    if ndims(model) > 1
        return MatrixPlot(τ, y)
    else
        return τ, real.(y)
    end
end

"""
    plotacv(model)
    plotacv!(model)
    plotacv(model, τ)
    plotacv!(model, τ)
    plotacv(model, n, Δ)
    plotacv!(model, n, Δ)

Plot the aliased spectral density function of a model at lags `τ`.
If the model does not have known acv, then the number of lags `n` and sampling rate `Δ` should be provided.
In this case, the lags are `-(n-1)*Δ:Δ:(n-1)*Δ`.
If unprovided, `n = 30` and `Δ = 1`.
"""
plotacv


@userplot PlotEI
@recipe function f(p::PlotEI)
    if length(p.args) == 1
        p.args[1] isa TimeSeriesModel || throw(ArgumentError("If only one argument is provided to plotei, it must be a TimeSeriesModel."))
        n = 1024
        Δ = 1
    elseif length(p.args) == 3
        p.args[1] isa TimeSeriesModel || throw(ArgumentError("First argument to plotei must be a TimeSeriesModel."))
        checkposreal(p.args[2]) || throw(ArgumentError("Second argument to plotei should be the number of observations, a positive real number."))
        checkposreal(p.args[3]) || throw(ArgumentError("Third argument to plotei should be the sampling rate, a positive real number."))
        n = p.args[2]
        Δ = p.args[3]
    else
        throw(ArgumentError("Either only a model should be provided to plotei or a model, number of observations and sampling rate Δ."))
    end
    Ω = fftshift(fftfreq(n, 2π/Δ))
    model = p.args[1]
    y = EI(model, n, Δ)
    if ndims(model) > 1
        return HermitianPlot(Ω, y)
    else
        return Ω, real.(y)
    end
end

"""
    plotei(model)
    plotei!(model)
    plotei(model, n, Δ)
    plotei!(model, n, Δ)

Plot the aliased spectral density function of a model at the angular Fourier frequencies `fftfreq(n,2π/Δ)`. 
By default, `n = 1024` and `Δ = 1`.
"""
plotei
@recipe function f(::Type{Val{:hermitianmatrixfunctionplot}},x,y,z)
    y isa AbstractVector{AbstractMatrix{T}} where {T<:Number} || error("When using hermitianmatrixfunctionplot, y should be a vector of matricies of numbers.")
    y_new = Vector{Matrix{Float64}}(undef, length(y))
    for i = 1:length(y)
        y_temp = zeros(size(y[i]))
        for j = 1:size(y[i],1), k = 1:size(y[i],2)
            y_temp[j,k] = j >= k ? real(y[i][j,k]) : imag(y[i][j,k])
        end
        y_new[i] = y_temp
    end
    seriestype := :realmatrixfunctionplot
    x := x
    y := y_new
end

"""
    hermitianmatrixfunctionplot(x,y)
    hermitianmatrixfunctionplot!(x,y)

Plots the spectral density matrix function over frequency with the lower triangle being the real part and upper triangle being the imaginary.
"""
@shorthand hermitianmatrixfunctionplot

@recipe function f(::Type{Val{:realmatrixfunctionplot}},x,y,z)
    all(size(yi)==size(y[1]) for yi in y) || error("Matricies not the same size")
    layout --> size(y[1])
    link --> :all
    label --> false
    count = 1
    for i = 1:size(y[1],1), j = 1:size(y[1],2)
        @series begin
            subplot --> count
            count += 1
            seriestype := :line
            x := x
            y := getindex.(y,i,j)
        end
    end
end
"""
    realmatrixfunctionplot(x,y)
    realmatrixfunctionplot!(x,y)

Plots a matrix valued function by plotting each element of the matrix in a separate panel.
"""
@shorthand realmatrixfunctionplot


@userplot PlotSdf
@recipe function f(p::PlotSdf)
    if length(p.args) == 1
        p.args[1] isa TimeSeriesModel || error("If only one argument is provided to plotsdf, it must be a TimeSeriesModel.")
        Ω = range(-π,π,length=200)
    elseif length(p.args) == 2
        p.args[1] isa TimeSeriesModel || error("First argument to plotsdf must be a TimeSeriesModel.")
        p.args[2] isa AbstractVector{T} where {T<:Number} || error("Second argument to plotasdf should be the frequencies, a vector of numbers.")
        Ω = p.args[2]
    else
        error("A maximum of two numbers should be passed to plotsdf.")
    end
    model = p.args[1]
    y = sdf(model, Ω)
    if ndims(model) > 1
        seriestype := :hermitianmatrixfunctionplot
        return Ω, y
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
        p.args[1] isa TimeSeriesModel || error("If only one argument is provided to plotasdf, it must be a TimeSeriesModel.")
        Ω = range(-π,π,length=200)
        Δ = 1
    elseif length(p.args) == 3
        p.args[1] isa TimeSeriesModel || error("First argument to plotasdf must be a TimeSeriesModel.")
        p.args[2] isa AbstractVector{T} where {T<:Number} || error("Second argument to plotasdf should be the frequencies, a vector of numbers.")
        checkposreal(p.args[3]) || error("Third argument to plotasdf should be the sampling rate, a positive real number.")
        Ω = p.args[2]
        Δ = p.args[3]
    else
        error("Either only a model should be provided to plotasdf or a model, vector of frequencies and sampling rate Δ.")
    end
    model = p.args[1]
    y = asdf(model, Ω, Δ)
    if ndims(model) > 1
        seriestype := :hermitianmatrixfunctionplot
        return Ω, y
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
        p.args[1] isa TimeSeriesModel || error("If only one argument is provided to plotacv, it must be a TimeSeriesModel.")
        n = 30
        Δ = 1
        τ = (-n+1:n-1).*Δ
    elseif length(p.args) == 2
        p.args[1] isa TimeSeriesModel || error("First argument to plotsdf must be a TimeSeriesModel.")
        p.args[2] isa AbstractVector{T} where {T<:Number} || error("If two arguments provided to the second must be a vector of time lags.")
        !(p.args[1] isa UnknownAcvTimeSeriesModel) || error("Custom lag vector only possible if acv of model is known, pass the number of observations and sampling rate instead.")
        τ = p.args[2]
    elseif length(p.args) == 3
        p.args[1] isa TimeSeriesModel || error("First argument to plotsdf must be a TimeSeriesModel.")
        checkposreal(p.args[2]) || error("Second argument to plotasdf should be the number of lags, a positive real number.")
        checkposreal(p.args[3]) || error("Third argument to plotasdf should be the sampling rate, a positive real number.")
        n = p.args[2]
        Δ = p.args[3]
        τ = (-n+1:n-1).*Δ
    else
        error("Either only a model should be provided to plotacv or a model, number of observations and sampling rate Δ (or vector of lags if appropriate).")
    end
    model = p.args[1]
    y = length(p.args==2) ? acv(model, τ) : acv(model, n, Δ)
    if ndims(model) > 1
        seriestype := :realmatrixfunctionplot
        return τ, y
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
        p.args[1] isa TimeSeriesModel || error("If only one argument is provided to plotei, it must be a TimeSeriesModel.")
        n = 1024
        Δ = 1
    elseif length(p.args) == 3
        p.args[1] isa TimeSeriesModel || error("First argument to plotei must be a TimeSeriesModel.")
        checkposreal(p.args[2]) || error("Second argument to plotei should be the number of observations, a positive real number.")
        checkposreal(p.args[3]) || error("Third argument to plotei should be the sampling rate, a positive real number.")
        Ω = p.args[2]
        Δ = p.args[3]
    else
        error("Either only a model should be provided to plotei or a model, number of observations and sampling rate Δ.")
    end
    Ω = fftshift(fftfreq(n, 2π/Δ))
    model = p.args[1]
    y = EI(model, n, Δ)
    if ndims(model) > 1
        seriestype := :hermitianmatrixfunctionplot
        return Ω, y
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
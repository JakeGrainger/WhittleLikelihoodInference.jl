function covmatrix(model::TimeSeriesModel, n, Δ)
    d = ndims(model)
    store = allocate_memory_EI_F(typeof(model), n , Δ)
    acv!(store, model)
    Areal = real.(extract_acv(store))
    C = Matrix{eltype(Areal)}(undef, n * d, n * d)
    for ii = 1:d # increase row
        for jj = 1:d # increase col
            # compute cross correlation
            C[(1:n).+(ii-1)*n, (1:n).+(jj-1)*n] =
                ii > jj ? # if in bottom triangle
                Toeplitz(
                    fftshift(Areal[indexLT(ii,jj,d), :])[n+1:end], # positive lags down
                    fftshift(Areal[indexLT(ii,jj,d), :])[n+1:-1:2], # negative lags across
                ) :
                ii < jj ? # if in top triangle
                Toeplitz(
                    fftshift(Areal[indexLT(ii,jj,d), :])[n+1:-1:2], # negative lags down
                    fftshift(Areal[indexLT(ii,jj,d), :])[n+1:end], # positive lags across
                ) : # if on diagonal
                Toeplitz(Areal[indexLT(ii,jj,d), 1:n], Areal[indexLT(ii,jj,d), 1:n]) # negative and positive lags equal
        end
    end
    if isposdef(C)
        return C
    else
        @warn "Covariance matrix not positive definite, adding a small number to diagonals to try and fix."
        for i ∈ 1:size(C,1)
            C[i,i] += 1e-10
        end
        if isposdef(C)
            return C
        else
            error("C not positive semi definite.")
        end
    end
end

function covmatrix(model::TimeSeriesModel{1}, n, Δ)
    store = allocate_memory_EI_F(typeof(model), n , Δ)
    acv!(store, model)
    Areal = real.(extract_acv(store))
    return Matrix(Toeplitz(Areal[1:n], Areal[1:n]))
end
struct GaussianProcess{T}
    X::T
    Δ::Float64
    function GaussianProcess(model::TimeSeriesModel, n::Int, Δ::Real)
        Δ > 0 || throw(ArgumentError("Δ should be positive."))
        n > 0 || throw(ArgumentError("n should be positive."))
        C = covmatrix(model, n, Δ)
        X = normaldist(model,C)
        new{typeof(X)}(X,Δ)
    end
end
normaldist(::TimeSeriesModel{D}, C) where {D} = reshape(MvNormal(C), size(C,1)÷D, D)
normaldist(::TimeSeriesModel{1}, C) = MvNormal(C)

struct TimeSeries{T,N}
    ts::Array{T,N}
    Δ::Float64
end
Base.ndims( ts::TimeSeries) = size(ts.ts, 2)
Base.length(ts::TimeSeries) = size(ts.ts, 1)

Distributions.rand(gp::GaussianProcess) = TimeSeries(rand(gp.X),gp.Δ)

@recipe function f(ts::TimeSeries)
    0:ts.Δ:ts.Δ*(length(ts)-1), ts.ts
end

function simulate_gp(model::TimeSeriesModel, n::Int, Δ::Number, nreps::Int)
    nreps > 0 || throw(ArgumentError("nreps should be positive."))
    GP = GaussianProcess(model, n , Δ)
    return [rand(GP) for _ in 1:nreps]
end
function covmatrix(model::TimeSeriesModel, n, Δ)
    d = ndims(model)
    store = allocate_memory_EI_F(typeof(model), n , Δ)
    acv!(typeof(model), store, parameter(model))
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
        @warn "Covariance matrix not positive definite, adding a small number to diagonals to fix."
        for i ∈ 1:size(C,1)
            C[i,i] += 1e-10
        end
        if isposdef(C)
            return C
        else
            for i ∈ 1:size(C,1)
                C[i,i] += 1e-10
            end
        end
        return C
    end
end

function covmatrix(model::TimeSeriesModel{1}, n, Δ)
    store = allocate_memory_EI_F(typeof(model), n , Δ)
    acv!(typeof(model), store, parameter(model))
    Areal = real.(extract_acv(store))
    return Matrix(Toeplitz(Areal[1:n], Areal[1:n]))
end
function FiniteNormal(model::TimeSeriesModel{1}, n, Δ)
    C = covmatrix(model, n, Δ)
    X = MvNormal(C)
    return X
end
function FiniteNormal(model::TimeSeriesModel{D}, n, Δ) where {D}
    C = covmatrix(model, n, Δ)
    X = MatrixReshaped(MvNormal(C), n, D)
    return X
end
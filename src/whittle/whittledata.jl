# General structure for storing timeseries for Whittle type computation.
abstract type GenWhittleData end

"""
    WhittleData(model::Type{<:TimeSeriesModel{D}}, timeseries, Δ; lowerΩcutoff, upperΩcutoff)

Create storage for timeseries data in a format which is useful for Whittle methods.
"""
struct WhittleData{T} <: GenWhittleData
    I::Vector{T}
    N::Int64
    Δ::Float64
    Ω_used_index::Vector{Int64}
    function WhittleData(::Type{<:TimeSeriesModel{D}}, timeseries::Matrix{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf) where {D,S<:Real}
        size(timeseries, 2) == D || error("Time series has $(size(timeseries, 2)) dimensions, but model has $D dimension.")
        N = size(timeseries, 1)
        Ω = fftfreq(N, 2π/Δ)
        Ω_used_index = (1:N)[lowerΩcutoff .< abs.(Ω) .< upperΩcutoff]
        J = fft(timeseries, 1)
        I = [SMatrix{D,D}(J[ii, :]*J[ii, :]'.*(Δ / (2π * N))) for ii = Ω_used_index]
        new{eltype(I)}(I, N, Δ, Ω_used_index)
    end
    function WhittleData(::Type{<:TimeSeriesModel{1}}, timeseries::Vector{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf) where {S<:Real}
        N = size(timeseries, 1)
        Ω = fftfreq(N, 2π/Δ)
        Ω_used_index = (1:N)[lowerΩcutoff .< abs.(Ω) .< upperΩcutoff]
        J = fft(timeseries, 1)
        I = abs.(J).^2 .* (Δ / (2π * N))
        new{eltype(I)}(I[Ω_used_index], N, Δ, Ω_used_index)
    end
end

"""
    DebiasedWhittleData(model::Type{<:TimeSeriesModel{D}}, timeseries, Δ; lowerΩcutoff, upperΩcutoff)

Create storage for timeseries data in a format which is useful for debiased Whittle methods.
"""
struct DebiasedWhittleData{T} <: GenWhittleData
    I::Vector{T}
    N::Int64
    Δ::Float64
    Ω_used_index::Vector{Int64}
    function DebiasedWhittleData(::Type{<:TimeSeriesModel{D}}, timeseries::Matrix{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf) where {D,S<:Real}
        size(timeseries, 2) == D || error("Time series has $(size(timeseries, 2)) dimensions, but model has $D dimension.")
        N = size(timeseries, 1)
        Ω = fftfreq(N, 2π/Δ)
        Ω_used_index = (1:N)[lowerΩcutoff .< abs.(Ω) .< upperΩcutoff]
        J = fft(timeseries, 1)
        I = [SMatrix{D,D}(J[ii, :]*J[ii, :]'.*(Δ / (2π * N))) for ii = Ω_used_index]
        new{eltype(I)}(I, N, Δ, 2 .* Ω_used_index .- 1) # accounts for double resolution in EI computations
    end
    function DebiasedWhittleData(::Type{<:TimeSeriesModel{1}}, timeseries::Vector{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf) where {S<:Real}
        N = size(timeseries, 1)
        Ω = fftfreq(N, 2π/Δ)
        Ω_used_index = (1:N)[lowerΩcutoff .< abs.(Ω) .< upperΩcutoff]
        J = fft(timeseries, 1)
        I = abs.(J).^2 .* (Δ / (2π * N))
        new{eltype(I)}(I[Ω_used_index], N, Δ, 2 .* Ω_used_index .- 1) # accounts for double resolution in EI computations
    end
end
Base.show(io::IO, W::GenWhittleData) = print(io, "Precomputed periodogram for $(size(W.I[1],1)) dimensional timeseries of length $(W.N) with a sampling rate of $(W.Δ).")

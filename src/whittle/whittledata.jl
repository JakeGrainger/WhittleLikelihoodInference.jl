# General structure for storing timeseries for Whittle type computation.
abstract type GenWhittleData end

"""
    WhittleData(model::Type{<:TimeSeriesModel{D}}, timeseries, Δ; lowerΩcutoff, upperΩcutoff)

Create storage for timeseries data in a format which is useful for Whittle methods.
"""
struct WhittleData{T} <: GenWhittleData
    I::Vector{T}
    n::Int64
    Δ::Float64
    Ω_used_index::Vector{Int64}
    function WhittleData(::Type{<:TimeSeriesModel{D}}, timeseries::Matrix{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf, taper = nothing) where {D,S<:Real}
        checkwhittledatainputs(timeseries,taper,Val{D}())
        ordinates,n,Ω_used_index = makewhittledata(timeseries, Δ, Val{D}())
        new{eltype(ordinates)}(ordinates, n, Δ, Ω_used_index)
    end
end

"""
    DebiasedWhittleData(model::Type{<:TimeSeriesModel{D}}, timeseries, Δ; lowerΩcutoff, upperΩcutoff)

Create storage for timeseries data in a format which is useful for debiased Whittle methods.
"""
struct DebiasedWhittleData{T} <: GenWhittleData
    I::Vector{T}
    n::Int64
    Δ::Float64
    Ω_used_index::Vector{Int64}
    function DebiasedWhittleData(::Type{<:TimeSeriesModel{D}}, timeseries::Vector{S}, Δ; lowerΩcutoff = 0, upperΩcutoff = Inf, taper = nothing) where {S<:Real,D}
        checkwhittledatainputs(timeseries,taper,Val{D}())
        ordinates,n,Ω_used_index = makewhittledata(timeseries, Δ, Val{D}())
        new{eltype(I)}(ordinates, n, Δ, 2 .* Ω_used_index .- 1) # accounts for double resolution in EI computations
    end
end
Base.show(io::IO, W::GenWhittleData) = print(io, "Precomputed periodogram for $(size(W.I[1],1)) dimensional timeseries of length $(W.n) with a sampling rate of $(W.Δ).")

## internal for whittle data
function checkwhittledatainputs(timeseries,taper,::Val{D}) where {D}
    size(timeseries, 2) == D || throw(ArgumentError("Time series has $(size(timeseries, 2)) dimensions, but model has $D dimension."))
    if !(taper === nothing)
        taper isa AbstractVector{T} where {T<:Real} || throw(ArgumentError("Taper should be an AbstractVector of reals."))
        length(taper) == size(timeseries,1) || throw(ArgumentError("taper should be same length as timeseries"))
    end
    nothing
end

function makewhittledata(timeseries, Δ, lowerΩcutoff, upperΩcutoff)
    n = size(timeseries, 1)
    Ω = fftfreq(n, 2π/Δ)
    Ω_used_index = (1:n)[lowerΩcutoff .< abs.(Ω) .< upperΩcutoff]
    return makeI(timeseries, Δ)[Ω_used_index], n, Ω_used_index
end
function makeI(timeseries::Matrix{T}, Δ, ::Val{D}) where {T,D}
    J = fft(timeseries, 1)
    return [SMatrix{D,D}(j*j'.*(Δ / (2π * n))) for j in eachrow(J)]
end
function makeI(timeseries::Vector{T}, Δ, ::Val{1}) where {T}
    J = fft(timeseries, 1)
    return abs.(J).^2 .* (Δ / (2π * n))
end
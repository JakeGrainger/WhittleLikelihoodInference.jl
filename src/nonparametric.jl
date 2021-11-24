abstract type SpectralEstimate{D,T} <: AbstractVector{T} end


struct Periodogram{D,T,V} <: SpectralEstimate{D,T}
    Ω::V
    ordinates::Vector{T}
    function Periodogram(timeseries::Matrix{T}, Δ::Real) where {T}
        J = fftshift(fft(timeseries, 1),1)
        n = size(J,1)
        D = size(J,2)
        ordinates = [(J[ii, :]*J[ii, :]'.*(Δ / (2π * n))) for ii ∈ 1:size(J,1)]
        Ω = fftshift(fftfreq(n, 2π/Δ))
        new{D, eltype(ordinates), typeof(Ω)}(Ω, ordinates)
    end
    function Periodogram(timeseries::Vector{T}, Δ::Real) where {T}
        J = fftshift(fft(timeseries, 1),1)
        n = size(J,1)
        ordinates = abs.(J.^2).*(Δ / (2π * n))
        Ω = fftshift(fftfreq(n, 2π/Δ))
        new{1, eltype(ordinates), typeof(Ω)}(Ω, ordinates)
    end
    function Periodogram(Ω, ordinates::Vector{T}) where {T}
        new{size(ordinates[1],1), T, typeof(Ω)}(Ω, ordinates)
    end
end

ndims(::SpectralEstimate{D,T}) where {D,T} = D
size(ŝ::SpectralEstimate) = (length(getfield(ŝ,1)),)
getindex(ŝ::T, inds) where {T<:SpectralEstimate} = T(getfield(ŝ,1)[inds], getfield(ŝ,2)[inds])
getindex(ŝ::SpectralEstimate, ind::Int) = (getfield(ŝ,1)[ind],getfield(ŝ,2)[ind])
log10(ŝ::T) where {T<:SpectralEstimate} = T(getfield(ŝ,1), log10.(getfield(ŝ,2)))


@recipe function f(ŝ::SpectralEstimate)
    layout := (ndims(ŝ), ndims(ŝ))
    label --> false
    link --> :all
    count = 1
    for i ∈ 1:ndims(ŝ), j ∈ 1:ndims(ŝ)
        @series begin
            subplot := count
            count += 1
            y = [getfield(ŝ,2)[k][i,j] for k ∈ 1:length(ŝ)]
            z = (i >= j) ? real.(y) : imag(y)
            getfield(ŝ,1), z
        end
    end
end


function Base.:+(ŝ₁::T, ŝ₂::T) where {T<:SpectralEstimate}
    getfield(ŝ₁,1) == getfield(ŝ₂,1) || error("Frequencies must be the same to add spectral estimates.")
    T(getfield(ŝ₁,1), getfield(ŝ₁,2) .+ getfield(ŝ₂,2))
end
function /(ŝ::T, a::S) where {T<:SpectralEstimate, S<:Real}
    T(getfield(ŝ,1), getfield(ŝ,2)./a)
end


struct BartlettPeriodogram{D,T,V} <: SpectralEstimate{D,T}
    Ω::V
    ordinates::Vector{T}
    function BartlettPeriodogram(timeseries::Matrix{T}, Δ::Real, segmentlength::Int = Int(100÷Δ)) where {T}
        nsegments = size(timeseries, 1) ÷ segmentlength
        P = Periodogram(timeseries[1:segmentlength, :], Δ)
        
        for ii = 1:nsegments-1
            P += Periodogram(timeseries[segmentlength*ii.+(1:segmentlength), :], Δ)
        end
        
        BartlettPeriodogram(P/nsegments)
    end
    function BartlettPeriodogram(timeseries::Vector{T}, Δ::Real, segmentlength::Int = Int(100÷Δ)) where {T}
        nsegments = size(timeseries, 1) ÷ segmentlength
        P = Periodogram(timeseries[1:segmentlength], Δ)
        
        for ii = 1:nsegments-1
            P += Periodogram(timeseries[segmentlength*ii.+(1:segmentlength)], Δ)
        end
        
        BartlettPeriodogram(P/nsegments)
    end
    function BartlettPeriodogram(p::Periodogram{D,T}) where {D,T}
        new{D, T, typeof(p.Ω)}(p.Ω, p.ordinates)
    end
    function BartlettPeriodogram(Ω, ordinates::Vector{T}) where {T}
        new{size(ordinates[1],1), T, typeof(Ω)}(Ω, ordinates)
    end
end

struct CoherancyEstimate{D,T,V} <: SpectralEstimate{D,T}
    Ω::V
    ordinates::Vector{T}

    function CoherancyEstimate(timeseries::Matrix{T}, Δ::Real, segmentlength::Int = Int(100÷Δ)) where {T}
        S = BartlettPeriodogram(timeseries, Δ, segmentlength)
        CoherancyEstimate(S)
    end

    function CoherancyEstimate(S::SpectralEstimate{D,T}) where {D,T}
        newordinates = [[s[i,j] / sqrt(s[i,i]*s[j,j]) for i ∈ 1:D, j ∈ 1:D] for s ∈ S.ordinates]
        new{D,eltype(newordinates),typeof(S.Ω)}(S.Ω, newordinates)
    end

    function CoherancyEstimate(Ω, ordinates::Vector{T}) where {T}
        new{size(ordinates[1],1), T, typeof(Ω)}(Ω, ordinates)
    end
end
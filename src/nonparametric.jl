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
size(ŝ::SpectralEstimate) = (length(getfreq(ŝ)),)
getindex(ŝ::T, inds) where {T<:SpectralEstimate} = T(getfreq(ŝ)[inds], getordinate(ŝ)[inds])
getindex(ŝ::SpectralEstimate, ind::Int) = (getfreq(ŝ)[ind],getordinate(ŝ)[ind])
log10(ŝ::T) where {T<:SpectralEstimate} = T(getfreq(ŝ), log10.(getordinate(ŝ)))
getfreq(p::Periodogram) = p.Ω
getfreq(b::BartlettPeriodogram) = b.Ω
getfreq(c::CoherancyEstimate) = c.Ω
getordinate(p::Periodogram) = p.ordinates
getordinate(b::BartlettPeriodogram) = b.ordinates
getordinate(c::CoherancyEstimate) = c.ordinates


@recipe function f(ŝ::SpectralEstimate)
    layout := (ndims(ŝ), ndims(ŝ))
    label --> false
    link --> :all
    count = 1
    for i ∈ 1:ndims(ŝ), j ∈ 1:ndims(ŝ)
        @series begin
            subplot := count
            count += 1
            y = [getordinate(ŝ)[k][i,j] for k ∈ 1:length(ŝ)]
            z = (i >= j) ? real.(y) : imag(y)
            MatrixPlot(getfreq(ŝ), z)
        end
    end
end


function Base.:+(ŝ₁::T, ŝ₂::T) where {T<:SpectralEstimate}
    getfreq(ŝ₁) == getfreq(ŝ₂) || error("Frequencies must be the same to add spectral estimates.")
    T(getfreq(ŝ₁), getordinate(ŝ₁) .+ getordinate(ŝ₂))
end
function Base.:/(ŝ::T, a::Real) where {T<:SpectralEstimate}
    T(getfreq(ŝ), getordinate(ŝ)./a)
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
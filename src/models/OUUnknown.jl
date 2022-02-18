struct OUUnknown{K} <: UnknownAcvTimeSeriesModel{1}
    σ::Float64
    θ::Float64
    σ²::Float64
    θ²::Float64
    function OUUnknown{K}(σ,θ) where {K}
        σ > 0 || throw(ArgumentError("OU process requires 0 < σ."))
        θ > 0 || throw(ArgumentError("OU process requires 0 < θ."))
        new(σ,θ,σ^2,θ^2)
    end
    function OUUnknown{K}(x::Vector{Float64}) where {K}
        @boundscheck checkparameterlength(x,OUUnknown)
        @inbounds OUUnknown{K}(x[1], x[2])
    end
end
nalias(::OUUnknown{K}) where {K} = K
npars(::Type{OUUnknown{K}}) where {K} = 2

sdf(m::OUUnknown, ω) = m.σ²/(π*(m.θ²+ω^2))

function grad_add_sdf!(out, m::OUUnknown, ω)
    σ, θ = m.σ, m.θ
    σ_part = 2σ / (π*(m.θ²+ω^2))
    out[1] = σ_part
    out[2] = -π*θ/2*(σ_part)^2
    nothing
end

function hess_add_sdf!(out, m::OUUnknown, ω)
    σ, θ = m.σ, m.θ
    σ_part = 2 / (π*(m.θ²+ω^2))
    θσ_part = -π*θ*σ*(σ_part)^2
    out[1] = σ_part
    out[2] = θσ_part
    out[3] = -π*σ_part*σ*(σ_part*σ/2 + θ*θσ_part)
    nothing
end

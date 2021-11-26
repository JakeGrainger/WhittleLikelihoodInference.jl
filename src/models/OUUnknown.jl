struct OUUnknown{K} <: UnknownAcvTimeSeriesModel{1}
    σ::Float64
    θ::Float64
    σ²::Float64
    θ²::Float64
    OUUnknown{K}(σ,θ) where {K} = new{K}(σ,θ,σ^2,θ^2)
    function OUUnknown{K}(x::Vector{Float64}) where {K}
        length(x) == npars(OUUnknown) || error("OUUnknown process has $(npars(OUUnknown)) parameters, but $(length(x)) were provided.")
        @inbounds OUUnknown{K}(x[1], x[2])
    end
end
nalias(::OUUnknown{K}) where {K} = K
npars(::Type{OUUnknown}) = 2

sdf(m::OUUnknown, ω) = m.σ²/(π*(m.θ²+ω^2))

function grad_add_sdf!(out, m::OUUnknown, ω)
    σ, θ = m.σ, m.θ
    σ_part = σ / (π*(m.θ²+ω^2))
    out[1] = σ_part
    out[2] = -σ_part*2π*θ
    nothing
end

function hess_add_sdf!(out, m::OUUnknown, ω)
    σ, θ = m.σ, m.θ
    σ_part = 1 / (π*(m.θ²+ω^2))
    partθσ = -σ_part*2π*θ
    out[1] = σ_part
    out[2] = partθσ
    out[3] = -2π * (θ*partθσ+σ*σ_part)
    nothing
end
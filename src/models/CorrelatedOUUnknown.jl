struct CorrelatedOUUknown{K} <: UnknownAcvTimeSeriesModel{2}
    σ::Float64
    θ::Float64
    ρ::Float64
    σ²::Float64
    θ²::Float64
    CorrelatedOUUknown{K}(σ,θ,ρ) where {K} = new{K}(σ,θ,ρ,σ^2,θ^2)
    function CorrelatedOUUknown{K}(x::Vector{Float64}) where {K}
        length(x) == npars(CorrelatedOUUknown) || error("CorrelatedOUUknown process has $(npars(CorrelatedOUUknown)) parameters, but $(length(x)) were provided.")
        @inbounds CorrelatedOUUknown{K}(x[1], x[2], x[3])
    end
end
nalias(::CorrelatedOUUknown{K}) where {K} = K
npars(::Type{CorrelatedOUUknown}) = 3

function add_sdf!(out, m::CorrelatedOUUknown, ω)
    s = m.σ²/(π*(m.θ²+ω^2))
    out[1] += s
    out[2] += s*m.ρ
    out[3] += s
end

function grad_add_sdf!(out, m::CorrelatedOUUknown, ω)
    σ, θ, ρ = m.σ, m.θ, m.ρ
    σ_part = 2σ / (π*(m.θ²+ω^2))
    # ∂σ
    out[1,1] = σ_part
    out[2,1] = σ_part * ρ
    out[3,1] = σ_part
    # ∂θ
    θ_part = -π*θ/2*(σ_part)^2
    out[1,2] = θ_part
    out[2,2] = θ_part * ρ
    out[3,2] = θ_part
    # ∂ρ
    # out[1,3] = 0
    out[2,3] = σ_part * σ / 2
    # out[3,3] = 0
    nothing
end

function hess_add_sdf!(out, m::CorrelatedOUUknown, ω)
    σ, θ, ρ = m.σ, m.θ, m.ρ
    σ_part = 2 / (π*(m.θ²+ω^2))
    θσ_part = -π*θ*σ*(σ_part)^2
    θ_part = -π*σ_part*σ*(σ_part*σ/2 + θ*θσ_part)

    # ∂σ²
    out[1,1] = σ_part
    out[2,1] = σ_part * ρ
    out[3,1] = σ_part

    # ∂σθ
    out[1,2] = θσ_part
    out[2,2] = θσ_part * ρ
    out[3,2] = θσ_part

    #∂σρ
    # out[1,3] = 0
    out[2,3] = σ_part * σ
    # out[3,3] = 0

    # ∂θ²
    out[1,4] = θ_part
    out[2,4] = θ_part * ρ
    out[3,4] = θ_part

    #∂θρ
    # out[1,5] = 0
    out[2,5] = σ/2 * θσ_part
    # out[3,5] = 0

    #∂ρ² all zero

    nothing
end

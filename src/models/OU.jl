struct OU <: TimeSeriesModel{1}
    σ::Float64
    θ::Float64
    σ²::Float64
    θ²::Float64
    function OU(σ,θ)
        σ > 0 || throw(ArgumentError("OU process requires 0 < σ."))
        θ > 0 || throw(ArgumentError("OU process requires 0 < θ."))
        new(σ,θ,σ^2,θ^2)
    end
function OU(x::AbstractVector{Float64})
        length(x) == npars(OU) || throw(ArgumentError("OU process has $(npars(OU)) parameters, but $(length(x)) were provided."))
        @inbounds OU(x[1], x[2])
    end
end

npars(::Type{OU}) = 2

sdf(m::OU, ω) = m.σ²/(π*(m.θ²+ω^2))

acv(m::OU, τ::Number) = exp(-m.θ*abs(τ)) * m.σ² / m.θ

function grad_add_sdf!(out, m::OU, ω)
    σ, θ = m.σ, m.θ
    σ_part = 2σ / (π*(m.θ²+ω^2))
    out[1] = σ_part
    out[2] = -π*θ/2*(σ_part)^2
    nothing
end

function hess_add_sdf!(out, m::OU, ω)
    σ, θ = m.σ, m.θ
    σ_part = 2 / (π*(m.θ²+ω^2))
    θσ_part = -π*θ*σ*(σ_part)^2
    out[1] = σ_part
    out[2] = θσ_part
    out[3] = -π*σ_part*σ*(σ_part*σ/2 + θ*θσ_part)
    nothing
end

function grad_acv!(out, m::OU, τ::Number)
    σ, θ = m.σ, m.θ
    absτ = abs(τ)
    part = 2exp(-θ*absτ)*σ/θ
    out[1] = part
    out[2] = -part*σ*(1/θ+absτ)/2
    nothing
end

function hess_acv!(out, m::OU, τ::Number)
    σ, θ = m.σ, m.θ
    absτ = abs(τ)
    part = 2exp(-θ*absτ)/θ
    out[1] = part
    out[2] = -part*σ*(1/θ+absτ)
    out[3] = part*(m.σ²)*(1/(m.θ²) + absτ/θ + (absτ^2)/2)
    nothing
end
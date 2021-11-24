struct OU <: TimeSeriesModel{1}
    σ::Float64
    θ::Float64
    σ²::Float64
    θ²::Float64
    OU(σ,θ) = new(σ,θ,σ^2,θ^2)
    function OU(x::Vector{Float64})
        length(x) == 2 || error("Incorrect number of parameters for OU process")
        @inbounds OU(x[1], x[2])
    end
end

npars(::Type{OU}) = 2
parameternames(::Type{OU}) = ("σ", "θ")

sdf(m::OU, ω) = m.σ²/((2π)*(m.θ²+ω^2))

acv(m::OU, τ) = exp(-m.θ*abs(τ)) * m.σ² / (2m.θ)

function grad_add_acv!(out, m::OU, τ)
    σ, θ = m.σ, m.θ
    absτ = abs(τ)
    part = exp(-θ*absτ)*σ/θ
    out[1] = part
    out[2] = -part*σ*(1/θ+absτ)/2
    nothing
end

function hess_add_acv!(::Type{OU}, out, τ, θ)
    σ, θ = m.σ, m.θ
    absτ = abs(τ)
    part = exp(-θ*absτ)/θ
    out[1] = part
    out[2] = -part*σ*(1/θ+absτ)
    out[3] = part*(m.σ²)*(1/(m.θ²) + absτ/θ + (absτ^2)/2)
    nothing
end
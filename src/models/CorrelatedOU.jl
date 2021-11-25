struct CorrelatedOU <: TimeSeriesModel{2}
    σ::Float64
    θ::Float64
    ρ::Float64
    σ²::Float64
    θ²::Float64
    CorrelatedOU(σ,θ,ρ) = new(σ,θ,ρ,σ^2,θ^2)
    function CorrelatedOU(x::Float64)
        length(x) == npars(CorrelatedOU) || error("CorrelatedOU process has $(npars(CorrelatedOU)) parameters, but $(length(x)) were provided.")
        @inbounds CorrelatedOU(x[1], x[2], x[3])
    end
end

npars(::Type{CorrelatedOU}) = 3

function add_sdf!(out, m::CorrelatedOU, ω)
    s = 2m.σ²/((2π)*(m.θ²+ω^2))
    out[1] += s
    out[2] += s*m.ρ
    out[3] += s
end
function acv!(out, m::CorrelatedOU, τ) 
    a = exp(-m.θ*abs(τ)) * m.σ² / m.θ
    out[1] = a
    out[2] = a*m.ρ
    out[3] = a
end
function grad_acv!(out, m::CorrelatedOU, τ)
    σ, θ, ρ = m.σ, m.θ, m.ρ
    absτ = abs(τ)
    σ_part = 2exp(-θ*absτ)*σ/θ
    # ∂σ
    out[1,1] = σ_part
    out[2,1] = σ_part * ρ
    out[3,1] = σ_part
    # ∂θ
    θ_part = -σ_part*σ*(1/θ+absτ)/2
    out[1,2] = θ_part
    out[2,2] = θ_part * ρ
    out[3,2] = θ_part
    # ∂ρ
    # out[1,2] = 0
    out[2,2] = σ_part * σ/2
    # out[3,2] = 0
    nothing
end

function hess_acv!(::Type{CorrelatedOU}, out, τ, θ)
    σ, θ = m.σ, m.θ
    absτ = abs(τ)
    σ_part = 2exp(-θ*absτ)/θ
    θσ_part = -σ_part*σ*(1/θ+absτ)
    θ_part = σ_part*(m.σ²)*(1/(m.θ²) + absτ/θ + (absτ^2)/2)

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
    out[2,5] = -σ_part*m.σ²*(1/θ+absτ)/2
    # out[3,5] = 0

    #∂ρ² all zero

    nothing
end
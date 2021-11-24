struct OU <: TimeSeriesModel{1}
    θ::Vector{Float64}
end

npars(::Type{OU}) = 2
parameternames(::Type{OU}) = ["σ", "θ"]

sdf(::Type{OU}, ω, θ) = θ[1]^2/((2π)*(θ[2]^2+ω^2))

acv(::Type{OU}, τ, θ) = exp(-θ[2]*abs(τ)) * θ[1]^2 / (2θ[2])

function grad_add_acv!(::Type{OU}, out, τ, θ)
    absτ = abs(τ)
    part = exp(-θ[2]*absτ)*θ[1]/θ[2]
    out[1] = part
    out[2] = -part*θ[1]*(1/θ[2]+absτ)/2
    nothing
end

function hess_add_acv!(::Type{OU}, out, τ, θ)
    absτ = abs(τ)
    part = exp(-θ[2]*absτ)/θ[2]
    out[1] = part
    out[2] = -part*θ[1]*(1/θ[2]+absτ)
    out[3] = part*(θ[1]^2)*(1/(θ[2]^2) + absτ/θ[2] + (absτ^2)/2)
    nothing
end
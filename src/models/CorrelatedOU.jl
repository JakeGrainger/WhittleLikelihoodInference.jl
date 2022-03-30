struct CorrelatedOU <: TimeSeriesModel{2,Float64}
    σ::Float64
    θ::Float64
    ρ::Float64
    σ²::Float64
    θ²::Float64
    function CorrelatedOU(σ,θ,ρ)
        σ > 0 || throw(ArgumentError("OU process requires 0 < σ."))
        θ > 0 || throw(ArgumentError("OU process requires 0 < θ."))
        1>ρ>0 || throw(ArgumentError("OU process requires 0 < ρ < 1."))
        new(σ,θ,ρ,σ^2,θ^2)
    end
function CorrelatedOU(x::AbstractVector{Float64})
        @boundscheck checkparameterlength(x,CorrelatedOU)
        @inbounds CorrelatedOU(x[1], x[2], x[3])
    end
end

npars(::Type{CorrelatedOU}) = 3

@propagate_inbounds function add_sdf!(out, m::CorrelatedOU, ω)
    @boundscheck checkbounds(out,1:3)
    @inbounds begin
        s = m.σ²/(π*(m.θ²+ω^2))
        out[1] += s
        out[2] += s*m.ρ
        out[3] += s
    end
end
@propagate_inbounds function acv!(out, m::CorrelatedOU, τ::Number)
    @boundscheck checkbounds(out,1:3)
    @inbounds begin
        a = exp(-m.θ*abs(τ)) * m.σ² / m.θ
        out[1] = a
        out[2] = a*m.ρ
        out[3] = a
    end
end

@propagate_inbounds function grad_add_sdf!(out, m::CorrelatedOU, ω)
    @boundscheck checkbounds(out,1:3,1:3)
    @inbounds begin
        σ, θ, ρ = m.σ, m.θ, m.ρ
        σ_part = 2σ / (π*(m.θ²+ω^2))
        # ∂σ
        out[1,1] += σ_part
        out[2,1] += σ_part * ρ
        out[3,1] += σ_part
        # ∂θ
        θ_part = -π*θ/2*(σ_part)^2
        out[1,2] += θ_part
        out[2,2] += θ_part * ρ
        out[3,2] += θ_part
        # ∂ρ
        # out[1,3] += 0
        out[2,3] += σ_part * σ / 2
        # out[3,3] += 0
    end
    nothing
end

@propagate_inbounds function hess_add_sdf!(out, m::CorrelatedOU, ω)
    @boundscheck checkbounds(out,1:3,1:5)
    @inbounds begin
        σ, θ, ρ = m.σ, m.θ, m.ρ
        σ_part = 2 / (π*(m.θ²+ω^2))
        θσ_part = -π*θ*σ*(σ_part)^2
        θ_part = -π*σ_part*σ*(σ_part*σ/2 + θ*θσ_part)

        # ∂σ²
        out[1,1] += σ_part
        out[2,1] += σ_part * ρ
        out[3,1] += σ_part

        # ∂σθ
        out[1,2] += θσ_part
        out[2,2] += θσ_part * ρ
        out[3,2] += θσ_part

        #∂σρ
        # out[1,3] += 0
        out[2,3] += σ_part * σ
        # out[3,3] += 0

        # ∂θ²
        out[1,4] += θ_part
        out[2,4] += θ_part * ρ
        out[3,4] += θ_part

        #∂θρ
        # out[1,5] += 0
        out[2,5] += σ/2 * θσ_part
        # out[3,5] += 0

        #∂ρ² all zero
    end

    nothing
end

@propagate_inbounds function grad_acv!(out, m::CorrelatedOU, τ::Number)
    @boundscheck checkbounds(out,1:3,1:3)
    @inbounds begin
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
        # out[1,3] = 0
        out[2,3] = σ_part * σ/2
        # out[3,3] = 0
    end
    nothing
end

@propagate_inbounds function hess_acv!(out, m::CorrelatedOU, τ::Number)
    @boundscheck checkbounds(out,1:3,1:5)
    @inbounds begin
        σ, θ, ρ = m.σ, m.θ, m.ρ
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
    end
    nothing
end
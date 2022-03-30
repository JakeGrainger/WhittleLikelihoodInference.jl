function ChainRulesCore.rrule(ℓ::Union{DebiasedWhittleLikelihood,WhittleLikelihood},θ)
    G = zeros(Float64, npars(getmodel(ℓ)))
    y = ℓ(zero(Float64),G,nothing,θ)
    function whittlelikelihood_pullback(Δy)
        return NoTangent(), G * Δy
    end
    return y, whittlelikelihood_pullback
end





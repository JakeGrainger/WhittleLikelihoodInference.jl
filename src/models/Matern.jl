struct Matern{D} <: TimeSeriesModel{D}
    θ
    function Matern{D}(θ) where {D}
        @assert length(θ) == npars(Matern{D})
        new{D}(θ)
    end
end

npars(::Type{Matern{D}}) where {D} = 3triangularnumber(D)
nalias(::Type{Matern{D}}) where {D} = 5
function parameternames(::Type{Matern{D}}) where {D}
    σ = reduce(vcat,[i==j ? "σ_$i" : "ρ_$i$j" for j in i:D] for i in 1:D)
    ν = reduce(vcat,["ν_$i$j" for j in i:D] for i in 1:D)
    a = reduce(vcat,["a_$i$j" for j in i:D] for i in 1:D)
    return [σ;ν;a]
end

# """
#     parameter(model::Matern)

# Return the parameter vector of a Matern model.

# In the D=3 case, θ = [σ₁,ρ₂₁,ρ₃₁,σ₂,ρ₂₃,σ₃,ν₁,ν₂₁,ν₃₁,ν₂,ν₂₃,ν₃,a₁,a₂₁,a₃₁,a₂,a₂₃,a₃].
# We have 3*D(D+1)/2 parameters.
# - σᵢ is the standard deviation of the ith process.
# - ρᵢⱼ is the colocated correlation coefficient between the ith and jth processes.
# - νᵢⱼ is the smoothness parameter between the ith and jth process.
# - aᵢⱼ is the scale parameter between the ith and jth process.
# """
# parameter(model::Matern) = model.θ

function add_sdf!(::Type{Matern{D}}, out, ω, θ) where {D}
    ndims_lt = triangularnumber(D)
    ndims_lt_p1 = 1+ndims_lt
    twice_ndims_lt = 2ndims_lt
    count = 1
    for i ∈ 1:D-1
        out[count] += θ[count]^2 * matern_corr_ft(ω,θ[count+ndims_lt],θ[count+twice_ndims_lt])
        ind_i = ndims_lt_p1-triangularnumber(i)
        count += 1
        for j ∈ i+1:D
            ind_j = ndims_lt_p1-triangularnumber(j)
            out[count] += θ[ind_i] * θ[ind_j] * θ[count] * matern_corr_ft(ω,θ[count+ndims_lt],θ[count+twice_ndims_lt])
            # i.e. outᵢⱼ = σᵢ     * σⱼ       * ρᵢⱼ      * f     (ω|νᵢⱼ,              aᵢⱼ)
            count += 1
        end
    end
    # final case (i=D)
    out[count] += θ[count]^2 * matern_corr_ft(ω,θ[count+ndims_lt],θ[count+twice_ndims_lt])
    return nothing
end

function acv!(::Type{Matern{D}}, out, τ::Number, θ) where {D}
    ndims_lt = triangularnumber(D)
    ndims_lt_p1 = 1+ndims_lt
    twice_ndims_lt = 2ndims_lt
    count = 1
    for i ∈ 1:D-1
        out[count] = θ[count]^2 * matern_corr(τ,θ[count+ndims_lt],θ[count+twice_ndims_lt])
        ind_i = ndims_lt_p1-triangularnumber(i)
        count += 1
        for j ∈ i+1:D
            ind_j = ndims_lt_p1-triangularnumber(j)
            out[count] = θ[ind_i] * θ[ind_j] * θ[count] * matern_corr(τ,θ[count+ndims_lt],θ[count+twice_ndims_lt])
            # i.e. outᵢⱼ = σᵢ     * σⱼ       * ρᵢⱼ      * M     (τ|νᵢⱼ,              aᵢⱼ)
            count += 1
        end
    end
    out[count] += θ[count]^2 * matern_corr(τ,θ[count+ndims_lt],θ[count+twice_ndims_lt])
    return nothing
end

##
function matern_corr(τ,ν,a)
    amodτ = a*abs(τ)
    return abs(τ)>1e-10 ? (2^(1-ν) / gamma(ν)) * (amodτ)^ν * besselk(ν,amodτ) : 1
end

matern_corr_ft(ω,ν,a) = gamma(ν+0.5)*a^(2ν) / (gamma(ν)*sqrt(π)*(a^2+ω^2)^(ν+0.5))
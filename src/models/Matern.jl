struct Matern{D,L} <: UnknownAcvTimeSeriesModel{D}
    σ::SHermitianCompact{D,Float64,L}
    ν::SHermitianCompact{D,Float64,L}
    a::SHermitianCompact{D,Float64,L}
    σ²::SHermitianCompact{D,Float64,L}
    a²::SHermitianCompact{D,Float64,L}
    νplushalf::SHermitianCompact{D,Float64,L}
    sdfconst::SHermitianCompact{D,Float64,L}
    ∂ν_part::SHermitianCompact{D,Float64,L}
    ∂a_part1::SHermitianCompact{D,Float64,L}
    ∂a_part2::SHermitianCompact{D,Float64,L}
    ∂ν²_part::SHermitianCompact{D,Float64,L}
    ∂ν∂a_part::SHermitianCompact{D,Float64,L}
    ∂a²_part::SHermitianCompact{D,Float64,L}
    function Matern{D,L}(θ) where {D,L}
        length(θ) == npars(Matern{D,L}) || throw(ArgumentError("Matern process has $(npars(Matern{D,L})) parameters, but $(length(θ)) were provided."))
        L == triangularnumber(D) || error("Matern{D,L} should satisfy L == D*(D+1)÷2")
        all(x->x>0, θ) || throw(ArgumentError("all parameters of Matern should be > 0."))
        σ = @views SHermitianCompact(SVector{L,Float64}(θ[1:L]))
        all(i==j ? true : σ[i,j] < 1 for i in 1:size(σ,1) for j in 1:i) || throw(ArgumentError("ρ parameters must be < 1."))
        ν = @views SHermitianCompact(SVector{L,Float64}(θ[L+1:2L]))
        a = @views SHermitianCompact(SVector{L,Float64}(θ[2L+1:end]))
        σ² = SHermitianCompact((σ.lowertriangle).^2)
        a² = SHermitianCompact((a.lowertriangle).^2)
        νplushalf = SHermitianCompact((ν.lowertriangle).+0.5)
        variance_part = SHermitianCompact(SMatrix{D,D,Float64}(i==j ? σ[i,i]^2 : σ[i,i]*σ[j,j]*σ[i,j] for i in 1:D, j in 1:D))
        sdfconst = SHermitianCompact(matern_sdf_normalising.(ν.lowertriangle,a.lowertriangle).*variance_part.lowertriangle)
        ∂ν_part  = SHermitianCompact(digamma.(νplushalf.lowertriangle).-digamma.(ν.lowertriangle).+2.0.*log.(a.lowertriangle))
        ∂a_part1 = SHermitianCompact(2.0.*ν.lowertriangle./a.lowertriangle)
        ∂a_part2 = SHermitianCompact(a.lowertriangle.*(2.0.*ν.lowertriangle.+1))
        ∂ν²_part  = SHermitianCompact(trigamma.(νplushalf.lowertriangle).-trigamma.(ν.lowertriangle))
        ∂ν∂a_part = SHermitianCompact(1.0./a.lowertriangle)
        ∂a²_part = SHermitianCompact(ν.lowertriangle./(a².lowertriangle))
        new{D,L}(σ,ν,a,σ²,a²,νplushalf,sdfconst,∂ν_part,∂a_part1,∂a_part2,∂ν²_part,∂ν∂a_part,∂a²_part)
    end
end
matern_sdf_normalising(ν,a) = gamma(ν+0.5)*a^(2ν) / (gamma(ν)*sqrt(π))

const Matern2D = Matern{2,3}
const Matern3D = Matern{3,6}
const Matern4D = Matern{4,10}

npars(::Type{Matern{D,L}}) where {D,L} = 3triangularnumber(D)
minbins(::Type{Matern{D,L}}) where {D,L} = 4096
nalias(model::Matern) = 3

function parameternames(::Type{Matern{D,L}}) where {D,L}
    σ = reduce(vcat,[i==j ? "σ_$i" : "ρ_$i$j" for j in i:D] for i in 1:D)
    ν = reduce(vcat,[i==j ? "ν_$i" : "ν_$i$j" for j in i:D] for i in 1:D)
    a = reduce(vcat,[i==j ? "a_$i" : "a_$i$j" for j in i:D] for i in 1:D)
    return Tuple(Symbol.([σ;ν;a]))
end

function add_sdf!(out, model::Matern{D,L}, ω) where {D,L}
    ω² = ω^2
    count = 1
    for i ∈ 1:D, j ∈ i:D
        out[count] += model.sdfconst[i,j]/((model.a²[i,j]+ω²)^(model.νplushalf[i,j]))
        count += 1
    end
    return nothing
end

function grad_add_sdf!(out, model::Matern{D,L}, ω) where {D,L}
    ω² = ω^2
    count = 1
    for i ∈ 1:D, j ∈ i:D
        asq_plus_omsq = model.a²[i,j]+ω²
        sdf = model.sdfconst[i,j]/((asq_plus_omsq)^(model.νplushalf[i,j]))
        # σ/ρ
        if i == j
            out[count,count] += 2sdf/model.σ[i,i]
        else
            out[count,count] += sdf/model.σ[i,j]
            out[count,indexLT(i,i,D)] += sdf/model.σ[i,i] # derivative with respect to σᵢ
            out[count,indexLT(j,j,D)] += sdf/model.σ[j,j] # derivative with respect to σⱼ
        end
        # ν ## digamma(model.νplushalf[i,j])-digamma(model.ν[i,j])+2log(model.a[i,j]) # precomputed as no dependency on ω
        out[count,L+count] += sdf * (model.∂ν_part[i,j]-log(asq_plus_omsq))
        # a ## 2model.ν[i,j]/model.a[i,j] - model.a[i,j]*(2model.ν[i,j]+1)/... # precomputed as no dependency on ω
        out[count,2L+count] += sdf * (model.∂a_part1[i,j] - model.∂a_part2[i,j]/asq_plus_omsq)
        count += 1
    end
    return nothing
end

function hess_add_sdf!(out, model::Matern{D,L}, ω) where {D,L}
    ω² = ω^2
    count = 1
    for i ∈ 1:D, j ∈ i:D
        asq_plus_omsq = model.a²[i,j]+ω²
        sdf = model.sdfconst[i,j]/((asq_plus_omsq)^(model.νplushalf[i,j]))
        sdf∂ν = sdf * (model.∂ν_part[i,j]-log(asq_plus_omsq))
        sdf∂a = sdf * (model.∂a_part1[i,j] - (model.∂a_part2[i,j])/asq_plus_omsq)
        # σ/ρ
        if i == j
            out[count,indexLT(count,count,   3L)]   += 2sdf/model.σ²[i,i]  # ∂σᵢ²
            out[count,indexLT(count,count+L, 3L)]   += 2sdf∂ν/model.σ[i,i] # ∂σᵢ∂νᵢⱼ
            out[count,indexLT(count,count+2L,3L)]   += 2sdf∂a/model.σ[i,i] # ∂σᵢ∂aᵢⱼ
        else
            iind = indexLT(i,i,D)
            jind = indexLT(j,j,D)
            out[count,indexLT(iind, count,   3L)] += sdf/(model.σ[i,i]*model.σ[i,j]) # ∂σᵢ∂ρᵢⱼ
            out[count,indexLT(iind, jind,    3L)] += sdf/(model.σ[i,i]*model.σ[j,j]) # ∂σᵢ∂σⱼ
            out[count,indexLT(jind, count,   3L)] += sdf/(model.σ[i,j]*model.σ[j,j]) # ∂σⱼ∂ρᵢⱼ
            out[count,indexLT(iind, count+L, 3L)] += sdf∂ν/model.σ[i,i] # ∂σᵢ∂νᵢⱼ
            out[count,indexLT(iind, count+2L,3L)] += sdf∂a/model.σ[i,i] # ∂σᵢ∂aᵢⱼ
            out[count,indexLT(count,count+L, 3L)] += sdf∂ν/model.σ[i,j] # ∂ρᵢⱼ∂νᵢⱼ
            out[count,indexLT(count,count+2L,3L)] += sdf∂a/model.σ[i,j] # ∂ρᵢⱼ∂aᵢⱼ
            out[count,indexLT(jind, count+L, 3L)] += sdf∂ν/model.σ[j,j] # ∂σⱼ∂νᵢⱼ
            out[count,indexLT(jind, count+2L,3L)] += sdf∂a/model.σ[j,j] # ∂σⱼ∂aᵢⱼ
        end
        # ∂νᵢⱼ²
        out[count,indexLT(count+L, count+L, 3L)] = sdf∂ν^2/sdf + sdf*model.∂ν²_part[i,j]
        # ∂νᵢⱼ∂aᵢⱼ
        out[count,indexLT(count+L, count+2L,3L)] = sdf∂ν*sdf∂a/sdf + 2sdf*(model.∂ν∂a_part[i,j]-model.a[i,j]/asq_plus_omsq)
        # ∂aᵢⱼ²
        out[count,indexLT(count+2L,count+2L,3L)] = sdf∂a^2/sdf + 2sdf*(model.νplushalf[i,j]*(2model.a²[i,j]/(asq_plus_omsq^2) - 1/asq_plus_omsq) - model.∂a²_part[i,j])
        count += 1
    end
    return nothing
end

## Univariate

struct Matern1D <: UnknownAcvTimeSeriesModel{1}
    σ::Float64
    ν::Float64
    a::Float64
    σ²::Float64
    a²::Float64
    νplushalf::Float64
    sdfconst::Float64
    ∂ν_part::Float64
    ∂a_part1::Float64
    ∂a_part2::Float64
    ∂ν²_part::Float64
    ∂ν∂a_part::Float64
    ∂a²_part::Float64
    function Matern1D(σ,ν,a)
        σ > 0 || throw(ArgumentError("Matern1D process requires 0 < σ."))
        ν > 0 || throw(ArgumentError("Matern1D process requires 0 < ν."))
        a > 0 || throw(ArgumentError("Matern1D process requires 0 < a."))
        sdfconst = matern_sdf_normalising(ν,a)*σ^2
        ∂ν_part  = digamma(ν+0.5)-digamma(ν)+2.0*log(a)
        ∂a_part1 = 2.0ν/a
        ∂a_part2 = a*(2.0*ν+1)
        ∂ν²_part  = trigamma(ν+0.5)-trigamma(ν)
        ∂ν∂a_part = 1.0/a
        ∂a²_part = ν/(a^2)
        new(σ, ν, a, σ^2, a^2, ν+0.5, sdfconst, ∂ν_part, ∂a_part1, ∂a_part2, ∂ν²_part, ∂ν∂a_part, ∂a²_part)
    end
    function Matern1D(x::AbstractVector{Float64})
        length(x) == npars(Matern1D) || throw(ArgumentError("Matern1D process has $(npars(Matern1D)) parameters, but $(length(x)) were provided."))
        @inbounds Matern1D(x[1], x[2], x[3])
    end
end

npars(::Type{Matern1D}) = 3
minbins(::Type{Matern1D}) = 4096
nalias(model::Matern1D) = 3

sdf(model::Matern1D, ω) = model.sdfconst/((model.a²+ω^2)^(model.νplushalf))

function grad_add_sdf!(out, model::Matern1D, ω)
    ω² = ω^2
    asq_plus_omsq = model.a²+ω²
    sdf = model.sdfconst/((asq_plus_omsq)^(model.νplushalf))
    # σ
    out[1] += 2sdf/model.σ
    # ν
    out[2] += sdf * (model.∂ν_part-log(asq_plus_omsq))
    # a
    out[3] += sdf * (model.∂a_part1 - model.∂a_part2/asq_plus_omsq)

    return nothing
end

function hess_add_sdf!(out, model::Matern1D, ω)
    ω² = ω^2
    asq_plus_omsq = model.a²+ω²
    sdf = model.sdfconst/((asq_plus_omsq)^(model.νplushalf))
    sdf∂ν = sdf * (model.∂ν_part-log(asq_plus_omsq))
    sdf∂a = sdf * (model.∂a_part1 - (model.∂a_part2)/asq_plus_omsq)
    
    # ∂σ²
    out[1]   += 2sdf/model.σ²
    # ∂σ∂ν
    out[2]   += 2sdf∂ν/model.σ 
    # ∂σ∂a
    out[3]   += 2sdf∂a/model.σ 
    # ∂ν²
    out[4] = sdf∂ν^2/sdf + sdf*model.∂ν²_part
    # ∂ν∂a
    out[5] = sdf∂ν*sdf∂a/sdf + 2sdf*(model.∂ν∂a_part-model.a/asq_plus_omsq)
    # ∂a²
    out[6] = sdf∂a^2/sdf + 2sdf*(model.νplushalf*(2model.a²/(asq_plus_omsq^2) - 1/asq_plus_omsq) - model.∂a²_part)
    
    return nothing
end
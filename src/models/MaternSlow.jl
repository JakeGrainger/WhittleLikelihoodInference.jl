struct MaternSlow{D,L} <: TimeSeriesModel{D}
    σ::SHermitianCompact{D,Float64,L}
    ν::SHermitianCompact{D,Float64,L}
    a::SHermitianCompact{D,Float64,L}
    a²::SHermitianCompact{D,Float64,L}
    νplushalf::SHermitianCompact{D,Float64,L}
    variance_part::SHermitianCompact{D,Float64,L}
    sdfconst::SHermitianCompact{D,Float64,L}
    acvconst::SHermitianCompact{D,Float64,L}
    function MaternSlow{D,L}(θ) where {D,L}
        length(θ) == npars(Matern{D,L}) || throw(ArgumentError("MaternSlow process has $(npars(MaternSlow{D,L})) parameters, but $(length(θ)) were provided."))
        L == triangularnumber(D) || error("MaternSlow{D,L} should satisfy L == D*(D+1)÷2")
        all(x->x>0, θ) || throw(ArgumentError("all parameters of MaternSlow should be > 0."))
        σ = @views SHermitianCompact(SVector{L,Float64}(θ[1:L]))
        all(i==j ? true : σ[i,j] < 1 for i in 1:size(σ,1) for j in 1:i) || throw(ArgumentError("ρ parameters must be < 1."))
        σ = @views SHermitianCompact(SVector{L,Float64}(θ[1:L]))
        ν = @views SHermitianCompact(SVector{L,Float64}(θ[L+1:2L]))
        a = @views SHermitianCompact(SVector{L,Float64}(θ[2L+1:end]))
        a² = SHermitianCompact((a.lowertriangle).^2)
        νplushalf = SHermitianCompact((ν.lowertriangle).+0.5)
        variance_part = SHermitianCompact(SMatrix{D,D,Float64}(i==j ? σ[i,i]^2 : σ[i,i]*σ[j,j]*σ[i,j] for i in 1:D, j in 1:D))
        sdfconst = SHermitianCompact(matern_sdf_normalising.(ν.lowertriangle,a.lowertriangle).*variance_part.lowertriangle)
        acvconst = SHermitianCompact(matern_acv_normalising.(ν.lowertriangle).*variance_part.lowertriangle)
        new{D,L}(σ,ν,a,a²,νplushalf,variance_part,sdfconst,acvconst)
    end
end

npars(::Type{MaternSlow{D,L}}) where {D,L} = 3triangularnumber(D)

parameternames(::Type{MaternSlow{D,L}}) where {D,L} = parameternames(Matern{D,L})

function add_sdf!(out, model::MaternSlow{D,L}, ω) where {D,L}
    ω² = ω^2
    count = 1
    for i ∈ 1:D, j ∈ i:D
        out[count] += model.sdfconst[i,j]/((model.a²[i,j]+ω²)^(model.νplushalf[i,j]))
        count += 1
    end
    return nothing
end

function acv!(out, model::MaternSlow{D,L}, τ::Number) where {D,L}
    modτ = abs(τ)
    count = 1
    for i ∈ 1:D, j ∈ i:D
        amodτ = model.a[i,j]*modτ
        out[count] = abs(τ)>1e-10 ? model.acvconst[i,j] * (amodτ)^model.ν[i,j] * besselk(model.ν[i,j],amodτ) : model.variance_part[i,j]
        count += 1
    end
    return nothing
end
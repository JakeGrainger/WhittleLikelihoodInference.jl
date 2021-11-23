function coherancy(model::TimeSeriesModel{D}, ω) where {D}
    S = sdf(model, ω)
    return [S[i,j]/sqrt(S[i,i]*S[j,j]) for i = 1:D, j = 1:D]
end
coherancy(model::TimeSeriesModel, Ω::AbstractVector{T}) where {T} = coherancy.(Ref(model), Ω)

coherance(model::TimeSeriesModel, ω) = abs.(coherancy(model, ω))
coherance(model::TimeSeriesModel, Ω::AbstractVector{T}) where {T} = coherance.(Ref(model), Ω)

groupdelay(model::TimeSeriesModel, ω) = angle.(coherancy(model, ω))
groupdelay(model::TimeSeriesModel, Ω::AbstractVector{T}) where {T} = groupdelay.(Ref(model), Ω)


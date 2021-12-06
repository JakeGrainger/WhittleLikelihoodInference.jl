"""
    coherancy(model::TimeSeriesModel, ω)

Compute the coherancy of a given model at frequency `ω`.
"""
function coherancy(model::TimeSeriesModel{D}, ω) where {D}
    S = sdf(model, ω)
    return [S[i,j]/sqrt(S[i,i]*S[j,j]) for i = 1:D, j = 1:D]
end
coherancy(::TimeSeriesModel{1}, ω) = error("coherancy not applicable for univariate models.")
"""
    coherance(model::TimeSeriesModel, ω)

Compute the coherance of a given model at frequency `ω`.
"""
coherance(model::TimeSeriesModel, ω) = abs.(coherancy(model, ω))

"""
    groupdelay(model::TimeSeriesModel, ω)

Compute the groupdelay of a given model at frequency `ω`.
"""
groupdelay(model::TimeSeriesModel, ω) = angle.(coherancy(model, ω))
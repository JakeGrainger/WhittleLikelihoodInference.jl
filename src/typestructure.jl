"Time series model with a parameter for the dimension of the timeseries model."
abstract type TimeSeriesModel{D} end

"""
Sub type to specify the need for additional routines approximating the acv from the sdf.
Such routines require additional storage and have different default functions in some cases.
"""
abstract type UnknownAcvTimeSeriesModel{D} <: TimeSeriesModel{D} end

"""
    AdditiveTimeSeriesModel(model1, model2)

Constructs an additive model from two timeseries models.
"""
struct AdditiveTimeSeriesModel{M₁,M₂,D} <: TimeSeriesModel{D}
    model1::M₁
    model2::M₂
    function AdditiveTimeSeriesModel(
        model1::M₁,
        model2::M₂,
    ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
        new{M₁,M₂,D}(model1, model2)
    end
    function AdditiveTimeSeriesModel{M₁,M₂,D}(θ) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D}
        @views new{M₁,M₂,D}(M₁(θ[1:npars(M₁)]), M₂(θ[npars(M₁)+1:end]))
    end
end

"""
    M₁::Type{<:TimeSeriesModel{D}}, M₂::Type{<:TimeSeriesModel{D}} -> AdditiveTimeSeriesModel{M₁,M₂,D}
"""
Base.:+(M₁::Type{<:TimeSeriesModel{D}}, M₂::Type{<:TimeSeriesModel{D}}) where {D} =
    AdditiveTimeSeriesModel{M₁,M₂,D}

# Broadcasting support for models
Base.broadcastable(model::TimeSeriesModel) = Ref(model)

"""
    ndims(::TimeSeriesModel) -> Integer
    ndims(::Type{<:TimeSeriesModel}) -> Integer

Return the dimension of a timeseries model.
"""
ndims(::TimeSeriesModel{D}) where {D} = D
ndims(::Type{<:TimeSeriesModel{D}}) where {D} = D

"""
    nlowertriangle_dimension(::TimeSeriesModel) -> Integer
    nlowertriangle_dimension(::Type{<:TimeSeriesModel}) -> Integer

Return the number of elements in the lower triangle the spectral density matrix function of a timeseries model.
"""
nlowertriangle_dimension(::TimeSeriesModel{D}) where {D} = triangularnumber(D)
nlowertriangle_dimension(::Type{<:TimeSeriesModel{D}}) where {D} =
    triangularnumber(D)

"""
    parameternames(::TimeSeriesModel) -> Vector{String}
    parameternames(::Type{<:TimeSeriesModel}) -> Vector{String}

Return the parameter names for a given timeseries model
"""
parameternames(model::TimeSeriesModel) = parameternames(typeof(model))

# """
#     parameter(::TimeSeriesModel) -> Vector

# Returns the parameter vector for a timeseries model
# """
# parameter(model::AdditiveTimeSeriesModel) =
#     vcat(parameter(model.model1), parameter(model.model2))
npars(model::TimeSeriesModel) = npars(typeof(model))
npars(
    ::Type{AdditiveTimeSeriesModel{M₁,M₂,D}},
) where {M₁<:TimeSeriesModel{D},M₂<:TimeSeriesModel{D}} where {D} = npars(M₁) + npars(M₂)

"""
    nlowertriangle_parameter(::TimeSeriesModel) -> Integer
    nlowertriangle_parameter(::Type{<:TimeSeriesModel}) -> Integer

Return the number of elements in the lower triangle of the hessian matrix for a model with given number of parameters.
"""
nlowertriangle_parameter(model::TimeSeriesModel) = nlowertriangle_parameter(typeof(model))
nlowertriangle_parameter(model::Type{<:TimeSeriesModel}) =
    StaticArrays.triangularnumber(npars(model))

"""
    indexLT(i,j,d)

Return the lower triangle index for an elements of a symmetric matrix assuming the lower triangle is encoded down the columns.
"""
indexLT(i::Int, j::Int, d::Int) =
    i >= j ? (((2d - j) * (j - 1)) ÷ 2) + i : (((2d - i) * (i - 1)) ÷ 2) + j

"
    nalias(model::TimeSeriesModel)

Returns the number of times a sdf should be alised for a given model."
nalias(::TimeSeriesModel) = 5 # default aliasing method

"
    minbins(::TimeSeriesModel)

Returns the minimum number of bins required for a good approximation of the sdf from the acv for a given model."
minbins(::TimeSeriesModel) = 8192 # default min bins method
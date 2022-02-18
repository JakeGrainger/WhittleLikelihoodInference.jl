"Time series model with a parameter for the dimension of the timeseries model."
abstract type TimeSeriesModel{D,T} end

"""
Sub type to specify the need for additional routines approximating the acv from the sdf.
Such routines require additional storage and have different default functions in some cases.
"""
abstract type UnknownAcvTimeSeriesModel{D,T} <: TimeSeriesModel{D,T} end

"""
    AdditiveTimeSeriesModel(model1, model2)

Constructs an additive model from two timeseries models.
"""
struct AdditiveTimeSeriesModel{M₁,M₂,D,T} <: TimeSeriesModel{D,T}
    model1::M₁
    model2::M₂
    function AdditiveTimeSeriesModel(
        model1::M₁,
        model2::M₂,
    ) where {M₁<:TimeSeriesModel{D,T},M₂<:TimeSeriesModel{D,T}} where {D,T}
        new{M₁,M₂,D,T}(model1, model2)
    end
    function AdditiveTimeSeriesModel{M₁,M₂,D}(θ) where {M₁<:TimeSeriesModel{D,T},M₂<:TimeSeriesModel{D,T}} where {D,T}
        @views new{M₁,M₂,D,T}(M₁(θ[1:npars(M₁)]), M₂(θ[npars(M₁)+1:end]))
    end
end

"""
    M₁::Type{<:TimeSeriesModel{D,T}} + M₂::Type{<:TimeSeriesModel{D,T}} -> AdditiveTimeSeriesModel{M₁,M₂,D,T}
"""
Base.:+(M₁::Type{<:TimeSeriesModel{D,T}}, M₂::Type{<:TimeSeriesModel{D,T}}) where {D,T} =
    AdditiveTimeSeriesModel{M₁,M₂,D,T}

# Broadcasting support for models
Base.broadcastable(model::TimeSeriesModel) = Ref(model)

"""
    ndims(::TimeSeriesModel) -> Integer
    ndims(::Type{<:TimeSeriesModel}) -> Integer

Return the dimension of a timeseries model.
"""
ndims(::TimeSeriesModel{D,T}) where {D,T} = D
ndims(::Type{<:TimeSeriesModel{D,T}}) where {D,T} = D

"""
    nlowertriangle_dimension(::TimeSeriesModel) -> Integer
    nlowertriangle_dimension(::Type{<:TimeSeriesModel}) -> Integer

Return the number of elements in the lower triangle the spectral density matrix function of a timeseries model.
"""
nlowertriangle_dimension(::TimeSeriesModel{D,T}) where {D,T} = triangularnumber(D)
nlowertriangle_dimension(::Type{<:TimeSeriesModel{D,T}}) where {D,T} =
    triangularnumber(D)

"""
    parameternames(::TimeSeriesModel)
    parameternames(::Type{<:TimeSeriesModel})

Return the parameter names for a given timeseries model
"""
parameternames(model::TimeSeriesModel) = parameternames(typeof(model))
parameternames(model::Type{<:TimeSeriesModel}) = fieldnames(model)[1:npars(model)]

npars(model::TimeSeriesModel) = npars(typeof(model))
npars(
    ::Type{AdditiveTimeSeriesModel{M₁,M₂,D,T}},
) where {M₁<:TimeSeriesModel{D,T},M₂<:TimeSeriesModel{D,T}} where {D,T} = npars(M₁) + npars(M₂)

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
    minbins(model::TimeSeriesModel)

Returns the minimum number of bins required for a good approximation of the sdf from the acv for a given model."
minbins(::Type{<:TimeSeriesModel}) = 8192 # default min bins method

"""
    checkparameterlength(x,model::Type{<:TimeSeriesModel})

checks if the parameter vector is the correct length for the given model.
"""
@inline function checkparameterlength(x,model::Type{<:TimeSeriesModel})
    length(x) == npars(model) || throw(ArgumentError("$model model has $(npars(model)) parameters, but $(length(x)) were provided."))
    Base.require_one_based_indexing(x) || throw(ArgumentError("parameter vector should be a type using 1 indexing."))
end
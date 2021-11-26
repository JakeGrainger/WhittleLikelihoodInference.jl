module WhittleLikelihoodInference

using FFTW, LinearAlgebra, ToeplitzMatrices, StaticArrays, RecipesBase
using Distributions, SpecialFunctions, LazyArrays

import Base: ndims, show, size, getindex
import StaticArrays: triangularnumber

export
    TimeSeriesModel,
    AdditiveTimeSeriesModel,
    npars,
    ndims,
    parameternames,
    parameter,
    sdf, 
    asdf, 
    acv, 
    EI,
    coherancy,
    coherance,
    groupdelay,
    FiniteNormal,
    ## non-parametrics
    Periodogram,
    BartlettPeriodogram,
    SpectralEstimate,
    CoherancyEstimate,
    ## models
    OU,
    CorrelatedOU,
    ## Whittle
    DebiasedWhittleLikelihood,
    WhittleLikelihood

    include("typestructure.jl")
    include("memoryallocation.jl")
    include("properties/spectraldensity.jl")
    include("properties/autocovariance.jl")
    include("properties/expectedperiodogram.jl")
    include("properties/coherancy.jl")

    include("Whittle/generalwhittle.jl")
    include("Whittle/whittle.jl")
    include("Whittle/debiasedwhittle.jl")

    include("models/OU.jl")
    include("models/CorrelatedOU.jl")

    include("simulation.jl")
    include("nonparametric.jl")
    include("plotting.jl")

end

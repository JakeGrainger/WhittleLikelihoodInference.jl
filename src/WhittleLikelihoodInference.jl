module WhittleLikelihoodInference

using FFTW, LinearAlgebra, ToeplitzMatrices, StaticArrays, RecipesBase
using Distributions, SpecialFunctions, LazyArrays

import Base: ndims, show, size, getindex, @propagate_inbounds
import StaticArrays: triangularnumber

export
    TimeSeriesModel,
    UnknownAcvTimeSeriesModel,
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
    simulate_gp,
    ## non-parametrics
    Periodogram,
    BartlettPeriodogram,
    SpectralEstimate,
    CoherancyEstimate,
    ## models
    OU,
    CorrelatedOU,
    Matern,
    Matern1D,
    Matern2D,
    Matern3D,
    Matern4D,
    ## Whittle
    DebiasedWhittleLikelihood,
    WhittleLikelihood

    include("typestructure.jl")
    include("memoryallocation.jl")
    include("properties/spectraldensity.jl")
    include("properties/autocovariance.jl")
    include("properties/expectedperiodogram.jl")
    include("properties/coherancy.jl")

    include("simulation.jl")
    include("nonparametric.jl")
    include("plotting.jl")

    include("whittle/whittledata.jl")
    include("whittle/generalwhittle.jl")
    include("whittle/standardwhittle.jl")
    include("whittle/debiasedwhittle.jl")

    include("models/OU.jl")
    include("models/CorrelatedOU.jl")
    include("models/OUUnknown.jl")
    include("models/CorrelatedOUUnknown.jl")
    include("models/Matern.jl")
    include("models/MaternAcv.jl")

end

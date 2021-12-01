using Test, WhittleLikelihoodInference, FiniteDifferences
import WhittleLikelihoodInference: grad_sdf, hess_sdf, grad_acv, hess_acv

## define useful functions
# Function for approximating the gradient using finite differences.
approx_gradient(func, x) = reinterpret(ComplexF64,jacobian(central_fdm(5, 1), func, x)[1])
approx_gradient_uni(func, x) = complex.(permutedims(jacobian(central_fdm(5, 1), func, x)[1]))
# Function for approximating the hessian using finite differences.
function approx_hessian(grad, x)
    hess = zeros(ComplexF64, size(grad(x),1), length(x)*(length(x)+1)÷2)
    count = 1
    @views for i ∈ 1:length(x)
        hess[:,count:count+length(x)-i] = approx_gradient(y->grad(y)[:,i], x)[:,i:end]
        count += length(x)-i+1
    end
    return hess
end
function approx_hessian_uni(grad, x)
    full_hess = approx_gradient(grad, x)
    hess = ones(ComplexF64, size(full_hess,1)*(size(full_hess,1)+1)÷2)
    for i ∈ 1:size(full_hess,1), j ∈ 1:i
        hess[WhittleLikelihoodInference.indexLT(i,j,size(full_hess,1))] = full_hess[i,j]
    end
    return hess
end

## run tests
# include("typestructure_test.jl")
# include("memoryallocation_test.jl")
# include("properties/spectraldensity_test.jl")
# include("properties/autocovariance_test.jl")
# include("properties/expectedperiodogram_test.jl")
# include("properties/coherancy_test.jl")

# include("Whittle/generalwhittle_test.jl")
# include("Whittle/standardwhittle_test.jl")
# include("Whittle/debiasedwhittle_test.jl")

include("models/OU_test.jl")
include("models/CorrelatedOU_test.jl")
include("models/OUUnknown_test.jl")
include("models/CorrelatedOUUnknown_test.jl")
include("models/Matern_test.jl")

# include("simulation_test.jl")
# include("nonparametric_test.jl")
# include("plotting_test.jl")
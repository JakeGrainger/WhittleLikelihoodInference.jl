using Documenter, WhittleLikelihoodInference

makedocs(sitename="WhittleLikelihoodInference.jl",
    modules = [WhittleLikelihoodInference],
    pages = [
        "Home" => "index.md",
        "background.md",
        "starting.md",
        "models.md",
        "Likelihood functions" => [
            "WhittleLikelihood.md",
            "debiasedwhittlelikelihood.md"
        ],
        "fit.md",
        "docstrings.md"
    ]
)

deploydocs(
    repo = "github.com/JakeGrainger/WhittleLikelihoodInference.jl.git"
)
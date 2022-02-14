using Documenter, SpinGlassDynamics
# makedocs(
#     modules=[SpinGlassDynamics],
#     sitename="SpinGlassTensors.jl",
#     format=Documenter.LaTeX()
# )
makedocs(
    modules=[SpinGlassDynamics],
    sitename="SpinGlassDynamics.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true"
    )
)
deploydocs(
    repo="github.com/euro-hpc-pl/SpinGlassDynamics.jl.git",
)

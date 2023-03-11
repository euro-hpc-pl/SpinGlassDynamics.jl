using Documenter, SpinGlassDynamics
# makedocs(
#     modules=[SpinGlassDynamics],
#     sitename="SpinGlassDynamics.jl",
#     format=Documenter.LaTeX(platform="none"),
#     pages=[
#         "API Reference" => "index.md"
#     ]
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

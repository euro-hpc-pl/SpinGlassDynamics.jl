
using SpinGlassNetworks
using SpinGlassDynamics

using Test

my_tests = []

push!(my_tests,
    #"base.jl",
    #"CIM.jl",
    #"SDE.jl",
    #"SB.jl",
    "SBcuda.jl",
    "CIMcuda.jl"
)

for my_test in my_tests
    include(my_test)
end

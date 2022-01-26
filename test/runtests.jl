
using SpinGlassNetworks
using SpinGlassDynamics

using Test

my_tests = []

push!(my_tests,
    "CIM.jl",
)

for my_test in my_tests
    include(my_test)
end


using DifferentialEquations
using SpinGlassNetworks

using Test

my_tests = []

push!(my_tests,
        "lagrange.jl",
)

for my_test in my_tests
    include(my_test)
end

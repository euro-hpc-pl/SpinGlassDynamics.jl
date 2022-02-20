module SpinGlassDynamics

using MKL
using LinearAlgebra
using SpinGlassNetworks
using Distributions
using DifferentialEquations
using CUDA

include("CIM.jl")
include("SB.jl")
include("SBcuda.jl")

end #module

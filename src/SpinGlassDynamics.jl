module SpinGlassDynamics

using MKL
using LinearAlgebra
using SpinGlassNetworks
using Distributions
using DifferentialEquations
using CUDA
using DocStringExtensions

include("CIM.jl")
include("SB.jl")
include("SBcuda.jl")
include("CIMcuda.jl")

end #module

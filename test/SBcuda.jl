using DifferentialEquations
using SpinGlassNetworks
using LinearAlgebra
using Distributions

@testset "simulated bifurcation simulator for chimera instances." begin
    L = 2048

    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt")

    J = couplings(ig)
    M = -(J + transpose(J))
    sp = eigen(Symmetric(M), 1:1)
    λ = sp.values[]

    kerr_coeff = 1.
    detuning = 1.
    scale = 0.9 #* detuning / abs(λ)

    init_state = rand(Uniform(-1, 1), 2 * L)
    num_steps = 100
    dt = 0.1
    pump = t -> t / num_steps / dt

    kpo = KerrOscillators{Float64}(ig, kerr_coeff, detuning, pump, scale)
    dyn = KPODynamics{Float64}(init_state, num_steps, dt)

    en = cuda_evolve_kerr_oscillators(kpo, dyn, 100)

    #@testset "Energy found is negative and within the bounds" begin
    #    @test  en < 0.
    #end
    println("cuda kpo: ", en)
end

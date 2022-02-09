using DifferentialEquations
using SpinGlassNetworks
using LinearAlgebra
using Distributions

@testset "simulated bifurcation simulator for chimera instances." begin
    L = 128

    #ig = ising_graph("$(@__DIR__)/instances/chimera_random/$(L).txt") # no biases, E = -204.73
    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/002.txt") # no biases, E = -210.13
    #ig = ising_graph("$(@__DIR__)/instances/basic/4_001.txt") # no biases, E = -4.625

    J = couplings(ig)
    M = -(J + transpose(J))
    sp = eigen(Symmetric(M), 1:1)
    λ = sp.values[]

    kerr_coeff = 1.
    detuning = 1.
    scale = 0.7 #* detuning / abs(λ)

    init_state = rand(Uniform(-1, 1), 2 * L)
    num_steps = 100
    dt = 0.25
    pump = t -> t / num_steps / dt

    kpo = KerrOscillators{Float64}(ig, kerr_coeff, detuning, pump, scale)
    dyn = KPODynamics{Float64}(init_state, num_steps, dt)

    N = 500
    states = Vector{Vector{Int}}(undef, N)
    states_naive = copy(states)
    Threads.@threads for i ∈ 1:N
        states[i] = evolve_kerr_oscillators(kpo, dyn)
        states_naive[i] = naive_evolve_kerr_oscillators(kpo, dyn)
    end

    en = minimum(energy(states, ig))
    en_naive = minimum(energy(states_naive, ig))

    @testset "Energy found is at least negative and within the bounds" begin
        @test  en < 0.
    end
    println("kpo: ", en,)
    println("naive kpo: ", en_naive,)
end

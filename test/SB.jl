using DifferentialEquations
using SpinGlassNetworks
using LinearAlgebra

@testset "simulated bifurcation simulator for chimera instances." begin
    L = 128

    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/002.txt") # no biases, E = -210.13
    #ig = ising_graph("$(@__DIR__)/instances/basic/4_001.txt") # no biases, E = -4.625

    #J = couplings(ig)

    kerr_coeff = 1.
    detuning = 1.
    scale = 0.9

    init_state = rand(2 * L)
    num_steps = 100
    dt = 0.25
    pump = t -> t / num_steps / dt

    kpo = KerrOscillators{Float64}(ig, kerr_coeff, detuning, pump, scale)
    dyn = KPODynamics{Float64}(init_state, num_steps, dt)

    N = 100
    states = Vector{Vector{Int}}(undef, N)
    states_naive = copy(states)
    Threads.@threads for i ∈ 1:N
        states[i] = evolve_kerr_oscillators(kpo, dyn)
        states_naive[i] = naive_evolve_kerr_oscillators(kpo, dyn)
    end

    en = minimum(energy(states, ig))
    en_naive = minimum(energy(states_naive, ig))

    #@testset "Energy found is at least negative and within the bounds" begin
    #    @test  en_exact <= en < 0.
    #end
    println("kpo: ", en,)
    println("naive kpo: ", en_naive,)
end

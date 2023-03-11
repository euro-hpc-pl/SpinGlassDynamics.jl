using DifferentialEquations
using SpinGlassNetworks
using LinearAlgebra
using Distributions

@testset "Simulated Bifurcation simulator for chimera instances." begin
    L = 128

    en_tn = -210.13
    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt")

    kerr_coeff = 1.
    detuning = 1.0
    scale = 0.9

    init_state = rand(Uniform(-1, 1), 2 * L)
    num_steps = 500
    dt = 0.2
    α = 2.0
    pump = t -> t / num_steps / α / dt

    kpo = KerrOscillators{Float64}(ig, kerr_coeff, detuning, pump, scale)
    dyn = KPODynamics{Float64}(init_state, num_steps, dt)

    N = 500
    states = Vector{Vector{Int}}(undef, N)
    states_naive = copy(states)
    Threads.@threads for i ∈ 1:N
        states[i] = evolve_kerr_oscillators(kpo, dyn)
        states_naive[i] = naive_evolve_kerr_oscillators(kpo, dyn)
    end

    en = minimum(energy.(states, Ref(ig)))
    en_naive = minimum(energy.(states_naive, Ref(ig)))

    @testset "Energy found is at least negative and within the bounds" begin
        @test en / en_tn >= 0.9
        @test en < 0.
    end
    println("kpo: ", en,)
    println("naive kpo: ", en_naive)
end

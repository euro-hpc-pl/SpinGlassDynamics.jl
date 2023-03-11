using DifferentialEquations
using SpinGlassNetworks
using LinearAlgebra
using Distributions
using CSV

function bench(instance::String)
    ig = ising_graph(instance)

    kerr_coeff = 1.
    detuning = 1.0
    scale = 0.9

    init_state = rand(Uniform(-1, 1), 2 * L) # this is not used for now
    num_steps = 500
    dt = 0.5
    α = 2.0
    pump = t -> t / num_steps / α / dt

    kpo = KerrOscillators{Float64}(ig, kerr_coeff, detuning, pump, scale)
    dyn = KPODynamics{Float64}(init_state, num_steps, dt)

    en = cuda_evolve_kerr_oscillators(kpo, dyn, 1, (16, 16))

    println("cuda kpo: ", en[1])
    println("found by TN: ", en_tn)
    println("ratio: ", round(en[1] / en_tn, digits=2))
end

bench("$(@__DIR__)/Maximum-2-SAT/Maximum-2-SAT_Huge_D.qubo")

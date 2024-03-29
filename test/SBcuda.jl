using DifferentialEquations
using SpinGlassNetworks
using LinearAlgebra
using Distributions

@testset "simulated bifurcation simulator for chimera instances." begin
    L = 2048

    # This instance is without biases
    #en_tn = -3296.53 # (found by SpinGlassEngine @ β = 1)
    #ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/002.txt")
    #@assert biases(ig) ≈ zeros(L)

    en_tn = -3336.773383 # (found by SpinGlassEngine @ β = 3)
    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt")

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

    en = cuda_evolve_kerr_oscillators(kpo, dyn, 256, (16, 16))

    @testset "Energies on CPU & GPU agree and there are close to the estimated ground." begin
        @test en[1] ≈ en[2]
        @test en[1] / en_tn >= 0.9
    end

    println("cuda kpo: ", en[1])
    println("found by TN: ", en_tn)
    println("ratio: ", round(en[1] / en_tn, digits=2))
    println("max en: ", round(en[3], digits=2))
end

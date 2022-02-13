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

    init_state = rand(Uniform(-1, 1), 2 * L) # this is not used here
    num_steps = 500
    dt = 0.5
    pump = t -> t / num_steps / dt

    kpo = KerrOscillators{Float64}(ig, kerr_coeff, detuning, pump, scale)
    dyn = KPODynamics{Float64}(init_state, num_steps, dt)

    en = cuda_evolve_kerr_oscillators(kpo, dyn, 256, (16, 16))

    @testset "kerr_adjacency_matrix is created properly." begin
        JK = kerr_adjacency_matrix(ig)
        J, h = couplings(ig), biases(ig)
        L = size(J, 1)
        @test size(JK) == (L+1, L+1)
        @test JK == transpose(JK)
        @test diag(JK) ≈ zeros(L+1)
        @test JK[1:L, 1:L] == -(J + transpose(J))
        @test JK[end, 1:L] == -h
    end

    @testset "Energy found is negative and within the bounds" begin
        @test en[1] ≈ en[2]
        @test en[1] < 0.0
    end
    println("cuda kpo: ", en[1])
    println("found by TN: ", en_tn)
end

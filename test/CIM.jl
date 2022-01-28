using Noise
using SpinGlassNetworks

@testset "Simple Coherent Ising Machine simulator." begin
    L = 16

    ig = ising_graph("$(@__DIR__)/instances/basic/$(L)_001.txt")

    scale = 0.3
    μ = 0.0
    σ = 0.3
    noise = add_gauss(zeros(L), σ, μ)

    x0 = zeros(L)
    sat = 1.0
    time = 1000
    pump = collect(LinRange(-15, 0.01, time))

    opo = OpticalOscillators{Float64}(ig, scale, noise)
    dyn = OPODynamics{Float64}(x0, sat, pump)

    @testset  "OpticalOscillators and OPODynamics work properly." begin
        @test opo.scale ≈ scale
        @test opo.noise ≈ noise

        @test dyn.initial_state ≈ x0
        @test dyn.saturation ≈ sat
        @test length(dyn.pump) ≈ time
        @test dyn.pump ≈ pump
    end

    @testset "activation function works properly." begin
        a = dyn.saturation - rand()
        b = dyn.saturation + rand()
        @test activation(a, dyn.saturation) ≈ a
        @test activation(b, dyn.saturation) ≈ dyn.saturation
    end

    sp = brute_force(ig, :CPU, num_states=10)
    println(sp.energies)

    x = evolve_optical_oscillators(opo, dyn)

    en = energy(ig, digitize_state(x))
    println("en: ", en)
    println(x)
end

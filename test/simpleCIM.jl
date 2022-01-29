using Noise
using SpinGlassNetworks

function ramp(t::T, τ::T, pi::T, pf::T) where T <: Real
    p = (pf + pi) + (pf - pi) * tanh(t / τ)
    p / 2.0
end

@testset "Simple Coherent Ising Machine simulator for small Ising instances." begin
    L = 4
    ig = ising_graph("$(@__DIR__)/instances/basic/$(L)_001.txt")

    scale = 0.2
    μ = 0.0
    σ = 0.3
    noise = add_gauss(zeros(L), σ, μ)

    x0 = zeros(L)
    sat = 1.0
    time = 1000.
    momentum = 0.9
    pump = [ramp(t, time, -15.0, 0.0) for t ∈ -2*time:2*time]

    opo = OpticalOscillators{Float64}(ig, scale, noise)
    dyn = OPODynamics{Float64}(x0, sat, pump, momentum)

    @testset  "OpticalOscillators and OPODynamics work properly." begin
        @test opo.scale ≈ scale
        @test opo.noise ≈ noise

        @test dyn.initial_state ≈ x0
        @test dyn.saturation ≈ sat
        @test dyn.pump ≈ pump
    end

    sp = brute_force(ig, :CPU, num_states=1)
    x = evolve_optical_oscillators(opo, dyn)

    @testset "OPO find correct ground state." begin
        σ = digitize_state(x)
        @test energy(ig, σ) ≈ sp.energies[1]
    end
end

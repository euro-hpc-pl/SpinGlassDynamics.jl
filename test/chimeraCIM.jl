using Noise
using SpinGlassNetworks

function ramp(t::T, τ::T, pi::T, pf::T) where T <: Real
    p = (pf + pi) + (pf - pi) * tanh(t / τ)
    p / 2.0
end

@testset "Simple Coherent Ising Machine simulator for small Ising instance." begin
    L = 128

    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt")

    scale = 0.5
    amp = 1.0
    μ = 0.0
    σ = 0.1

    x0 = zeros(L)
    sat = 1.0
    time = 500.
    momentum = 0.6
    pump = [ramp(t, time, -15.0, 0.0) for t ∈ -2*time:2*time]

    opo = OpticalOscillators{Float64}(ig, scale, amp, μ, σ)
    dyn = OPODynamics{Float64}(x0, sat, pump, momentum)

    N = 100
    states = Vector{Vector{Int}}(undef, N)
    Threads.@threads for i ∈ 1:N
        states[i] = evolve_optical_oscillators(opo, dyn)
    end

    println(minimum(energy(states, ig)))
end

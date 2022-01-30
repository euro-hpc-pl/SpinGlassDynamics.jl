using Distributions
using SpinGlassNetworks

function ramp(t::T, τ::T, α::T, pi::T, pf::T) where T <: Real
    p = (pf + pi) + (pf - pi) * tanh(α * (2.0 * t / τ - 1.0))
    p / 2.0
end

@testset "Simple Coherent Ising Machine simulator for small Ising instance." begin
    L = 128

    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt")

    scale = 0.3
    noise = Normal(0.0, 0.1)

    x0 = zeros(L)
    sat = 1.0
    time = 1000.
    pi, pf, α = -15.0, 0.0, 2.0
    momentum = 0.4

    pump = [ramp(t, time, α, pi, pf) for t ∈ 0:time]

    opo = OpticalOscillators{Float64}(ig, scale, noise)
    dyn = OPODynamics{Float64}(x0, sat, pump, momentum)

    pump_nmfa = [10.0-0.1*i for i ∈ 1:time]
    opo_nmfa = OpticalOscillators{Float64}(ig, scale, noise)
    dyn_nmfa = OPODynamics{Float64}(x0, sat, pump_nmfa, momentum)

    N = 500
    states = Vector{Vector{Int}}(undef, N)
    states_nmfa = copy(states)
    Threads.@threads for i ∈ 1:N
        states[i] = evolve_optical_oscillators(opo, dyn)
        states_nmfa[i] = noisy_mean_field_annealing(opo_nmfa, dyn_nmfa)
    end

    # exact en = -210.93333400000003
    println(minimum(energy(states, ig)))
    println(minimum(energy(states_nmfa, ig)))

end

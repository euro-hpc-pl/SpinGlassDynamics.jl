using Distributions
using SpinGlassNetworks

function ramp(t::T, τ::T, α::T, pi::T, pf::T) where T <: Real
    p = (pf + pi) + (pf - pi) * tanh(α * (2.0 * t / τ - 1.0))
    p / 2.0
end

@testset "Coherent Ising Machine simulator for chimera instances (droplets)." begin
    L = 128

    en_exact = -210.933334
    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt")

    scale = 0.7
    noise = Normal(0.1, 0.3)

    x0 = 2.0 .* rand(L) .- 1.0
    sat = 1.0
    time = 1000.
    pi, pf, α = -5.0, 0.0, 2.0
    momentum = 0.9

    pump = [ramp(t, time, α, pi, pf) for t ∈ 1:time]

    opo = OpticalOscillators{Float64}(ig, scale, noise)
    dyn = OPODynamics{Float64}(x0, sat, pump, momentum)

    pump_nmfa = [ramp(t, time, α, 10., 0.01) for t ∈ 1:time]

    opo_nmfa = OpticalOscillators{Float64}(ig, scale, noise)
    dyn_nmfa = OPODynamics{Float64}(x0, sat, pump_nmfa, momentum)

    N = 500
    states = Vector{Vector{Int}}(undef, N)
    states_nmfa = copy(states)
    Threads.@threads for i ∈ 1:N
        states[i] = evolve_optical_oscillators(opo, dyn)
        states_nmfa[i] = noisy_mean_field_annealing(opo_nmfa, dyn_nmfa)
    end

    en = minimum(energy.(states, Ref(ig)))
    en_nmfa = minimum(energy.(states_nmfa, Ref(ig)))

    println(en_nmfa, " ", en)
    @testset "Energy is close to the estimated ground." begin
        @test en / en_exact > 0.9
        @test en_exact <= en_nmfa < 0.0
    end
end

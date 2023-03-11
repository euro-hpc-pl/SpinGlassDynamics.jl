using Distributions
using SpinGlassNetworks

function ramp(t::T, τ::T, α::T, pi::T, pf::T) where T <: Real
    p = (pf + pi) + (pf - pi) * tanh(α * (2.0 * t / τ - 1.0))
    p / 2.0
end

# function energy(ig, σ)
#     energy = 0
#     # quadratic
#     for edge ∈ edges(ig)
#         i, j = src(edge), dst(edge)         
#         J = get_prop(ig, i, j, :J) 
#         energy += σ[i] * J * σ[j]   
#     end 

#     # linear
#     for i ∈ vertices(ig)
#         h = get_prop(ig, i, :h)   
#         energy += h * σ[i]     
#     end    
#     return -energy
# end

@testset "Simple Coherent Ising Machine simulator for small Ising instance." begin
    L = 4

    ig = ising_graph("$(@__DIR__)/instances/basic/$(L)_001.txt")

    scale = 0.2
    noise = Normal(0.0, 0.1)

    x0 = zeros(L)
    sat = 1.0
    time = 1000.
    pi, pf, α = -15.0, 0.0, 2.0
    momentum = 0.4

    pump = [ramp(t, time, α, pi, pf) for t ∈ 0:time]

    opo = OpticalOscillators{Float64}(ig, scale, noise)
    dyn = OPODynamics{Float64}(x0, sat, pump, momentum)

    @testset  "OpticalOscillators and OPODynamics work properly." begin
        @test opo.scale ≈ scale
        @test opo.noise ≈ noise

        @test dyn.initial_state ≈ x0
        @test dyn.saturation ≈ sat
        @test dyn.pump ≈ pump
    end

    N = 500
    states = Vector{Vector{Int}}(undef, N)
    Threads.@threads for i ∈ 1:N
        states[i] = evolve_optical_oscillators(opo, dyn)
    end

    @test minimum(energy.(states, Ref(ig))) ≈ brute_force(ig, num_states=1).energies[1]
end

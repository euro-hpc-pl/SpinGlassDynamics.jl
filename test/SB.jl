@testset "simulated bifurcation simulator for chimera instances." begin
    L = 128

    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/002.txt") # no biases

    kerr_coeff = 1.
    detuning = 1.
    scale =  0.7 / sqrt(L)

    init_state = zeros(2 * L)
    num_steps = 100
    pump = t -> 2 * tanh(2 * t / L)
    dt = 0.1

    kpo = KerrOscillators{Float64}(ig, kerr_coeff, detuning, scale)
    dyn = KPODynamics{Float64}(init_state, num_steps, pump, dt)

    N = 1
    states = Vector{Vector{Int}}(undef, N)
    #Threads.@threads
    for i ∈ 1:N
        states[i] = evolve_kerr_oscillators(kpo, dyn)
    end

    en = minimum(energy(states, ig))
    en_exact = -210.13333399999996

    #@testset "Energy found is at least negative and within the bounds" begin
    #    @test  en_exact <= en < 0.
    #end
    println(states[1])
    println("kpo: ", en)
end

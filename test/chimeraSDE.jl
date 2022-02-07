using DifferentialEquations

@testset "Stochastic differential equations for chimera." begin    L = 128

    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/002.txt") # no biases

    scale = sqrt(10)
    amp = 70.
    x0 = zeros(2 * L)
    time = (0.0, 100.)
    pump = t -> 2 * tanh(2 * t / L)

    dopo = DegenerateOscillators{Float64}(ig, scale, amp, x0, pump, time)

    N = 1
    states = Vector{Vector{Int}}(undef, N)
    #Threads.@threads
    for i ∈ 1:N
        states[i] = evolve_degenerate_oscillators(dopo; args=(SRIW1(), ))
    end

    en = minimum(energy(states, ig))
    en_exact = -210.933334

    #@testset "Energy found is at least negative and within the bounds" begin
    #    @test  en_exact <= en < 0.
    #end
    println("dopo: ", en)
end

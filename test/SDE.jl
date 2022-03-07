using DifferentialEquations

@testset "Stochastic differential equations." begin
    L = 128

    en_exact = -210.13
    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt")

    scale = sqrt(10)
    amp = 30.
    x0 = zeros(2 * L)
    time = (0.0, 100.)
    pump = t -> 2 * tanh(2 * t / L)

    dopo = DegenerateOscillators{Float64}(ig, scale, amp, x0, pump, time)

    N = 100
    states = Vector{Vector{Int}}(undef, N)
    Threads.@threads for i ∈ 1:N
        states[i] = evolve_degenerate_oscillators(dopo; args=(SRIW1(), ))
    end

    en = minimum(energy(ig,states))

    @testset "Energy is close to the estimated ground." begin
        @test  en / en_exact >= 0.9
    end
end

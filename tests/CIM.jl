
@testset "Simple Coherent Ising Machine simulator." begin
    m, n, t = 3, 4, 3
    L = m * n * t

    ig = ising_graph("$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt")

    x0 = rand(L)
    total_time = 100
    μ = 0.2
    ξ = 0.6
    f = rand(L)
    xsat = 2.0

    x = naive_coherent_Ising_machine(ig, x0, μ, ξ, f, xsat)
end

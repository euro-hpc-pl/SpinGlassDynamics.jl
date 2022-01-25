
@testset "Simple Coherent Ising Machine simulator." begin
    m, n, t = 3, 4, 3
    L = m * n * t

    ig = ising_graph("$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt")

    opo = OpticalOscillators{Float64}(ig, -5.0, 0.6, rand(L))
    dyn = OPODynamics{Float64}(rand(L), 2.0, 100)
    x = evolve_optical_oscillators(opo, dyn)
end

@testset "Simple Coherent Ising Machine simulator." begin
    m, n, t = 3, 4, 3
    L = m * n * t

    ig = ising_graph("$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt")

    pump = -5.0
    scale = 0.6
    noise = rand(L)

    x0 = rand(L)
    sat = 0.7
    time = 1000.0

    opo = OpticalOscillators{Float64}(ig, pump, scale, noise)
    dyn = OPODynamics{Float64}(x0, sat, time)

    @testset  "OpticalOscillators and OPODynamics work properly." begin
        @test opo.pump ≈ pump
        @test opo.scale ≈ scale
        @test opo.noise ≈ noise

        @test dyn.initial_state ≈ x0
        @test dyn.saturation ≈ sat
        @test dyn.total_time ≈ time
    end

    x = evolve_optical_oscillators(opo, dyn)
    println(x)
end

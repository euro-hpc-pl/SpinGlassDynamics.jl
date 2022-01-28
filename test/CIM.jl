using Noise

@testset "Simple Coherent Ising Machine simulator." begin
    m, n, t = 3, 4, 3
    L = m * n * t

    ig = ising_graph("$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt")

    scale = 0.6
    μ = 0.0
    σ = 0.3
    noise = add_gauss(zeros(L), σ, μ)

    x0 = zeros(L)
    sat = 1.0
    time = 4000.0
    pump = [-15.0 for i ∈ 1:time]

    opo = OpticalOscillators{Float64}(ig, scale, noise)
    dyn = OPODynamics{Float64}(x0, sat, pump)

    @testset  "OpticalOscillators and OPODynamics work properly." begin
        @test opo.scale ≈ scale
        @test opo.noise ≈ noise

        @test dyn.initial_state ≈ x0
        @test dyn.saturation ≈ sat
        @test length(dyn.pump) ≈ time
        @test dyn.pump ≈ pump
    end

    @testset "activation function works properly." begin
        a = dyn.saturation - rand()
        b = dyn.saturation + rand()
        @test activation(a, dyn.saturation) ≈ a
        @test activation(b, dyn.saturation) ≈ dyn.saturation
    end

    x = evolve_optical_oscillators(opo, dyn)
    println(x)
end

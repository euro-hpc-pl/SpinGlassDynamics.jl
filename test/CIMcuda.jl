using SpinGlassNetworks
using Distributions

function ramp(t::T, τ::T, α::T, pi::T, pf::T) where T <: Real
    p = (pf + pi) + (pf - pi) * tanh(α * (2.0 * t / τ - 1.0))
    p / 2.0
end

@testset "CIM simulator for chimera instances." begin
    L = 2048

    # This instance is without biases
    #en_tn = -3296.53 # (found by SpinGlassEngine @ β = 1)
    #ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/002.txt")
    #@assert biases(ig) ≈ zeros(L)

    en_tn = -3336.773383 # (found by SpinGlassEngine @ β = 3)
    ig = ising_graph("$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt")

    scale = 0.7
    noise = Normal(0.1, 0.3)

    x0 = zeros(L) # this is not used for now
    sat = 1.0
    time = 250.
    pi, pf, α = -5.0, 0.0, 5.0
    momentum = 0.9

    pump = [ramp(t, time, α, pi, pf) for t ∈ 1:time]

    opo = OpticalOscillators{Float64}(ig, scale, noise)
    dyn = OPODynamics{Float64}(x0, sat, pump, momentum)

    en = cuda_evolve_optical_oscillators(opo, dyn, 256, (16, 16))

    @testset "Energies on CPU & GPU agree and there are close to the estimated ground." begin
        @test en[1] ≈ en[2]
        @test en[1] / en_tn >= 0.9
    end

    println("cuda opo: ", en[1])
    println("found by TN: ", en_tn)
    println("ratio: ", round(en[1] / en_tn, digits=2))
    println("max en: ", round(en[3], digits=2))
end

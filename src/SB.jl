# SB.jl: different algorithms for simulated bifurcation.
export
    KerrOscillators,
    KPODynamics,
    evolve_kerr_oscillators

# https://www.science.org/doi/pdf/10.1126/sciadv.aav2372
struct KerrOscillators{T <: Real}
    ig::IsingGraph
    kerr_coeff::T
    detuning::T
    scale::T
end
struct KPODynamics{T <: Real}
    init_state::Vector{T}
    num_steps::Int
    pump::Function
    dt::T
end

function evolve_kerr_oscillators(
    kpo::KerrOscillators{T},
    dyn::KPODynamics
) where T <: Real
    N = length(dyn.init_state)
    @assert N % 2 == 0

    J = couplings(kpo.ig)
    x = dyn.init_state[1:N÷2]
    y = dyn.init_state[N÷2+1:end]

    for i ∈ 1:dyn.num_steps
        x .+= kpo.detuning .* y .* dyn.dt
        Φ = kpo.scale .* J * x
        p = dyn.pump(i * dyn.dt)
        y .-= (kpo.kerr_coeff .* x .^ 3 .+ (kpo.detuning - p) .* x .- Φ) .* dyn.dt
    end
    Int.(sign.(x))
end

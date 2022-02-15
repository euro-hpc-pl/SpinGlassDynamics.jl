# SB.jl: different algorithms for simulated bifurcation.
export
    KerrOscillators,
    KPODynamics,
    evolve_kerr_oscillators,
    naive_evolve_kerr_oscillators

# https://www.science.org/doi/pdf/10.1126/sciadv.aav2372
struct KerrOscillators{T <: Real}
    ig::IsingGraph
    kerr_coeff::T
    detuning::T
    pump::Function
    scale::T
end
struct KPODynamics{T <: Real}
    init_state::Vector{T}
    num_steps::Int
    dt::T
end

# https://www.science.org/doi/epdf/10.1126/sciadv.abe7953
# This procedure uses the symplectic Euler method,
# which make it potentially fast for CPUs/ GPU / FPGA.
function evolve_kerr_oscillators(kpo::KerrOscillators{T}, dyn::KPODynamics) where T <: Real
    N = length(dyn.init_state)
    @assert N % 2 == 0

    J, h = couplings(kpo.ig), biases(kpo.ig)
    J += transpose(J)

    x = dyn.init_state[1:N÷2]
    y = dyn.init_state[N÷2+1:end]

    for i ∈ 1:dyn.num_steps
        p = kpo.pump(i * dyn.dt)
        Φ = kpo.scale .* (J * x .+ h)
        y .-= (kpo.kerr_coeff .* x .^ 3 .+ (kpo.detuning - p) .* x .+ Φ) .* dyn.dt
        #y .-= ((kpo.detuning - p) .* x .+ Φ) .* dyn.dt
        x .+= kpo.detuning .* y .* dyn.dt

        m = abs.(x) .> 1.0
        y .*= m
        x = x .* (.!m) .+ sign.(x) .* m
    end
    Int.(sign.(x))
end

# This procedure does the same as the evolve_kerr_oscillators.
# However, it uses DifferentialEquations engine to solve ODEs.
function kerr_system(u, kpo, t)
    J, h = couplings(kpo.ig), biases(kpo.ig)
    J += transpose(J)
    N = length(u) ÷ 2
    x = @view u[1:N]
    y = @view u[N+1:end]

    Φ = kpo.scale .* (J * x .+ h)
    vcat(
        kpo.detuning .* y,
        -(kpo.kerr_coeff .* x .^ 2 .+ (kpo.detuning - kpo.pump(t))) .* x .- Φ
    )
    #=
    # These equations are based on the simplification from
    # https://www.science.org/doi/epdf/10.1126/sciadv.abe7953
    Φ = kpo.scale .* (J * sign.(x) .+ h)
    vcat(
        kpo.detuning .* y,
        -(kpo.detuning - kpo.pump(t)) .* x .- Φ
    )
    =#
end

function naive_evolve_kerr_oscillators(
    kpo::KerrOscillators{T},
    dyn::KPODynamics,
    args=()
) where T <: Real
    tspan = (0., dyn.num_steps * dyn.dt)
    ode = ODEProblem(kerr_system, dyn.init_state, tspan, kpo)
    sol = solve(ode, save_everystep=false, args...)
    x = sol.u[end]
    N = length(x) ÷ 2
    Int.(sign.(x[1:N]))
end

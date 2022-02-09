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

function kerr_kernel(states, J, dt, Δ, p, K, ξ, num_steps, num_rep)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = size(J, 2)

    for r ∈ 1:num_rep
        x[idx], y[idx] = 2 * rand() - 1, 2 * rand() - 1
        for i ∈ 1:num_steps
            x[idx] += Δ * y[idx] * dt
            # syn is needed
            Φ = 0.0
            for j ∈ 1:L Φ += J[idx, j] * x[j] end
            y[idx] -= (K * x[idx] ^ 3 + (Δ - p[i+1]) * x[idx] - ξ * Φ) * dt
        end
        states[r, idx] = Int(sign(x[idx]))
    end
    return
end

function energy_kernel(J, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = size(J, 1)
    en = 0.0
    for k=1:L, l=k+1:L
        @inbounds en += σ[k, idx] * J[k, l] * σ[l, idx]
    end
    energies[idx] = en
    return
end

function cuda_kerr_oscillators(kpo::KerrOscillators{T}, dyn::KPODynamics) where T <: Real
    L = nv(kpo.ig)
    C = -couplings(kpo.ig)

    σ = CUDA.zeros(Int32, num_rep, L)
    J = CUDA.CuArray(C + transpose(C))

    bl = 16
    th = ceil(L / bl)

    @cuda threads=th blocks=bl kerr_kernel(σ, J, dt, Δ, p, K, ξ, num_steps, num_rep)
end

# https://www.science.org/doi/epdf/10.1126/sciadv.abe7953
# This procedure uses the symplectic Euler method,
# which make it potentially fast for CPUs/ GPU / FPGA.
function evolve_kerr_oscillators(kpo::KerrOscillators{T}, dyn::KPODynamics) where T <: Real
    N = length(dyn.init_state)
    @assert N % 2 == 0

    J = -couplings(kpo.ig)
    J += transpose(J)
    x = dyn.init_state[1:N÷2]
    y = dyn.init_state[N÷2+1:end]

    for i ∈ 1:dyn.num_steps
        x .+= kpo.detuning .* y .* dyn.dt
        Φ = kpo.scale .* J * x
        p = kpo.pump((i+1) * dyn.dt)
        y .-= (kpo.kerr_coeff .* x .^ 3 .+ (kpo.detuning - p) .* x .- Φ) .* dyn.dt
    end
    Int.(sign.(x))
end

# This procedure does the same as the evolve_kerr_oscillators.
# However, it uses DifferentialEquations engine to solve ODEs.
# This is slow but potentially accurate to an arbitrary precision.
function kerr_system(u, kpo, t)
    J = -couplings(kpo.ig)
    J += transpose(J)
    N = length(u) ÷ 2
    x = @view u[1:N]
    y = @view u[N+1:end]

    Φ = kpo.scale .* J * x
    vcat(
        kpo.detuning .* y,
        -(kpo.kerr_coeff .* x .^ 2 .+ (kpo.detuning - kpo.pump(t))) .* x .+ Φ
    )
    #=
    # These equations are based on the simplification from
    # https://www.science.org/doi/epdf/10.1126/sciadv.abe7953
    Φ = kpo.scale .* J * sign.(x)
    vcat(
        kpo.detuning .* y,
        -(kpo.detuning - kpo.pump(t)) .* x .+ Φ
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

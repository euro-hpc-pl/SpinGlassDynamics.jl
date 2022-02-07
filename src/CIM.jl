# CIM.jl: different algorithms to simulate coherent Ising machines.

export
       OPODynamics,
       OpticalOscillators,
       DegenerateOscillators,
       noisy_mean_field_annealing,
       evolve_optical_oscillators,
       evolve_degenerate_oscillators

@inline zeros_like(x::AbstractArray) = zeros(eltype(x), size(x))

# Optical Parametric Oscillator (OPO)
# Based on https://arxiv.org/pdf/1901.08927.pdf
struct OpticalOscillators{T <: Real}
    ig::IsingGraph
    scale::T
    noise::Distributions.Sampleable
end
struct OPODynamics{T <: Real}
    initial_state::Vector{T}
    saturation::T
    pump::Vector{T}
    momentum::T
end

function evolve_optical_oscillators(
    opo::OpticalOscillators{T},
    dyn::OPODynamics{T}
)  where T <: Real
    J, h = couplings(opo.ig), biases(opo.ig)
    x = dyn.initial_state
    L = length(x)
    Δm = zeros_like(x)
    for p ∈ dyn.pump
        Δx = p .* x .+ opo.scale .* (J * x .+ h) .+ rand(opo.noise, L)
        m = (1 - dyn.momentum) .* Δx + dyn.momentum .* Δm
        x .+= m .* (abs.(x .+ m) .< dyn.saturation)
        Δm = m
    end
    Int.(sign.(x))
end

# Noisy mean-field annealing (NMFA)
# Based on https://arxiv.org/pdf/1806.08422.pdf
function noisy_mean_field_annealing(
    opo::OpticalOscillators{T},
    dyn::OPODynamics{T}
)  where T <: Real
    J, h = couplings(opo.ig), biases(opo.ig)
    x = dyn.initial_state
    L = length(x)
    nmr = sqrt.(h .^ 2 .+ dropdims(sum(J .^ 2, dims=2), dims=2))
    for p ∈ dyn.pump
        ϕ = (J * x .+ h) ./ nmr .+ rand(opo.noise, L)
        x = (1 - dyn.momentum) .* x .- dyn.momentum .* tanh.(ϕ ./ p)
    end
    Int.(sign.(x))
end

# Degenerate Optical Parametric Oscillator (DOPO)
# Based on https://www.nature.com/articles/s41467-018-07328-1
struct DegenerateOscillators{T <: Real}
    ig::IsingGraph
    scale::T
    amp::T
    initial_state::Vector{T}
    pump::Function
    time::NTuple{2, T}
end

function coherence(u, dopo, t)
    J, h = couplings(dopo.ig), biases(dopo.ig)
    N = length(u) ÷ 2
    c = @view u[1:N]
    q = @view u[N+1:end]
    v = c .^ 2 .+ q .^ 2 .+ 1
    Φ = dopo.scale .* (J * c .+ h)
    vcat((dopo.pump(t) .- v) .* c .+ Φ, (-dopo.pump(t) .- v) .* q)
end

function noise(u, dopo, t)
    N = length(u) ÷ 2
    c = @view u[1:N]
    q = @view u[N+1:end]
    v = sqrt.(c .^ 2 + q .^ 2 .+ 1/2) ./ dopo.amp
    vcat(v, v)
end

function evolve_degenerate_oscillators(
    dopo::DegenerateOscillators{T};
    args=()
) where T <: Real
    sde = SDEProblem(coherence, noise, dopo.initial_state, dopo.time, dopo)
    sol = solve(sde, save_everystep=false, args...)
    x = sol.u[end]
    N = length(x) ÷ 2
    Int.(sign.(x[1:N]))
end

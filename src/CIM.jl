# CIM.jl: different algorithms to simulate coherent Ising machines.

export
       OpticalOscillators,
       OPODynamics,
       evolve_optical_oscillators,
       noisy_mean_field_annealing

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
        m = (1.0 - dyn.momentum) .* Δx + dyn.momentum .* Δm
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
        x = (1.0 - dyn.momentum) .* x .- dyn.momentum .* tanh.(ϕ ./ p)
    end
    Int.(sign.(x))
end

# Degenerate Optical Parametric Oscillator (DOPO)
# Based on https://www.nature.com/articles/s41467-018-07328-1
struct DegenerateOscillators{T <: Real}
    ig::IsingGraph
    scale::T
    initial_state::Vector{T}
    pump::Function
    time::NTuple{T, 2}
end

function coherence!(du, u, dopo, t)
    J, h = couplings(dopo.ig), biases(dopo.ig)
    N = length(u) / 2
    c = @view u[1:N]
    s = @view u[N+1:end]
    v = c .^ 2 .+ s .^ 2 .+ 1.0
    du = vcat(
        (dopo.pump(t) .- v) .* c .+ dopo.scale .* (J * c .+ h), (-dopo.pump(t) .- v) .* s
    )
end

function noise!(du, u, dopo, t)
    N = length(u) / 2
    c = @view u[1:N]
    s = @view u[N+1:end]
    v = sqrt.(c .^ 2 + s .^ 2 .+ 1/2) ./ dopo.amp
    du = vcat(v, v)
end

function evolve_degenerate_oscillators(dopo::OpticalOscillators{T})  where T <: Real
    sol = solve(SDEProblem(coherence!, noise!, dopo.initial_state, dopo.time, dopo))
    N = length(sol.u) / 2
    Int.(sign.(sol.u[end][N+1:end]))
end

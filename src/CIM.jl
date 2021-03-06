# CIM.jl: different algorithms to simulate coherent Ising machines.

export
       OpticalOscillators,
       OPODynamics,
       evolve_optical_oscillators,
       noisy_mean_field_annealing

@inline zeros_like(x::AbstractArray) = zeros(eltype(x), size(x))

# Optical Parametric Oscillator (OPO)
# Based on https://arxiv.org/pdf/1901.08927.pdf
"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct OpticalOscillators{T <: Real}
    ig::IsingGraph
    scale::T
    noise::Distributions.Sampleable
end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct OPODynamics{T <: Real}
    initial_state::Vector{T}
    saturation::T
    pump::Vector{T}
    momentum::T
end

"""
$(TYPEDSIGNATURES)

"""
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
"""
$(TYPEDSIGNATURES)

"""
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
"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct DegenerateOscillators{T <: Real}
    ig::IsingGraph
    pump::T
    saturation::T
end

"""
$(TYPEDSIGNATURES)

"""
function evolve_degenerate_oscillators()
    # To be written
end

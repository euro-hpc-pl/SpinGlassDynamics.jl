export
       OpticalOscillators,
       OPODynamics,
       evolve_optical_oscillators

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

# Degenerate Optical Parametric Oscillator (DOPO)
# Based on https://www.nature.com/articles/s41467-018-07328-1
struct DegenerateOscillators{T <: Real}
    ig::IsingGraph
    pump::T
    saturation::T
end

function evolve_degenerate_oscillators()
end

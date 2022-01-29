export
       OpticalOscillators,
       OPODynamics,
       digitize_state,
       evolve_optical_oscillators

# Optical Parametric Oscillator (OPO)
# Based on https://arxiv.org/pdf/1901.08927.pdf
struct OpticalOscillators{T <: Real}
    ig::IsingGraph
    scale::T
    noise::Vector{T}
end
struct OPODynamics{T <: Real}
    initial_state::Vector{T}
    saturation::T
    pump::Vector{T}
    momentum::T
end

function digitize_state(state::Vector{<:Real})
    Dict(i => σ < 0 ? -1 : 1 for (i, σ) ∈ enumerate(state))
end

function evolve_optical_oscillators(
    opo::OpticalOscillators{T},
    dyn::OPODynamics{T}
)  where T <: Real
    J, h = couplings(opo.ig), biases(opo.ig)
    x = dyn.initial_state
    m_old = zeros(length(x))
    for p ∈ dyn.pump
        Δx = p .* x .+ opo.scale .* (J * x .+ h) .+ opo.noise
        m = (1.0 - dyn.momentum) .* Δx + dyn.momentum .* m_old
        x .+= m .* (abs.(x .+ m) .< dyn.saturation)
        m_old = m
    end
    x
end

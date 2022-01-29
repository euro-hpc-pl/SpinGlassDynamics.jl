export OpticalOscillators,
       OPODynamics,
       activation,
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

activation(x::T, xsat::T=1.0) where T <: Real = abs(x) <= xsat ? x : xsat

function digitize_state(state::Vector{<:Real})
    Dict(i => σ < 0 ? -1 : 1 for (i, σ) ∈ enumerate(state))
end

# This is naive version just to see if the paper does not lie
function __evolve_optical_oscillators(
    opo::OpticalOscillators{T},
    dyn::OPODynamics{T}
)  where T <: Real
    N = length(dyn.initial_state)
    x = dyn.initial_state
    for p ∈ dyn.pump
        Δx = p .* x .+ opo.noise
        for i ∈ 1:N
            if has_vertex(opo.ig, i) Δx[i] += get_prop(opo.ig, i, :h) end
            for j ∈ 1:N
                if has_edge(opo.ig, i, j)
                    Δx[i] += opo.scale * get_prop(opo.ig, i, j, :J) * x[j]
                end
            end
        end
        x = activation.(x .+ Δx, Ref(dyn.saturation))
    end
    x
end

# This is less naive version ...
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

# Degenerate Optical Parametric Oscillator (DOPO)
# Based on https://www.nature.com/articles/s41467-018-07328-1
struct DegenerateOscillators{T <: Real}
    ig::IsingGraph
    pump::T
    saturation::T
end

function evolve_degenerate_oscillators()
end

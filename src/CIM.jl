export OpticalOscillators,
       OPODynamics,
       evolve_optical_oscillators

# Optical Parametric Oscillator (OPO)
# Based on https://arxiv.org/pdf/1901.08927.pdf
struct OpticalOscillators{T <: Real}
    ig::IsingGraph
    pump::T
    scale::T
    noise::Vector{T}
end
struct OPODynamics{T <: Real}
    initial_state::Vector{T}
    saturation::T
    total_time::Int
end

activation(x::T, xsat::T) where T <: Real = abs(x) < xsat ? x : xsat

function evolve_optical_oscillators(
    opo::OpticalOscillators{T},
    dyn::OPODynamics{T}
)  where T <: Real
    N = length(dyn.initial_state)
    x = dyn.initial_state
    for _ ∈ 1:dyn.total_time
        Δx = opo.pump .* x .+ opo.noise
        for i ∈ 1:N, j ∈ 1:N
            if has_edge(opo.ig, i, j)
                J = get_prop(opo.ig, i, j :J)
                Δx[i] += opo.scale * J * x[j]
            end
        end
        x = activation.(Δx, Ref(dyn.saturation))
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

export OpticalOscillators,
       OPODynamics,
       evolve

struct OpticalOscillators{T <: Real}
    μ::T,
    ξ::T,
    f::Vector{T}
end
struct OPODynamics{T <: Real}
    x0::Vector{T},
    xsat::T,
    total_time::Int
end

activation(x::T, xsat::T) where T <: Real = abs(x) < xsat ? x : xsat

# Based on https://arxiv.org/pdf/1901.08927.pdf
function evolve_optical_oscillators(
    ig::IsingGraph,
    opo::OpticalOscillators{T},
    dyn::OPODynamics{T}
)  where T <: Real
    N = length(dyn.x0)
    x = dyn.x0
    for _ ∈ 1:dyn.total_time
        Δx = opo.μ .* x .+ opo.f
        for i ∈ 1:N, j ∈ 1:N
            if has_edge(ig, i, j)
                Δx[i] += opo.ξ * get_prop(ig, i, j :J) * x[j]
            end
        end
        x = activation.(Δx, Ref(dyn.xsat))
    end
    x
end

# Based on https://www.nature.com/articles/s41467-018-07328-1
function degenerate_optical_parametric_oscillators()
end

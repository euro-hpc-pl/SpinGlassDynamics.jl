export naive_coherent_Ising_machine

activation(x::T, xsat::T) where T <: Real = abs(x) < xsat ? x : xsat

# https://arxiv.org/pdf/1901.08927.pdf
function naive_coherent_Ising_machine(
    ig::IsingGraph,
    x0::Vector{T},
    μ::T,
    ξ::T,
    f::Vector{T},
    xsat::T,
    total_time::Int
)  where T <: Real
    N = length(x0)
    x = x0
    for _ ∈ 1:total_time
        Δx = μ .* x .+ f
        for i ∈ 1:N, j ∈ 1:N
            if has_edge(ig, i, j)
                Δx[i] += ξ * get_prop(ig, i, j :J) * x[j]
            end
        end
        x = activation.(Δx, Ref(xsat))
    end
    x
end

# https://www.nature.com/articles/s41467-018-07328-1

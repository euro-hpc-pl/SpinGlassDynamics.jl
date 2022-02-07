# SB.jl: different algorithms for simulated bifurcation.
export
    KerrlOscillators

# https://www.science.org/doi/pdf/10.1126/sciadv.aav2372

struct KerrlOscillators{T <: Real}
    ig::IsingGraph
    Kerr_coefficient::T
    detuning::Vector{T}
    scale::T
    pump::Function
end

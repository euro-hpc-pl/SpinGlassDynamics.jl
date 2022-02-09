
export
    cuda_kerr_oscillators

function kerr_kernel(states, J, dt, Δ, p, K, ξ, num_steps, num_rep)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = size(J, 2)

    for r ∈ 1:num_rep
        x[idx], y[idx] = 2 * rand() - 1, 2 * rand() - 1
        for i ∈ 1:num_steps
            x[idx] += Δ * y[idx] * dt
            # add sync
            Φ = 0.0
            for j ∈ 1:L Φ += J[idx, j] * x[j] end
            y[idx] -= (K * x[idx] ^ 3 + (Δ - p[i+1]) * x[idx] - ξ * Φ) * dt
        end
        # add sync
        states[r, idx] = Int(sign(x[idx]))
    end
    return
end

function energy_kernel(J, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = size(J, 1)
    en = 0.0
    for k=1:L, l=k+1:L
        @inbounds en += σ[k, idx] * J[k, l] * σ[l, idx]
    end
    energies[idx] = en
    return
end

function cuda_kerr_oscillators(
    kpo::KerrOscillators{T},
    dyn::KPODynamics,
    num_rep = 10000
) where T <: Real
    L = nv(kpo.ig)
    C = -couplings(kpo.ig)

    σ = CUDA.zeros(Int, num_rep, L)
    energies = CUDA.zeros(L)
    J = CUDA.CuArray(C + transpose(C))

    dt = dyn.dt
    Δ = kpo.detuning
    K = kop.kerr_coeff
    ξ = kpo.scale
    steps = dyn.num_steps
    p = [dt * (i - 1) / num_steps / dt for i ∈ 1:dyn.num_steps]

    bl = 16
    th = ceil(Int, L / bl)
    @cuda threads=th blocks=bl kerr_kernel(σ, J, dt, Δ, p, K, ξ, steps, num_rep)

    L = size(σ, 1)
    bl = 16
    th = ceil(Int, L / bl)
    @cuda threads=th blocks=bl energy_kernel(J, energies, σ)

    energies, σ
end

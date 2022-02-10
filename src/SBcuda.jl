
export
    cuda_evolve_kerr_oscillators

function _kerr_kernel(x, y, states, J, pump, fparams, iparams)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = size(J, 2)

    dt, Δ, K, ξ = fparams
    num_steps, num_rep = iparams

    for r ∈ 1:num_rep
        x[idx] = 2 * rand() - 1
        y[idx] = 2 * rand() - 1
        for i ∈ 1:num_steps
            @inbounds x[idx] += Δ * y[idx] * dt
            # add sync
            Φ = 0.0
            for j ∈ 1:L @inbounds Φ += J[idx, j] * x[j] end
            @inbounds y[idx] -= (K * x[idx] ^ 3 + (Δ - pump[i+1]) * x[idx] - ξ * Φ) * dt
        end
        # add sync
        states[idx, r] = Int(sign(x[idx]))
    end
    return
end

function kerr_kernel(a, b, states, J, pump, fparams, iparams)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride_x = gridDim().x * blockDim().x

    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_y = gridDim().y * blockDim().y

    L = size(J, 1)
    dt, Δ, K, ξ = fparams
    num_steps, _ = iparams

    for i ∈ idx:stride_x:L, j ∈ idy:stride_y:num_rep
        a[i, j] = 2 * rand() - 1
        b[i, j] = 2 * rand() - 1
        for k ∈ 1:num_steps
            @inbounds a[i, j] += Δ * b[i, j] * dt
            Φ = 0.0
            for l ∈ 1:L @inbounds Φ += J[i, l] * a[l, j] end
            @inbounds b[i, j] -= dt * (K * a[i, j] ^ 3 + (Δ - pump[k+1]) * a[i, j] - ξ * Φ)
        end
        states[i, j] = Int(sign(a[i, j]))
    end
    return
end

function energy_kernel(J, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    L = size(J, 1)
    for i ∈ idx:stride:length(energies)
        en = 0.0
        for k=1:L, l=k+1:L
            @inbounds en += σ[k, i] * J[k, l] * σ[l, i]
        end
        energies[i] = en
    end
    return
end

function cuda_evolve_kerr_oscillators(
    kpo::KerrOscillators{T},
    dyn::KPODynamics,
    num_rep = 10000
) where T <: Real
    L = nv(kpo.ig)
    C = -couplings(kpo.ig)

    σ = CUDA.zeros(Int, L, num_rep)
    J = CUDA.CuArray(C + transpose(C))

    #x, y = CUDA.zeros(L), CUDA.zeros(L)
    x, y = CUDA.zeros(L, num_rep), CUDA.zeros(L, num_rep)

    fparams = CUDA.CuArray([dyn.dt, kpo.detuning, kpo.kerr_coeff, kpo.scale])
    iparams = CUDA.CuArray([dyn.num_steps, num_rep])
    pump = CUDA.CuArray([kpo.pump(dyn.dt * i) for i ∈ 1:dyn.num_steps+1])

    #th = 256
    #bl = ceil(Int, L / th)

    th = (16, 16)
    bl = (ceil(Int, L / th[1]), ceil(Int, num_rep / th[2]))

    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl kerr_kernel(x, y, σ, J, pump, fparams, iparams)
        end
    end
#=
    th = 16
    bl = ceil(Int, num_rep / th)
    energies = CUDA.zeros(num_rep)
    CUDA.@sync begin
        @cuda threads=th blocks=bl energy_kernel(J, energies, σ)
    end

    minimum(Array(energies))
=#
    @time σ_cpu = Array(σ)
    @time states = [σ_cpu[:, i] for i ∈ 1:size(σ_cpu, 2)]

    @time en = minimum(energy(states, kpo.ig))
    en
end

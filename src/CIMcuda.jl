export
    cuda_evolve_optical_oscillators

"""
This is CUDA kernel to evolve OPO.
"""
function opo_kernel(x, y, m, states, J, h, pump, noise, fparams, iparams)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    x_stride = gridDim().x * blockDim().x

    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    y_stride = gridDim().y * blockDim().y

    L = size(J, 1)
    α, ξ = fparams
    num_steps, num_rep = iparams

    for i ∈ idx:x_stride:L, j ∈ idy:y_stride:num_rep
        for k ∈ 1:num_steps
            Φ = 0.0
            for l ∈ 1:L @inbounds Φ += J[i, l] * x[l, j] + h[i] end
            @inbounds Δx[i, j] = pump[k] * x[i, j] + ξ * Φ + noise[i]
            @inbounds m[i, j] = (1 - α) * Δx[i, j] + α * Δm[i, j]
            @inbounds x[i, j] += m[i, j] * (abs(x[i, j] + m[i, j]) < 1.0)
            @inbounds Δm[i, j] = m[i, j]
        end
        @inbounds states[i, j] = Int(sign(x[i, j]))
    end
    return
end

"""
This is experimental function to run simulations and test ideas.
"""
function cuda_evolve_optical_oscillators(
    kpo::KerrOscillators{T},
    dyn::KPODynamics{T},
    num_rep = 512,
    threads_per_block = (16, 16)
) where T <: Real
    N = nv(kpo.ig)
    L = N + 1

    JK = CUDA.CuArray(kerr_adjacency_matrix(kpo.ig))

    σ = CUDA.zeros(Int, L, num_rep)
    x = CUDA.CuArray(2 .* rand(L, num_rep) .- 1)
    y = CUDA.CuArray(2 .* rand(L, num_rep) .- 1)

    iparams = CUDA.CuArray([dyn.num_steps, num_rep])
    fparams = CUDA.CuArray([dyn.dt, kpo.detuning, kpo.kerr_coeff, kpo.scale])
    pump = CUDA.CuArray([kpo.pump(dyn.dt * (i-1)) for i ∈ 1:dyn.num_steps+1])

    th = threads_per_block
    bl = (ceil(Int, L / th[1]), ceil(Int, num_rep / th[2]))

    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl opo_kernel(x, y, σ, JK, pump, fparams, iparams)
        end
    end

    # energy GPU
    th = prod(threads_per_block)
    bl = ceil(Int, num_rep / th)

    J = CUDA.CuArray(couplings(kpo.ig))
    h = CUDA.CuArray(biases(kpo.ig))
    energies = CUDA.zeros(num_rep)

    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl kerr_energy_kernel(J, h, energies, σ)
        end
    end

    en0 = minimum(Array(energies))

    # energy CPU
    σ_cpu = Array(σ)
    states = [σ_cpu[end, i] * σ_cpu[1:end-1, i] for i ∈ 1:size(σ_cpu, 2)]
    @time en = minimum(energy(states, kpo.ig))

    en, en0
end

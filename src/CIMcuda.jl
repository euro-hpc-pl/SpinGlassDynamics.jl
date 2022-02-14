export
    cuda_evolve_optical_oscillators

# This should be merged with kerr_energy_kernel.
function opo_energy_kernel(J, h, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    L = size(J, 1)
    for i ∈ idx:stride:length(energies)
        en = 0.0
        for k=1:L
            @inbounds en += h[k] * σ[k, i]
            for l=k+1:L @inbounds en += σ[k, i] * J[k, l] * σ[l, i] end
        end
        @inbounds energies[i] = en
    end
    return
end

"""
This is CUDA kernel to evolve optical oscillators (CIM)
"""
function opo_kernel(x, states, J, h, pump, noise, fparams, iparams)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    x_stride = gridDim().x * blockDim().x

    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    y_stride = gridDim().y * blockDim().y

    L = size(J, 1)
    α, ξ, sat = fparams
    num_steps, num_rep = iparams

    for i ∈ idx:x_stride:L, j ∈ idy:y_stride:num_rep
        Δm = 0.0
        for k ∈ 1:num_steps
            Φ = 0.0
            for l ∈ 1:L @inbounds Φ += J[i, l] * x[l, j] + h[i] end
            @inbounds Δx = pump[k] * x[i, j] + ξ * Φ + noise[i, j]
            m = (1.0 - α) * Δx + α * Δm
            @inbounds x[i, j] += m * (abs(x[i, j] + m) < sat)
            Δm = m
        end
        @inbounds states[i, j] = Int(sign(x[i, j]))
    end
    return
end

"""
This is experimental function to run simulations and test ideas.
"""
function cuda_evolve_optical_oscillators(
    opo::OpticalOscillators{T},
    dyn::OPODynamics{T},
    num_rep = 512,
    threads_per_block = (16, 16)
) where T <: Real
    L = nv(opo.ig)

    C = couplings(opo.ig)
    h = CUDA.CuArray(biases(opo.ig))
    JO = CUDA.CuArray(-(C + transpose(C)))

    σ = CUDA.zeros(Int, L, num_rep)
    x = CUDA.CuArray(2 .* rand(L, num_rep) .- 1)

    iparams = CUDA.CuArray([length(dyn.pump), num_rep])
    fparams = CUDA.CuArray([dyn.momentum, opo.scale, dyn.saturation])

    noise = CUDA.CuArray(rand(opo.noise, L, num_rep))
    pump = CUDA.CuArray(dyn.pump)

    th = threads_per_block
    bl = (ceil(Int, L / th[1]), ceil(Int, num_rep / th[2]))

    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl opo_kernel(
                x, σ, JO, h, pump, noise, fparams, iparams
        )
        end
    end

    # energy GPU
    th = prod(threads_per_block)
    bl = ceil(Int, num_rep / th)

    J = CUDA.CuArray(C)
    energies = CUDA.zeros(num_rep)
    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl opo_energy_kernel(J, h, energies, σ)
        end
    end

    en0 = minimum(Array(energies))

    # energy CPU
    σ_cpu = Array(σ)
    states = [σ_cpu[:, i] for i ∈ 1:size(σ_cpu, 2)]
    @time en = minimum(energy(states, opo.ig))

    en, en0
end

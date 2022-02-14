export
    cuda_evolve_optical_oscillators

"""
This is CUDA kernel to evolve OPO.
"""
function opo_kernel(x, states, J, h, pump, noise, fparams, iparams)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    x_stride = gridDim().x * blockDim().x

    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    y_stride = gridDim().y * blockDim().y

    L = size(J, 1)
    α, ξ = fparams
    num_steps, num_rep = iparams

    for i ∈ idx:x_stride:L, j ∈ idy:y_stride:num_rep
        Δm = 0.0
        for k ∈ 1:num_steps
            Φ = 0.0
            for l ∈ 1:L @inbounds Φ += J[i, l] * x[l, j] + h[i] end
            @inbounds Δx = pump[k] * x[i, j] + ξ * Φ + noise[i, j]
            @inbounds m = (1 - α) * Δx + α * Δm
            @inbounds x[i, j] += m * (abs(x[i, j] + m) < 1.0)
            @inbounds Δm = m
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

    J = CUDA.CuArray(couplings(opo.ig))
    h = CUDA.CuArray(biases(opo.ig))
    JO = -(J + transpose(J))

    σ = CUDA.zeros(Int, L, num_rep)
    x = CUDA.CuArray(2 .* rand(L, num_rep) .- 1)
    y = CUDA.CuArray(2 .* rand(L, num_rep) .- 1)

    iparams = CUDA.CuArray([dyn.num_steps, num_rep])
    fparams = CUDA.CuArray([α, opo.scale])

    noise = CUDA.CuArray(rand(opo.noise. L, num_rep))
    pump = CUDA.CuArray([kpo.pump(dyn.dt * (i-1)) for i ∈ 1:dyn.num_steps+1])

    th = threads_per_block
    bl = (ceil(Int, L / th[1]), ceil(Int, num_rep / th[2]))

    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl opo_kernel(
                x, y, σ, JO, pump, noise, fparams, iparams
            )
        end
    end

    # energy GPU
    th = prod(threads_per_block)
    bl = ceil(Int, num_rep / th)

    energies = CUDA.zeros(num_rep)
    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl kerr_energy_kernel(J, h, energies, σ)
        end
    end

    en0 = minimum(Array(energies))

    # energy CPU
    σ_cpu = Array(σ)
    states = [σ_cpu[:, i] for i ∈ 1:size(σ_cpu, 2)]
    @time en = minimum(energy(states, kpo.ig))

    en, en0
end

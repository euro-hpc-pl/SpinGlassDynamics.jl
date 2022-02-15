export
    energy_kernel,
    cuda_evolve_kerr_oscillators

"""
This is CUDA kernel to evolve Kerr oscillators.
It threads over both the system size and repetitions.
There is room for improvement although it works quite well.
"""
function kerr_kernel(x, states, J, h, pump, fparams, iparams)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    x_stride = gridDim().x * blockDim().x

    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    y_stride = gridDim().y * blockDim().y

    L = size(J, 1)
    dt, Δ, K, ξ = fparams
    num_steps, num_rep = iparams

    for i ∈ idx:x_stride:L, j ∈ idy:y_stride:num_rep
        y = 0.0
        # This uses symplectic Euler method (different variations are possible)
        for k ∈ 1:num_steps
            Φ = h[i]
            # TODO consider: x[l, j] <-> sign(x[l, j])
            for l ∈ 1:L @inbounds Φ += J[i, l] * x[l, j] end

            # TODO consider syncing here
            #CUDA.sync_threads()
            # TODO adding noise [~W_i * sqrt(dt)] should produce behaviour similar to CIM

            # TODO consider also using this instead:
            @inbounds y -= (K * x[i, j] ^ 3 + (Δ - pump[k]) * x[i, j] + ξ * Φ) * dt
            #@inbounds y -= ((Δ - pump[k]) * x[i, j] - ξ * Φ) * dt

            @inbounds x[i, j] += Δ * y * dt

            # inelastic walls at +/- 1
            if abs(x[i, j]) > 1.0
                @inbounds x[i, j] = sign(x[i, j])
                @inbounds y = 0.0

                # TODO consider this instead:
                #@inbounds y = 2.0 * rand() - 1.0
            end
        end
        @inbounds states[i, j] = Int(sign(x[i, j]))
    end
    return
end

"""
This is CUDA kernel to compute energies from states.
"""
function energy_kernel(J, h, energies, σ)
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
This is experimental function to run simulations and test ideas.
"""
function cuda_evolve_kerr_oscillators(
    kpo::KerrOscillators{T},
    dyn::KPODynamics{T},
    num_rep = 512,
    threads_per_block = (16, 16)
) where T <: Real

    C, b = couplings(kpo.ig), biases(kpo.ig)
    C += transpose(C)
    L = size(C, 1)

    σ = CUDA.zeros(Int, L, num_rep)
    x = CUDA.CuArray(2 .* rand(L, num_rep) .- 1)
    J, h = CUDA.CuArray(C),  CUDA.CuArray(b)

    iparams = CUDA.CuArray([dyn.num_steps, num_rep])
    fparams = CUDA.CuArray([dyn.dt, kpo.detuning, kpo.kerr_coeff, kpo.scale])
    pump = CUDA.CuArray([kpo.pump(dyn.dt * (i-1)) for i ∈ 1:dyn.num_steps+1])

    th = threads_per_block
    bl = (ceil(Int, L / th[1]), ceil(Int, num_rep / th[2]))

    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl kerr_kernel(
                x, σ, J, h, pump, fparams, iparams
            )
        end
    end

    # energy GPU
    th = prod(threads_per_block)
    bl = ceil(Int, num_rep / th)

    energies = CUDA.zeros(num_rep)
    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl energy_kernel(J, h, energies, σ)
        end
    end

    en0 = minimum(Array(energies))

    # energy CPU
    σ_cpu = Array(σ)
    states = [σ_cpu[:, i] for i ∈ 1:size(σ_cpu, 2)]
    @time en = minimum(energy(states, kpo.ig))

    en, en0
end

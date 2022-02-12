export
    cuda_evolve_kerr_oscillators

"""
This is CUDA kernel to evolve Kerr oscillators.
It threads over system (Ising) size and repetitions.
There is room for improvement although is works quite well.
"""
function kerr_kernel(a, b, states, J, pump, fparams, iparams)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = gridDim().x * blockDim().x

    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = gridDim().y * blockDim().y

    L = size(J, 1)
    dt, Δ, K, ξ = fparams
    num_steps, num_rep = iparams

    for i ∈ idx:xstride:L, j ∈ idy:ystride:num_rep
        a[i, j] = 2 * rand() - 1
        b[i, j] = 2 * rand() - 1

        for k ∈ 1:num_steps
            Φ = 0.0
            # TODO consider: a[l, j] <-> sign(a[l, j])
            for l ∈ 1:L @inbounds Φ += J[i, l] * a[l, j] end

            # TODO: consider syncing here
            #sync_threads()

            @inbounds b[i, j] -= (K * a[i, j] ^ 3 + (Δ - pump[k]) * a[i, j] - ξ * Φ) * dt
            # TODO consider also using this instead:
            # @inbounds b[i, j] -= ((Δ - pump[k]) * a[i, j] - ξ * Φ) * dt

            @inbounds a[i, j] += Δ * b[i, j] * dt

            # inelastic walls at +/- 1
            if abs(a[i, j]) > 1
                @inbounds a[i, j] = sign(a[i, j])
                @inbounds b[i, j] = 2 * rand() - 1
                # TODO consider this instead:
                #@inbounds b[i, j] = 0.0
            end
        end
        @inbounds states[i, j] = Int(sign(a[i, j]))
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
            @inbounds en += h[k] * σ[k]
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
    dyn::KPODynamics,
    num_rep = 512,
    threads_per_block = (16, 16)
) where T <: Real
    L = nv(kpo.ig)
    C = couplings(kpo.ig)

    # TODO add capability to solve with biases
    #h = biases(kpo.ig)
    #@assert h ≈ zeros(L)

    σ = CUDA.zeros(Int, L, num_rep)
    J = CUDA.CuArray(-C - transpose(C))
    h = CUDA.CuArray(biases(kpo.ig))

    x = CUDA.zeros(L, num_rep)
    y = CUDA.zeros(L, num_rep)

    iparams = CUDA.CuArray([dyn.num_steps, num_rep])
    fparams = CUDA.CuArray(
        [dyn.dt, kpo.detuning, kpo.kerr_coeff, kpo.scale]
    )
    pump = CUDA.CuArray(
        [kpo.pump(dyn.dt * (i-1)) for i ∈ 1:dyn.num_steps+1]
    )

    th = threads_per_block
    bl = (ceil(Int, L / th[1]), ceil(Int, num_rep / th[2]))

    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl kerr_kernel(x, y, σ, J, pump, fparams, iparams)
        end
    end

    # energy GPU
    th = prod(threads_per_block)
    bl = ceil(Int, num_rep / th)
    energies = CUDA.zeros(num_rep)
    J = CUDA.CuArray(C)
    h = CUDA.CuArray(biases(kpo.ig))

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

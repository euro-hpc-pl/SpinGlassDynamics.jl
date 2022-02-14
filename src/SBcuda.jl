export
    kerr_adjacency_matrix,
    cuda_evolve_kerr_oscillators

"""
Creates the adjacency matrix for the Ising graph with an auxiliary spin.
"""
function kerr_adjacency_matrix(ig::IsingGraph)
    C, b = couplings(ig), biases(ig)
    L = size(C, 1)
    J = zeros(L+1, L+1)
    J[1:L, 1:L] = C
    J[1:L, end] = b
    J += transpose(J)
    -J
end

"""
This is CUDA kernel to evolve Kerr oscillators.
It threads over system (Ising) size and repetitions.
There is room for improvement although is works quite well.
"""
function kerr_kernel(x, states, J, pump, fparams, iparams)
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
            Φ = 0.0
            # TODO consider: x[l, j] <-> sign(x[l, j])
            for l ∈ 1:L @inbounds Φ += J[i, l] * x[l, j] end

            # TODO consider syncing here
            #sync_threads()
            # TODO adding noise [~W_i * sqrt(dt)] should produce behaviour similar to CIM

            @inbounds y -= (K * x[i, j] ^ 3 + (Δ - pump[k]) * x[i, j] - ξ * Φ) * dt
            # TODO consider also using this instead:
            #@inbounds y -= ((Δ - pump[k]) * x[i, j] - ξ * Φ) * dt

            @inbounds x[i, j] += Δ * y * dt

            # inelastic walls at +/- 1
            if abs(x[i, j]) > 1.0
                @inbounds x[i, j] = sign(x[i, j])
                @inbounds y = 0.0

                # TODO consider this instead:
                #@inbounds y = 2 * rand() - 1
            end
        end
        @inbounds states[i, j] = Int(sign(x[i, j]))
    end
    return
end

"""
This is CUDA kernel to compute energies from states.
Takes into account embedding.
"""
function kerr_energy_kernel(J, h, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    L = size(J, 1)
    for i ∈ idx:stride:length(energies)
        en = 0.0
        for k=1:L
            @inbounds en += h[k] * σ[k, i] * σ[end, i]
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
    N = nv(kpo.ig)
    L = N + 1

    JK = CUDA.CuArray(kerr_adjacency_matrix(kpo.ig))

    σ = CUDA.zeros(Int, L, num_rep)
    x = CUDA.CuArray(2 .* rand(L, num_rep) .- 1)

    iparams = CUDA.CuArray([dyn.num_steps, num_rep])
    fparams = CUDA.CuArray([dyn.dt, kpo.detuning, kpo.kerr_coeff, kpo.scale])
    pump = CUDA.CuArray([kpo.pump(dyn.dt * (i-1)) for i ∈ 1:dyn.num_steps+1])

    th = threads_per_block
    bl = (ceil(Int, L / th[1]), ceil(Int, num_rep / th[2]))

    @time begin
        CUDA.@sync begin
            @cuda threads=th blocks=bl kerr_kernel(x, σ, JK, pump, fparams, iparams)
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

# [1] https://www.pnas.org/content/pnas/117/43/26639.full.pdf
# [2] https://www.pnas.org/content/pnas/suppl/2020/10/11/2015192117.DCSupplemental/pnas.2015192117.sapp.pdf

using MetaGraphs

@testset "Test simple Ising ODE" begin

   function ising_ODE!(du, u, p, t)
      κ, κ2, α, J = p
      n = length(α)
      x = view(u, 1:n)
      dx = view(du, 1:n)
      γ = view(u, n+1:2n)
      dγ = view(du, n+1:2n)

      dx[:] = 2κ * ((-α .+ γ) .* x - J * x)
      dγ[:] = κ2 * (1 .- x .^ 2)
   end


   instance = "$(@__DIR__)/instances/basic/4_001.txt" 

   ig = ising_graph(instance)
   J = get_prop(ig, :J)
   h = get_prop(ig, :h)
   n = length(h)
   u0 = 2 * ((rand(2*n) .> 0.5) .- 0.5)
   κ = 0.1
   κ2 = 0.2
   p = (κ, κ2, h, J)
   time = (0.0, 10000.0)
   prob = ODEProblem(ising_ODE!, u0, time, p)
   sol = solve(prob)

   using Plots
   plot(sol)
   savefig("sol.png")
end
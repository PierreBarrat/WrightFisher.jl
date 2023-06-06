begin
	using Pkg; Pkg.activate(".")
	using BenchmarkTools
	using Profile, ProfileView
	using WrightFisher
end

begin
	N = 25_000
	μ = 1/N/5

	L = 10

	# partial sweep
	β = .3
	ρ = .01
	Ne = 1/ρ/β^2
	T = round(Int, 10*Ne)
	# T = 100

	s = 0.05
    H = s*ones(Float64, L)
	α = -2*s/log(1-β)
end

Δt = Int(round(Int, .1/ρ)) # sample frequencies 10 times faster than ρ

out = let
	fitness_landscape = WF.ExpiringFitness(; α, L, H)
	pop = WF.Pop(fitness_landscape; N, L, μ)
	out = WF.Tools.evolve_sample!(pop, 100, Δt, (div = WF.diversity,); switchgen = 1/ρ)
end
# compile run
let
	fitness_landscape = WF.ExpiringFitness(; α, L, H)
	pop = WF.Pop(fitness_landscape; N, L, μ)
	frequencies, _, _ = WF.Tools.evolve_sample_freqs!(pop, 100, Δt; switchgen = 1/ρ)
end;



begin
	fitness_landscape = WF.ExpiringFitness(; α, L, H)
	pop = WF.Pop(fitness_landscape; N, L, μ)
	Profile.clear()
	@profile frequencies, _, _ = WF.Tools.evolve_sample_freqs!(pop, T, Δt; switchgen = 1/ρ)
end;

@btime let
	fitness_landscape = WF.ExpiringFitness(; α, L, H)
	pop = WF.Pop(fitness_landscape; N, L, μ)
	frequencies, _, _ = WF.Tools.evolve_sample_freqs!(pop, T, Δt; switchgen = 1/ρ)
end;

# ProfileView.view()

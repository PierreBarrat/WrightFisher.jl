module Tools

using StochasticExpiringFitness
using StatsBase

"""
	evolve_sample_freqs(
		pop, evtime, Δt;
		N = 100, L = 10, μ = 0.1/L, s = 0.0, fitness = SEF.additive_fitness
	)

Evolve `pop` for `evtime` generations and sample its frequencies every `Δt`.
"""
function evolve_sample_freqs!(
	pop, evtime, Δt;
	fitness = SEF.additive_fitness,
	switchgen = Inf,
)
	freqs = zeros(Float64, div(evtime, Δt) + 1, 2*pop.param.L)
	freqs[1, :] .= SEF.f1(pop)

	ϕ = zeros(Float64, div(evtime, Δt) + 1, 2*pop.param.L)
	ϕ[1,:] .= SEF.get_fitness_vector(pop, fitness)

	SF = zeros(Float64, div(evtime, Δt) + 1, 2*pop.param.L)
	SF[1,:] .= SEF.get_summed_frequencies(pop)

	t = 0
	i = 2
	while t < evtime
		ϕ[i,:] .= SEF.get_fitness_vector(pop, fitness)

		SEF.evolve!(pop, Δt; fitness)
		freqs[i, :] .= SEF.f1(pop)

		t += Δt
		i += 1
		if mod(t, switchgen) == 0
			SEF.change_random_field!(pop)
		end
	end

	return freqs, ϕ
end

"""
	evolve_sample_freqs(
		evtime, Δt;
		N = 100, L = 10, μ = 0.1/L, s = 0.0, α = 0.,
		fitness = SEF.additive_fitness,
		switchgen = Inf,
	)

Evolve a population for `evtime` generations and sample its frequencies every `Δt`.
"""
function evolve_sample_freqs(
	evtime, Δt;
	N = 100, L = 10, μ = 0.1/L, s = 0.0, α = 0.,
	fitness = SEF.additive_fitness,
	switchgen = Inf,
)
	pop = SEF.init(; N, L, μ, s, α)
	return evolve_sample_freqs!(pop, evtime, Δt; fitness, switchgen)
end






end # module

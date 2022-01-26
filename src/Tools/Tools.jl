module Tools

using WrightFisher
using StatsBase

"""
	evolve_sample_freqs(
		pop, evtime, Δt;
		N = 100, L = 10, μ = 0.1/L, s = 0.0, fitness = WF.additive_fitness
	)

Evolve `pop` for `evtime` generations and sample its frequencies every `Δt`.
"""
function evolve_sample_freqs!(
	pop, evtime, Δt;
	switchgen = Inf,
)
	freqs = zeros(Float64, div(evtime, Δt) + 1, 2*pop.param.L)
	freqs[1, :] .= WF.f1(pop)

	ϕ = zeros(Float64, div(evtime, Δt) + 1, 2*pop.param.L)
	ϕ[1,:] .= WF.get_fitness_vector(pop)

	# SF = zeros(Float64, div(evtime, Δt) + 1, 2*pop.param.L)
	# SF[1,:] .= WF.get_summed_frequencies(pop)

	t = 0
	i = 2
	while t < evtime
		ϕ[i,:] .= WF.get_fitness_vector(pop)

		WF.evolve!(pop, Δt)
		freqs[i, :] .= WF.f1(pop)

		t += Δt
		i += 1
		if mod(t, switchgen) == 0
			WF.change_random_field!(pop)
		end
	end

	return freqs, ϕ
end

"""
	evolve_sample_freqs(
		evtime, Δt;
		N = 100, L = 10, μ = 0.1/L, s = 0.0, α = 0.,
		fitness = WF.additive_fitness,
		switchgen = Inf,
	)

Evolve a population for `evtime` generations and sample its frequencies every `Δt`.
"""
function evolve_sample_freqs(
	evtime, Δt;
	N = 100, L = 10, μ = 0.1/L, s = 0.0, α = 0.,
	fitness_type = :additive,
	switchgen = Inf,
)
	# pop = WF.init(; N, L, μ, s, α)
	pop = Pop(; N, L, μ, fitness_type, s, α)
	return evolve_sample_freqs!(pop, evtime, Δt; switchgen)
end






end # module

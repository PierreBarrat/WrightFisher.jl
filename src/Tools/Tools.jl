module Tools

using WrightFisher
using StatsBase

export evolve_sample_freqs, evolve_sample_freqs!, evolve_sample_pop!


"""
	evolve_sample_freqs(
		pop, evtime, Δt;
		switchgen = Inf, kwargs...
	)

Evolve `pop` for `evtime` generations and sample its frequencies every `Δt`. Extra keyword
arguments `kwargs` are passed to `WF.change_random_field!`.
"""
function evolve_sample_freqs!(
	pop, evtime, Δt;
	switchgen = Inf, change_init_field = true, kwargs...
)
	change_init_field && WF.change_random_field!(pop; kwargs...)

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
			i, h = WF.change_random_field!(pop; kwargs...)
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

"""
	evolve_sample_pop!(pop, evtime, Δt, n; burnin = 0)

Evolve `pop` for `evtime` generations and sample `n` genomes every `Δt` generations.
"""
function evolve_sample_pop!(pop, evtime, Δt, n; burnin = 0, format = :onehot)
	nsamples = Int(ceil(evtime / Δt))
	aln = if format == :onehot
		zeros(Int, nsamples*n, 2*pop.param.L)
	else
		zeros(Int, nsamples*n, pop.param.L)
	end
	# Initial evolution
	WF.evolve!(pop, burnin)
	# Evolution
	t = 0
	it = 0
	while t < evtime
		WF.evolve!(pop, Δt)
		X = WF.sample(pop, n; format)
		aln[(it*n + 1):((it+1)*n), :] .= X
		t += Δt
		it += 1
	end

	return aln
end

"""
	evolve_sample_pop!(pop, evtime, Δt; burnin = 0)

Evolve `pop` for `evtime` generations and return a copy of `pop` every `Δt` generations.
Yet untested version -- not sure whether this is useful
"""
function __evolve_sample_pop!(pop, evtime, Δt; burnin = 0, format = :onehot)
	nsamples = Int(ceil(evtime / Δt))
	popsample = Dict()
	# Initial evolution
	WF.evolve!(pop, burnin)
	# Evolution
	t = 0
	it = 0
	while t < evtime
		WF.evolve!(pop, Δt)
		t += Δt
		popsample[t] = deepcopy(pop)
		it += 1
	end

	return popsample
end






end # module

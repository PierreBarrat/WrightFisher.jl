module Tools

using WrightFisher

using Chain
using Distributions
using StatsBase

export evolve_sample_freqs, evolve_sample_freqs!, evolve_sample_pop!

"""
	evolve_sample!(
		pop, evtime, Δt, cb = NamedTuple();
		switchgen = Inf, change_init_field = true, change_field_time = :random, kwargs...
	)

Evolve `pop` while calling callback functions in the named tuple `cb` every `Δt` steps.
Fields in the fitness landscape of `pop` are changed at rate `1/switchgen`. Return
the results of the callback values and the times at which the fitness landscape was changed.

More specifically:
- every `Δt`, for all `(name, f)` in `cb`, store `(name => f(pop))` in a named tuple `cb_vals`;
- at rate `1/switchgen`, change a field in the fitness landscape of `pop` by calling
  `WF.change_random_field!(pop; kwargs...)`; store the corresponding time in an array `switch_times`;
- in between these events, evolve `pop` by calling `WF.evolve!`.

Return `(cb_vals, switch_times)`.

*Note*: to not change the fitness landscape, use `switchgen=Inf` (default).

## Kwargs

- `change_init_field`: if `true`, a field is changed at time `0` in the simulation.
- `change_field_time`: if `:random`, uses an exponentially distributed waiting time
 `switchgen`; if `:periodic`, change exactly every `switchgen`.

Extra keyword arguments are passed to `WF.change_random_field!`.

## Example



"""
function evolve_sample!(
	pop, evtime, Δt, cb = NamedTuple();
	fitness_distribution = nothing,
	switchgen = Inf,
	change_init_field = true,
	change_field_time = :random,
	kwargs...
)

	if haskey(cb, :t)
		@warn "Key `:t` is reserved for time in argument `cb`. `cb[:t]` ignored."
	end
	cb_vals = []
	switch_times = []

	sample_next_switch() = if !isfinite(switchgen)
        Inf
    elseif change_field_time == :random
		@chain switchgen Distributions.Exponential rand round(Int, _) Int
	elseif change_field_time == :periodic
		Int(round(Int, switchgen))
	end

	nh = 0 # Number of changed fields
	if switchgen < Inf && change_init_field
		pos, h = WF.change_random_field!(pop, fitness_distribution; kwargs...)
		nh += !isnothing(pos)
		push!(switch_times, (t=0, pos=pos, h=h))
	end

	t = 0
	push!(cb_vals, run_callbacks(pop, t, cb))

	next_switch = sample_next_switch()
	next_cb = Δt
	while t <= evtime
		event, τ = let
			τ, event = findmin((switch=next_switch, cb=next_cb))
			if event == :switch
				next_switch = sample_next_switch()
				next_cb -= τ
			elseif event == :cb
				next_switch -= τ
				next_cb = Δt
			end
			event, τ
		end

		@debug "Next (event, time)" (event, τ)

		WF.evolve!(pop, τ)
		t += τ

		if event == :cb
			push!(cb_vals, run_callbacks(pop, t, cb))
		elseif event == :switch
			pos, h = WF.change_random_field!(pop, fitness_distribution; kwargs...)
			nh += !isnothing(pos)
			push!(switch_times, (t=t, pos=pos, h=h))
		end
	end
	return cb_vals, switch_times

end


function run_callbacks(pop::Pop, t, cb)
	names = (:t, Iterators.filter(!=(:t), keys(cb))...)
	values = (
		t,
		(@chain cb pairs Iterators.filter(p -> p[1] != :t, _) map(p -> p[2](pop), _))...
	)
	return NamedTuple{names}(values)
end

"""
	evolve_sample_freqs(
		pop, evtime, Δt;
		switchgen = Inf,
		change_init_field = true,
		change_field_time = :random, # :periodic or :random
		change_field_pos = :cyclic, # :cyclic or :random !! NOT IMPLEMENTED !!
		kwargs...
	)

Evolve `pop` for `evtime` generations and sample its frequencies every `Δt`.

Every `switchgen` generation, change the sign of a fitness field at one genome position
using `WF.change_random_field!`.
Extra keyword arguments `kwargs` are passed to `WF.change_random_field!`.
"""
function evolve_sample_freqs!(
	pop, evtime, Δt;
	switchgen = Inf, change_init_field = true, change_field_time = :random, kwargs...
)
	nh = 0 # Number of changed fields
	if switchgen < Inf && change_init_field
		pos, h = WF.change_random_field!(pop; kwargs...)
		nh += !isnothing(pos)
	end

	freqs = zeros(Float64, div(evtime, Δt) + 2, 2*pop.param.L)
	freqs[1, :] .= WF.f1(pop)

	ϕ = zeros(Float64, div(evtime, Δt) + 2, 2*pop.param.L)
	ϕ[1,:] .= WF.get_fitness_vector(pop)

	t = 0
	i = 2
	while t <= evtime
		WF.evolve!(pop, Δt)

		freqs[i, :] .= WF.f1(pop)
		ϕ[i,:] .= WF.get_fitness_vector(pop)

		t += Δt
		i += 1

		# Introducing the possibility of a beneficial mutation
		if change_field_time == :periodic
			if mod(t, switchgen) == 0
				pos, h = WF.change_random_field!(pop; kwargs...)
				nh += !isnothing(pos)
			end
		elseif change_field_time == :random
			for r in rand(Δt)
				if r < 1/switchgen
					pos, h = WF.change_random_field!(pop; kwargs...)
					nh += !isnothing(pos)
				end
			end
		else
			@error "kwarg `change_field_time` must be one of `(:periodic, :random)`; \
				got $(change_field)"
		end
	end

	return freqs, ϕ, nh
end

"""
	evolve_sample_freqs(
		evtime, Δt;
		N = 100, L = 10, μ = 0.1/L, s = 0.0, α = 0.,
		fitness = WF.additive_fitness,
		switchgen = Inf,
		change_init_field = true,
		kwargs...,
	)

Construct and evolve a population for `evtime` generations and sample its frequencies
every `Δt`. Extra keyword arguments `kwargs` are passed to `WF.change_random_field!`.
"""
function evolve_sample_freqs(
	evtime, Δt;
	N = 100, L = 10, μ = 0.1/L, s = 0.0, α = 0.,
	fitness_type = :additive,
	switchgen = Inf,
	change_init_field = true,
	kwargs...
)
	pop = Pop(; N, L, μ, fitness_type, s, α)
	return evolve_sample_freqs!(pop, evtime, Δt; switchgen, change_init_field, kwargs...)
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

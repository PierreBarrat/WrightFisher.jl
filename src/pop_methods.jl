######################################################################
############################# STATISTICS #############################
######################################################################

"""
	frequencies(pop::Pop)
	f1(pop::Pop)

Single site frequencies of `pop`.
"""
function frequencies(pop::Pop)
	f1 = zeros(Float64, 2 * pop.param.L)
	for (id, x) in pairs(pop.genotypes)
		for (i,s) in enumerate(x.seq)
			if s > 0
				f1[2*(i-1) + 1] += pop.counts[id]
			elseif s < 0
				f1[2*(i-1) + 2] += pop.counts[id]
			end
		end
	end
	f1 ./= size(pop)
	return f1
end
f1(pop::Pop) = frequencies(pop)
function f2(pop::Pop)
	f2 = zeros(Float64, 2 * pop.param.L, 2 * pop.param.L)
	for (id, x) in pairs(pop.genotypes)
		for i in 1:length(x.seq), j in i:length(x.seq)
			if x.seq[i] > 0 && x.seq[j] > 0
				f2[2*(i-1) + 1, 2*(j-1) + 1] += pop.counts[id]
			elseif x.seq[i] > 0 && x.seq[j] < 0
				f2[2*(i-1) + 1, 2*(j-1) + 2] += pop.counts[id]
			elseif x.seq[i] < 0 && x.seq[j] > 0
				f2[2*(i-1) + 2, 2*(j-1) + 1] += pop.counts[id]
			elseif x.seq[i] < 0 && x.seq[j] < 0
				f2[2*(i-1) + 2, 2*(j-1) + 2] += pop.counts[id]
			end
		end
	end
	# Making it symmetric
	for i in 1:pop.param.L, j in (i+1):pop.param.L
		for a in 1:2, b in 1:2
			f2[2*(j-1)+b, 2*(i-1)+a] = f2[2*(i-1)+a, 2*(j-1)+b]
		end
	end

	f2 ./= size(pop)
	return f2
end


"""
	sum_frequencies!(pop::Pop)

Update `pop.fitness.integrated_freq`: for the state `s` at each position `i`,
add the frequency of `s` to `integrated_freq` if `s` is favored by the field at `i`.
"""
function sum_frequencies!(pop::Pop{ExpiringFitness})
	N = size(pop)
	for (id,x) in pairs(pop.genotypes)
		for (i,(s,h)) in enumerate(zip(x.seq, pop.fitness.H))
			if h != 0 && sign(h) == sign(s)
				pop.fitness.integrated_freq[i] += pop.counts[id]/N
			end
		end
	end

	return nothing
end

function get_summed_frequencies(pop::Pop{ExpiringFitness})
	SF = zeros(Float64, 2 * pop.param.L)
	for i in 1:Int(length(SF)/2)
		if pop.H[i] > 0
			SF[2*i - 1] = pop.fitness.integrated_freq[i]
			SF[2*i] = 0.
		else
			SF[2*i] = pop.fitness.integrated_freq[i]
			SF[2*i - 1] = 0.
		end
	end

	return SF
end

######################################################################
############################### FITNESS ##############################
######################################################################


"""
	update_fitness!(pop::Pop{ExpiringFitness})

Apply `s(t+1) -= α*s(t)*f` for fitness at all positions.
"""
function update_fitness!(pop::Pop{ExpiringFitness})
	N = size(pop)
	αN = pop.fitness.α/N
	for (id, x) in pairs(pop.genotypes)
		X = pop.counts[id]
		for (i, (s, h)) in enumerate(zip(x.seq, pop.fitness.H))
			if (h > 0 && s > 0) || (h < 0 && s < 0) # ~ h != 0 && sign(h) == sign(s)
				pop.fitness.H[i] -= αN * h * X
			end
		end
	end
end


fields(pop::Pop) = fields(pop.fitness)
function fields(ϕ::FitnessLandscape)
	H = zeros(Float64, 2 * ϕ.L)
	H[1:2:end] .= ϕ.H
	for i in 1:Int(length(H)/2)
		f = H[2*i - 1]
		H[2*i] = -f
	end

	return H
end

couplings(pop::Pop{PairwiseFitness}) = couplings(pop.fitness)
function couplings(ϕ::PairwiseFitness)
	J = zeros(Float64, 2*ϕ.L, 2*ϕ.L)
	for i in 1:ϕ.L, j in (i+1):ϕ.L
		J[2*(i-1) + 1, 2*(j-1) + 1] = ϕ.J[i,j]
		J[2*(i-1) + 2, 2*(j-1) + 2] = ϕ.J[i,j]
		J[2*(i-1) + 1, 2*(j-1) + 2] = -ϕ.J[i,j]
		J[2*(i-1) + 2, 2*(j-1) + 1] = -ϕ.J[i,j]
	end
	J .= J .+ J'

	return J
end



function get_fitness_vector(pop)
	ϕ = zeros(Float64, 2 * pop.param.L)
	for i in 1:pop.param.L
		ϕ[2*i - 1] = fitness(1, i, pop.fitness)
		ϕ[2*i] = fitness(-1, i, pop.fitness)
	end

	return ϕ
end

"""
	change_random_field!(
		pop::Pop;
		epitopes = 1:pop.param.L,
		max_freq = 0.5,
		distribution = nothing,
		set_to_finite_freq = true,
		f0 = 0.02,
	)

Try to change a field in `pop.fitness`.
If a field was changed, return `(i, new_field)`. If no field was changed because none of the
epitope positions matched the conditions, return `(nothing, nothing)`.

## Arguments
- `epitopes`: positions where the fields can be changed
- `max_freq`: a field at position `i` is only changed if `f_i(1-f_i) < max_freq(1-max_freq)`.
  This selects non-variable positions. No effect if `max_freq=0.5`.
- `distribution`: distribution of the fitness effects.
	If `nothing`, only the sign of the field is changed.
	If `::Number`, uses a constant value for new fitness effects.
	Should have support over positive numbers only.
- `set_to_finite_freq`: immediatly set new advantageous mutation to frequency `f0`. Useful
    when mutation rate is 0.
"""
function change_random_field!(pop::Pop; distribution = nothing, kwargs...)
	change_random_field!(pop, distribution; kwargs...)
end
function change_random_field!(pop::Pop, distribution::Number; kwargs...)
	change_random_field!(pop, Dirac(distribution); kwargs...)
end

function change_random_field!(
	pop::Pop, distribution;
	epitopes = 1:pop.param.L,
	max_freq = 0.0,
	set_to_finite_freq = true,
	f0 = 0.02,
)
	f = f1(pop)
	idx = findall(i -> f[2*(i-1)+1] * (1-f[2*(i-1)+1]) <= max_freq*(1-max_freq), epitopes)
	if isempty(idx)
		return nothing, nothing
	else
		i = rand(epitopes[idx])
		σ = f[2*(i-1)+1] > f[2*(i-1)+2] ? 1 : -1 # Is 1 or -1 fixed?
		if isnothing(distribution)
			pop.fitness.H[i] = -σ * abs(pop.fitness.H[i])
		else
			pop.fitness.H[i] = -σ * abs(rand(distribution))
		end

		# introduce the new fit mutation at a frequency f0 in the population
		if set_to_finite_freq #&& f[2*(i-1)+1] * (1-f[2*(i-1)+1]) == 0
			m = false
			while !m
				x = sample(pop, 1, format=:genotype)[1]
				if x.seq[i] == σ
					m = true
					y = mutate_position(x, i)
					push!(pop, y, round(Int, f0*size(pop)))
				end
			end
		end

		return i, pop.fitness.H[i]
	end
end

######################################################################
############################### SAMPLE ###############################
######################################################################

function one_hot(X::Genotype)
	s = zeros(Int8, 2*length(X))
	for (i, x) in enumerate(X.seq)
		if x > 0
			s[2*(i-1) + 1] = 1
		else
			s[2*(i-1) + 2] = 1
		end
	end
	return s
end

"""
	sample(
		pop::Pop, n::Int;
		rng = Xorshifts.Xoroshiro128Plus(), format = :onehot,
	)

Sample `n` genotypes from `pop`.
"""
function sample(
	pop::Pop, n::Int;
	rng = Xorshifts.Xoroshiro128Plus(), format = :onehot,
)
	if format == :onehot
		return sample_onehot(pop, n; rng)
	elseif format == :spin
		return sample_qstate(pop, n; rng)
	elseif format == :qstates
		return Int.(-sample_qstate(pop, n; rng)/2 .+ 1.5)
	elseif format == :genotype
		ids = _sample_ids(pop, n, rng)
		return [pop.genotypes[id] for id in ids]
	else
		@error "Unrecognized sample format"
	end
end

function sample_qstate(pop::Pop, n::Int; rng = Xorshifts.Xoroshiro128Plus())
	aln = zeros(Int8, n, pop.param.L)
	ids = StatsBase.sample(
		rng,
		collect(keys(pop.counts)),
		weights(collect(values(pop.counts))),
		n
	)
	for (i,id) in enumerate(ids)
		aln[i,:] .= pop.genotypes[id].seq
	end

	return aln
end
function sample_onehot(pop::Pop, n::Int; rng = Xorshifts.Xoroshiro128Plus())
	aln = zeros(Int8, n, 2*pop.param.L)
	ids = StatsBase.sample(
		rng,
		collect(keys(pop.counts)),
		weights(collect(values(pop.counts))),
		n
	)
	for (i,id) in enumerate(ids)
		aln[i,:] .= one_hot(pop.genotypes[id])
	end

	return aln
end

function _sample_ids(pop, n, rng = Xorshifts.Xoroshiro128Plus())
	return StatsBase.sample(
		rng,
		collect(keys(pop.counts)),
		weights(collect(values(pop.counts))),
		n
	)
end


######################################################################
############################# DIVERSITY ##############################
######################################################################

"""
	diversity(pop; method=:renyi_entropy, α=1, variable=0.05, positions = 1:pop.param.L)

Only `:renyi_entropy` method is implemented.

## Methods
- `:renyi_entropy`: return the exponential of the Rényi entropy with parameter `α`.
- `:variable_positions`: return the number of genome positions where the frequency of one
  of the states is in the range `[variable, 1-variable]`.
- `:polymorphism`: average of `2x(1-x)` where `x` is the frequency of one of the states,
  over the set of sites `positions` (kwarg).
"""
function diversity(
    pop;
    method=:renyi_entropy, α=1, variable=0.05, positions = 1:pop.param.L
)
	return if method == :renyi_entropy
		exp(renyi_entropy(pop, α))
    elseif method == :polymorphism
        polymorphism(pop, positions)
	elseif method == :variable_positions
		f1 = frequencies(pop)
		count(f1[1:2:end]) do f
			variable < f < 1-variable
		end
	else
		error("Unknown method $method.")
	end
end

function renyi_entropy(pop::Pop, α)
	P = collect(pop.counts) / sum(pop.counts)
	return StatsBase.renyientropy(P, α)
end

function polymorphism(pop::Pop, positions)
    freq = frequencies(pop)
    return 2 * mean(i -> freq[2*i-1]*freq[2*i], positions)
end

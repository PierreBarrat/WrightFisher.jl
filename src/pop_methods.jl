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
		# f1 .+= x.seq * pop.counts[id]
		for (i,s) in enumerate(x.seq)
			if s > 0
				f1[2*(i-1) + 1] += pop.counts[id]
			elseif s < 0
				f1[2*(i-1) + 2] += pop.counts[id]
			end
		end
	end
	# f1 .= (f1 / size(pop)) / 2 .+ 0.5
	# # Adding state -1
	# for i in 1:(Int(length(f1)/2))
	# 	f1[2*i] = 1 - f1[2*i - 1]
	# end
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
	for (id,x) in pairs(pop.genotypes)
		for (i,(s,h)) in enumerate(zip(x.seq, pop.fitness.H))
			if h != 0 && sign(h) == sign(s)
				pop.fitness.integrated_freq[i] += pop.counts[id]/size(pop)
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

fields(pop::Pop) = fields(pop.fitness)
function fields(??::FitnessLandscape)
	H = zeros(Float64, 2 * ??.L)
	H[1:2:end] .= ??.H
	for i in 1:Int(length(H)/2)
		f = H[2*i - 1]
		H[2*i] = -f
	end

	return H
end

couplings(pop::Pop{PairwiseFitness}) = couplings(pop.fitness)
function couplings(??::PairwiseFitness)
	J = zeros(Float64, 2*??.L, 2*??.L)
	for i in 1:??.L, j in (i+1):??.L
		J[2*(i-1) + 1, 2*(j-1) + 1] = ??.J[i,j]
		J[2*(i-1) + 2, 2*(j-1) + 2] = ??.J[i,j]
		J[2*(i-1) + 1, 2*(j-1) + 2] = -??.J[i,j]
		J[2*(i-1) + 2, 2*(j-1) + 1] = -??.J[i,j]
	end
	J .= J .+ J'

	return J
end



function get_fitness_vector(pop)
	?? = zeros(Float64, 2 * pop.param.L)
	for i in 1:pop.param.L
		??[2*i - 1] = fitness(1, i, pop.fitness)
		??[2*i] = fitness(-1, i, pop.fitness)
	end

	return ??
end

# """
# 	change_random_field!(pop::Pop)
# """
# function change_random_field!(pop::Pop)
# 	i = rand(1:length(pop.fitness.H))
# 	pop.fitness.H[i] *= -1
# 	return i, pop.fitness.H[i]
# end

"""
	change_random_field!(
		pop::Pop;
		epitopes = 1:pop.param.L,
		max_freq = 0.5,
		distribution = nothing,
	)

Try to change a field in `pop.fitness`. Only consider `epitopes` position at which the
frequency of one character is lower than `max_freq`. The new field is drawn from \
`distribution` and has the opposite sign of the previous field.
If a field was changed, return `(i, new_field)`. If no field was changed because none of the
epitope positions matched the conditions, return `(nothing, old_field)`.


## Arguments

- `epitopes`: positions where the fields can be changed
- `max_freq`: a field at position `i` is only changed if `f_i(1-f_i) < max_freq(1-max_freq)`.
- `distribution`: distribution of the fitness effects. If `nothing`, only the sign of the \
	field is changed. Should have support over positive numbers only.
"""
function change_random_field!(
	pop::Pop;
	epitopes = 1:pop.param.L,
	max_freq = 0.5,
	distribution = nothing,
)
	f = f1(pop)
	idx = findall(i->f[2*(i-1)+1] * (1-f[2*(i-1)+1]) < max_freq*(1-max_freq), epitopes)
	if isempty(idx)
		return nothing, pop.fitness.H[i]
	else
		i = rand(epitopes[idx])
		?? = f[2*(i-1)+1] > f[2*(i-1)+2] ? 1 : -1 # Is 1 or -1 fixed?
		if isnothing(distribution)
			pop.fitness.H[i] = -?? * pop.fitness.H[i]
		else
			pop.fitness.H[i] = -?? * rand(distribution)
		end
		return i, pop.fitness.H[i]
	end
end
function change_random_field!(
	pop::Pop{ExpiringFitness};
	epitopes = 1:pop.param.L,
	max_freq = 0.5,
	distribution = nothing,
)
	f = f1(pop)
	idx = findall(i->f[2*(i-1)+1] * (1-f[2*(i-1)+1]) < max_freq*(1-max_freq), epitopes)
	if isempty(idx)
		return nothing, nothing
	else
		i = rand(epitopes[idx])
		?? = f[2*(i-1)+1] > f[2*(i-1)+2] ? 1 : -1 # Is 1 or -1 fixed?
		if isnothing(distribution)
			pop.fitness.H[i] = -?? * pop.fitness.H[i]
		else
			pop.fitness.H[i] = -?? * rand(distribution)
		end
		pop.fitness.integrated_freq[i] = 0.
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
	end
end

function sample_qstate(pop::Pop, n::Int; rng = Xorshifts.Xoroshiro128Plus())
	aln = zeros(Int8, n, pop.param.L)
	ids = StatsBase.sample(
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
		collect(keys(pop.counts)),
		weights(collect(values(pop.counts))),
		n
	)
	for (i,id) in enumerate(ids)
		aln[i,:] .= one_hot(pop.genotypes[id])
	end

	return aln
end


######################################################################
############################# DIVERSITY ##############################
######################################################################

"""
	diversity(pop; ??=1, method=:renyi_entropy)

Only `:renyi_entropy` method is implemented.

## Methods
- `:renyi_entropy`: return the exponential of the R??nyi entropy with parameter `??`.
"""
function diversity(pop; ??=1, method=:renyi_entropy)
	P = collect(pop.counts) / sum(pop.counts)
	return exp(StatsBase.renyientropy(P, ??))
end

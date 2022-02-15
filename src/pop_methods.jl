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
	# f1 .= (f1 / pop.N) / 2 .+ 0.5
	# # Adding state -1
	# for i in 1:(Int(length(f1)/2))
	# 	f1[2*i] = 1 - f1[2*i - 1]
	# end
	f1 ./= pop.N
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

	f2 ./= pop.N
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
				pop.fitness.integrated_freq[i] += pop.counts[id]/pop.N
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
function fields(ϕ::FitnessLandscape)
	H = zeros(Float64, 2 * ϕ.L)
	H[1:2:end] .= ϕ.H
	for i in 1:Int(length(fitness)/2)
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

function change_random_field!(pop::Pop)
	i = rand(1:length(pop.fitness.H))
	pop.fitness.H[i] *= -1
	return i, pop.fitness.H[i]
end
function change_random_field!(pop::Pop{ExpiringFitness})
	i = rand(1:length(pop.fitness.H))
	pop.fitness.H[i] *= -1
	pop.fitness.integrated_freq[i] = 0.
	return i, pop.fitness.H[i]
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

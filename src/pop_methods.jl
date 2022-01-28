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
	for (id, x) in pop.genotypes
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
	for (id, x) in pop.genotypes
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
	for (id,x) in pop.genotypes
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

function fields(pop::Pop)
	H = zeros(Float64, 2 * pop.param.L)
	H[1:2:end] .= pop.fitness.H
	for i in 1:Int(length(fitness)/2)
		ϕ = H[2*i - 1]
		H[2*i] = -ϕ
	end

	return H
end
function couplings(pop::Pop{PairwiseFitness})
	J = zeros(Float64, 2*pop.param.L, 2*pop.param.L)
	for i in 1:pop.param.L, for j in (i+1):pop.param.L
		J[2*(i-1) + 1, 2*(j-1) + 1] = pop.fitness.J[i,j]
		J[2*(i-1) + 2, 2*(j-1) + 2] = pop.fitness.J[i,j]
		J[2*(i-1) + 1, 2*(j-1) + 2] = -pop.fitness.J[i,j]
		J[2*(i-1) + 2, 2*(j-1) + 1] = -pop.fitness.J[i,j]
	end
	J .= J + J'

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

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
			f1[2*(i-1) + 1] += s * pop.counts[id]
		end
	end
	f1 .= (f1 / pop.N) / 2 .+ 0.5
	# Adding state -1
	for i in 1:(Int(length(f1)/2))
		f1[2*i] = 1 - f1[2*i - 1]
	end

	return f1
end
f1(pop::Pop) = frequencies(pop)

function fields(pop::Pop)
	fitness = zeros(Float64, 2 * pop.param.L)
	fitness[1:2:end] .= pop.H
	for i in 1:Int(length(fitness)/2)
		ϕ = fitness[2*i - 1]
		fitness[2*i] = -ϕ
	end

	return fitness
end

"""
	sum_frequencies!(pop::Pop)

Update `pop.integrated_freq`: for the state `s` at each position `i`,
add the frequency of `s` to `integrated_freq` if `s` is favored by the field at `i`.
"""
function sum_frequencies!(pop::Pop)
	for (id,x) in pop.genotypes
		for (i,(s,h)) in enumerate(zip(x.seq, pop.H))
			if h != 0 && sign(h) == sign(s)
				pop.integrated_freq[i] += pop.counts[id]/pop.N
			end
		end
	end

	return nothing
end

function get_summed_frequencies(pop::Pop)
	SF = zeros(Float64, 2 * pop.param.L)
	for i in 1:Int(length(SF)/2)
		if pop.H[i] > 0
			SF[2*i - 1] = pop.integrated_freq[i]
			SF[2*i] = 0.
		else
			SF[2*i] = pop.integrated_freq[i]
			SF[2*i - 1] = 0.
		end
	end

	return SF
end

function get_fitness_vector(pop, fitness::Function)
	ϕ = zeros(Float64, 2 * pop.param.L)
	for i in 1:pop.param.L
		ϕ[2*i - 1] = fitness(1, i, pop)
		ϕ[2*i] = fitness(-1, i, pop)
	end

	return ϕ
end

function change_random_field!(pop::Pop)
	i = rand(1:length(pop.H))
	pop.H[i] *= -1
	pop.integrated_freq[i] = 0.
	return i, pop.H[i]
end

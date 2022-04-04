function mutate(x::Genotype, nmut, rng)
	s = copy(x.seq)
	for n in 1:nmut
		i = rand(rng, 1:length(x))
		s[i] = -s[i]
	end

	return Genotype(s, hash(s))
end

function mutate!(pop::Pop, rng = Xorshifts.Xoroshiro128Plus())
	λ = pop.param.μ * pop.param.L
	Z = 0
	ids = collect(keys(pop.genotypes))
	for id in ids
		x = pop.genotypes[id]
		C = Int(floor(pop.counts[id]))
		z = 0
		i = 1
		for i in 1:C
			nm = pois_rand(rng, λ)
			if nm > 0
				y = mutate(x, nm, rng)
				z += 1
				push!(pop, y)
			end
		end
		remove!(pop, x, z)
		Z += z
	end

	return Z
end

function select!(pop::Pop)
	for (id, x) in pairs(pop.genotypes)
		# pop.counts[id] *= 1+fitness(x, pop.fitness)
		pop.counts[id] *= exp(fitness(x, pop.fitness))
	end

	return nothing
end
function select!(pop::Pop{ExpiringFitness})
	sum_frequencies!(pop)
	for (id, x) in pairs(pop.genotypes)
		# pop.counts[id] *= 1+fitness(x, pop.fitness)
		pop.counts[id] *= exp(fitness(x, pop.fitness))
	end

	return nothing
end


function sample!(pop::Pop, rng = Xorshifts.Xoroshiro128Plus())
	for (id, cnt) in pairs(pop.counts)
		pop.counts[id] = pois_rand(rng, cnt) # I should/could make this a multinomial
	end

	for (id,x) in pairs(pop.genotypes)
		if pop.counts[id] == 0.
			delete!(pop, x)
		end
	end

	return pop
end

function normalize!(pop::Pop)
	N = size(pop) / pop.param.N
	for (id, cnt) in pairs(pop.counts)
		pop.counts[id] = pop.counts[id] / N
	end

	return N
end

"""
	evolve!(pop::Pop, n=1)

Evolve `pop` for `n` generations.
"""
function evolve!(pop::Pop, n=1)
	rng = Xorshifts.Xoroshiro128Plus()
	for i in 1:n
		mutate!(pop, rng)
		select!(pop)
		pop.N = size(pop)
		sample!(pop, rng)
		normalize!(pop)
	end

	return nothing
end

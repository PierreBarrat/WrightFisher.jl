function mutate(x::Genotype, nmut)
	s = copy(x.seq)
	for n in 1:nmut
		i = rand(1:length(x))
		s[i] = -s[i]
	end

	return Genotype(s, hash(s))
end

function mutate!(pop::Pop)
	# Expected number of double mutants: 1/2*L^2*μ^2*N
	Z = if 1/2 * (pop.param.μ)^2 * (pop.param.L)^2 * (pop.param.N) < 0.05
		mutate_unique!(pop)
	else
		mutate_exact!(pop)
	end
	return Z
end
function mutate_unique!(pop::Pop)
	# Introducing at most one mutation per sequence
	λ = pop.param.μ * pop.param.L
	Z = 0
	ids = collect(keys(pop.genotypes))
	for id in ids
		x = pop.genotypes[id]
		C = Int(floor(pop.counts[id]))
		z = 0
		i = 1
		for i in 1:C
			if rand() < λ
				y = mutate(x, 1)
				z += 1
				push!(pop, y)
			end
		end
		remove!(pop, x, z)
		Z += z
	end

	return Z
end
function mutate_exact!(pop::Pop)
	λ = pop.param.μ * pop.param.L
	Z = 0
	ids = collect(keys(pop.genotypes))
	for id in ids
		x = pop.genotypes[id]
		C = Int(floor(pop.counts[id]))
		z = 0
		i = 1
		for i in 1:C
			nm = pois_rand(λ)
			if nm > 0
				y = mutate(x, nm)
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
		ϕ = fitness(x, pop.fitness)
		if ϕ != 0
			pop.counts[id] *= exp(fitness(x, pop.fitness))
		end

	end

	return nothing
end
function select!(pop::Pop{ExpiringFitness})
	sum_frequencies!(pop)
	for (id, x) in pairs(pop.genotypes)
		if ϕ != 0
			pop.counts[id] *= exp(fitness(x, pop.fitness))
		end
	end

	return nothing
end


function sample!(pop::Pop)
	for (id, cnt) in pairs(pop.counts)
		pop.counts[id] = pois_rand(cnt) # I should/could make this a multinomial
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
	pop.N = pop.param.N

	return N
end

"""
	evolve!(pop::Pop, n=1)

Evolve `pop` for `n` generations.
"""
function evolve!(pop::Pop, n=1)
	for i in 1:n
		mutate!(pop)
		select!(pop)
		# pop.N = size(pop)
		sample!(pop)
		normalize!(pop)
	end

	return nothing
end

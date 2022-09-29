function mutate(x::Genotype, nmut)
	s = copy(x.seq)
	for n in 1:nmut
		i = rand(1:length(x))
		s[i] = -s[i]
	end

	return Genotype(s, hash(s))
end


function mutate_position(x::Genotype, i::Int)
	s = copy(x.seq)
	s[i] = -s[i]
	return Genotype(s, hash(s))
end

function mutate!(pop::Pop)
	# Expected number of double mutants: 1/2*L^2*μ^2*N
	λ = pop.param.N * pop.param.μ * pop.param.L
	Z = if λ < Inf
		mutate_low!(pop)
	else
		mutate_high!(pop)
	end
	return Z
end

function mutate_low!(pop)
	λ = pop.param.N * pop.param.μ * pop.param.L
	ids = collect(keys(pop.genotypes))

	# Get position and id of genotype of mutations
	Nmuts = pois_rand(λ)
	pos = sort!(rand(1:(pop.param.N*pop.param.L), Nmuts); rev=true)
	muts = map(pos) do x
		i = mod(x-1, pop.param.L) + 1 # need a number in [1,L]
		id_nb = Int((x - i)/pop.param.L)
		(id_nb, i)
	end
	if isempty(muts)
		return muts
	end

	# Utility for loop below
	function is_mut_position(id_cursor, C, muts)
		if isempty(muts)
			return false
		else
			return id_cursor + C > muts[end][1]
		end
	end
	# Iterate over clones until all mutations are introduced
	id_cursor = 0
	for id in ids
		C = Int(floor(pop.counts[id]))
		z = 0 # Number of times we mutated this clone
		while is_mut_position(id_cursor, C, muts)
			# This clone must be mutated
			id_nb, i = pop!(muts) # Retrieve mut info
			y = WF.mutate_position(pop.genotypes[id], i)
			push!(pop, y)
			z += 1
		end
		WF.remove!(pop, pop.genotypes[id], min(z, C)) # `min` to avoid rare issues with low pop clones
		# if no more muts, break
		isempty(muts) && break
		# Done with this clone - increase cursor
		id_cursor += C
	end

	return Nmuts
end
function mutate_high!(pop::Pop)
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
			pop.counts[id] *= exp(ϕ)
		end

	end

	return nothing
end
function select!(pop::Pop{ExpiringFitness})
	sum_frequencies!(pop)
	for (id, x) in pairs(pop.genotypes)
		ϕ = fitness(x, pop.fitness)
		if ϕ != 0
			pop.counts[id] *= exp(ϕ)
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

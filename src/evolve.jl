function mutate(x::Genotype, nmut)
	s = copy(x.seq)
	for n in 1:nmut
		i = rand(1:length(x))
		s[i] = -s[i]
	end

	return Genotype(s, hash(s))
end

function mutate(x::Genotype, nmut, mut_weights)
    s = copy(x.seq)
    for i in sample(1:length(x), mut_weights, nmut)
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
	λ = pop.param.N * sum(pop.param.μ)

	Z = if λ < 250
		mutate_low!(pop)
	else
		mutate_high!(pop)
	end
	return Z
end

function mutate_low!(pop)
	λ = pop.param.N * sum(pop.param.μ)
	ids = collect(keys(pop.genotypes))

	# Get position and id of genotype of mutations
	Nmuts = pois_rand(λ)
    # 1
    muts = Vector{Tuple{Int, Int}}(undef, Nmuts)
    w = weights(pop.param.μ)
    for m in 1:Nmuts
        id = rand(1:pop.param.N) - 1 # id of the genotype (Int)
        i = sample(1:pop.param.L, w)
        muts[m] = (id, i)
    end
    sort!(muts; rev=true)
    # 2
    # pos = sort!(rand(1:(pop.param.N*pop.param.L), Nmuts); rev=true)
    # muts = map(pos) do x
    #     i = mod(x-1, pop.param.L) + 1 # need a number in [1,L]
    #     id_nb = Int((x - i)/pop.param.L)
    #     (id_nb, i)
    # end

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
	λ = sum(pop.param.μ) # average number of mutations in the sequence
    w = weights(pop.param.μ) # weights to pick mutation position from
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
				y = mutate(x, nm, w)
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

function sample_multinomial!(pop::Pop)
	@debug "Sampling with multinomial method. Counts before: $(collect(pop.counts))"
	p = collect(pop.counts) / size(pop)
	new_counts = rand(Multinomial(Int(pop.N), p))
	for (i, (id, cnt)) in enumerate(pairs(pop.counts))
		pop.counts[id] = new_counts[i]
	end

	delete_null_genotypes!(pop)
	@debug "Sampling with multinomial method. Counts after: $(collect(pop.counts))"

	return pop
end
function sample_poisson!(pop::Pop)
	@debug "Sampling with poisson method. Counts before: $(collect(pop.counts))"
	for (id, cnt) in pairs(pop.counts)
		pop.counts[id] = pois_rand(cnt) # I should/could make this a multinomial
	end
	if sum(pop.counts) == 0
		@warn "Sampled an empty population. For small populations, use multinomial sampling instead of Poisson."
	end
	delete_null_genotypes!(pop)
	normalize!(pop)

	@debug "Sampling with poisson method. Counts after: $(collect(pop.counts))"

	return pop
end
function sample!(pop::Pop; method = :free)
	if method == :free
		return length(pop) > 25 ? sample_poisson!(pop) : sample_multinomial!(pop)
	elseif method == :poisson
		return sample_poisson!(pop)
	elseif method == :multinomial
		return sample_multinomial!(pop)
	end
end

function delete_null_genotypes!(pop)
	# does not work like intended but it's probably not too bad
	# ids = findall(id -> pop.counts[id] == 0, keys(pop.genotypes))
	for id in keys(pop.genotypes)
		if pop.counts[id] == 0
			delete!(pop, pop.genotypes[id])
		end
	end

	return nothing
end

function normalize!(pop::Pop)
	N = size(pop) / pop.param.N
	for (id, cnt) in pairs(pop.counts)
		pop.counts[id] /=  N
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
		sample!(pop; method=pop.param.sampling_method)
		@debug "Generation $i - Counts of genomes $(collect(pop.counts)) - Actual pop size $(size(pop))"
		if length(pop) == 0
			@warn "Empty population at generation $i: there was likely an issue somewhere."
		end
	end

	return nothing
end
function evolve!(pop::Pop{ExpiringFitness}, n=1)
	for i in 1:n
		mutate!(pop)
		update_fitness!(pop)
		select!(pop)
		sample!(pop; method=pop.param.sampling_method)
		@debug "Generation $i - Counts of genomes $(collect(pop.counts)) - Actual pop size $(size(pop))"
		if length(pop) == 0
			@warn "Empty population at generation $i: there was likely an issue somewhere."
		end
	end

	return nothing
end

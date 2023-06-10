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

function mutate_position_push!(pop, x::Genotype, i::Int)
    s = x.seq
    s[i] = -s[i]
    new_id = hash(s)
    return if haskey(pop.genotypes, new_id)
        pop.counts[new_id] += 1
        s[i] = -s[i]
        pop.genotypes[new_id]
    else
        new_s = copy(s)
        new_x = Genotype(new_s, new_id)
        insert!(pop.genotypes, new_id, new_x)
        insert!(pop.counts, new_id, 1)
        s[i] = -s[i]
        new_x
    end
end

function mutate!(pop::Pop)
	# Expected number of double mutants: 1/2*L^2*μ^2*N
	λ = pop.param.N * sum(pop.param.μ)

	Z = if 0 < λ < 1001
		mutate_low!(pop)
	elseif λ >= 1001
		mutate_low!(pop)
        # error("FIX `mutate_high!`")
	end
	return Z
end

function mutate_low!(pop)
	λ = pop.param.N * sum(pop.param.μ)
	ids = collect(keys(pop.genotypes))

	# Get position and id of genotype of mutations
	Nmuts = pois_rand(λ)
    # 1
    w = weights(pop.param.μ)
    clone_numbers = rand(1:pop.param.N, Nmuts)
    positions = sample(1:pop.param.L, w, Nmuts)
    muts = [(clone_number = id-1, pos = i) for (id, i) in zip(clone_numbers, positions)]
    sort!(muts; rev=true)

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
    #= !! I think this will never mutate the same genome twice !!  could lead to bugs=#
	id_cursor = 0
	for id in ids
		C = Int(floor(pop.counts[id]))
		z = 0 # Number of times we mutated this clone
		while is_mut_position(id_cursor, C, muts)
			# This clone must be mutated
			id_nb, i = pop!(muts) # Retrieve mut info
            mutate_position_push!(pop, pop.genotypes[id], i)
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

#=BROKEN -- needs to be fixed=#
# function mutate_high!(pop::Pop)
# 	λ = sum(pop.param.μ) # average number of mutations in the sequence
#     w = weights(pop.param.μ) # weights to pick mutation position from
# 	Z = 0
# 	ids = collect(keys(pop.genotypes))
# 	for id in ids
# 		x = pop.genotypes[id]
# 		C = Int(floor(pop.counts[id]))
# 		z = 0
# 		i = 1
# 		for i in 1:C
# 			nm = pois_rand(λ)
# 			if nm > 0
# 				y = mutate(x, nm, w)
# 				z += 1
# 				push!(pop, y)
# 			end
# 		end
# 		remove!(pop, x, z)
# 		Z += z
# 	end

# 	return Z
# end

function select!(pop::Pop)
    @debug "Selecting: fitness of genotypes $(map(g -> fitness(g, pop.fitness), pop.genotypes) |> collect)"
    mean_fitness = 0
	for (id, x) in pairs(pop.genotypes)
        ϕ = fitness(x, pop.fitness)
        mean_fitness += ϕ
		if ϕ != 0
			pop.counts[id] *= exp(ϕ) # - mean_fitness)
		end
	end
    # rescale by mean fitness; avoid trouble with extremely low pop counts
    mean_fitness /= length(pop.genotypes)
    for (id, x) in pairs(pop.genotypes)
        pop.counts[id] *= exp(-mean_fitness)
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
	@debug "Sampling with poisson method. Counts after (not normalized): $(collect(pop.counts))"
	if sum(pop.counts) == 0
		@warn "Sampled an empty population. For small populations, use multinomial sampling instead of Poisson." length(pop)
	end
	delete_null_genotypes!(pop)
	normalize!(pop)


	return pop
end
function sample!(pop::Pop; method = :free)
	if method == :free
		return length(pop) > 25 ? sample_poisson!(pop) : sample_multinomial!(pop)
	elseif method == :poisson
		return sample_poisson!(pop)
	elseif method == :multinomial
		return sample_multinomial!(pop)
    else
        error("Unrecognized sampling method $method. Choose from `:free, :poisson, :multinomial`")
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
		@debug "Generation $i - Counts of genomes $(collect(pop.counts)) - Actual pop size $(sum(pop.counts))"
		if length(pop) == 0 || sum(pop.counts) == 0
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
		if length(pop) == 0 || sum(pop.counts) == 0
			@warn "Empty population at generation $i: there was likely an issue somewhere."
		end
	end

	return nothing
end

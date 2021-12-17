"""
	init(; N = 100, L = 10, μ = 0.1/L, s = 0.0, α = 0.)
"""
function init(; N = 100, L = 10, μ = 0.1/L, s = 0.0, α = 0.)
	return init(N, L, μ, s, α)
end
function init(N, L, μ, s, α)
	param = PopParam(N, L, μ, s, α)
	x = Genotype(L)
	pop = Pop(
		Dict(x.id => x),
		Dict(x.id => N),
		s * ones(L),
		zeros(Float64, L),
		N,
		param
	)
	return pop
end

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
		z = 0
		for i in 1:pop.counts[id]
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

function select!(pop::Pop, fitness = additive_fitness)
	if fitness == expiring_fitness
		sum_frequencies!(pop)
	end
	for (id, x) in pop.genotypes
		pop.counts[id] *= 1+fitness(x, pop)
	end

	return nothing
end


function sample!(pop::Pop, rng = Xorshifts.Xoroshiro128Plus())
	for (id, cnt) in pop.counts
		pop.counts[id] = pois_rand(rng, cnt)
	end
	for (id,x) in pop.genotypes
		if pop.counts[id] == 0.
			delete!(pop, x)
		end
	end

	return pop
end

function normalize!(pop::Pop)
	N = 0
	for cnt in values(pop.counts)
		N += cnt
	end
	N /= pop.param.N
	for (id, cnt) in pop.counts
		pop.counts[id] = pop.counts[id] / N
	end

	return N
end

function evolve!(pop::Pop, n=1; fitness = additive_fitness)
	# ids_array = Vector{UInt}(undef, pop.param.N)
	rng = Xorshifts.Xoroshiro128Plus()
	for i in 1:n
		mutate!(pop, rng)
		select!(pop, fitness)
		sample!(pop, rng)
		normalize!(pop)
	end

	return nothing
end

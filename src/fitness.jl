function additive_fitness(x::Genotype, pop::Pop)
	f = 0
	for (i, xi) in enumerate(x.seq)
		f += additive_fitness(xi, i, pop)
	end
	return f
end
function additive_fitness(xi::Integer, i::Integer, pop)
	if sign(xi) == sign(pop.H[i])
		return abs(pop.H[i])
	else
		return 0.
	end
end


function expiring_fitness(x::Genotype, pop::Pop)
	f = 0
	for (i, xi) in enumerate(x.seq)
		f += expiring_fitness(xi, i, pop)
	end
	return f
end
function expiring_fitness(xi::Integer, i::Integer, pop::Pop)
	if sign(xi) == pop.H[i]
		return pop.H[i] * exp(- pop.param.α*pop.integrated_freq[i])
	else
		return 0.
	end
end

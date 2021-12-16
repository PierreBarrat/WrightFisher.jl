function additive_fitness(x::Genotype, pop::Pop)
	f = 0
	for (i, xi) in enumerate(x.seq)
		f += additive_fitness(xi, i, pop)
	end
	return f
end
function additive_fitness(xi::Integer, i::Integer, pop)
	# if sign(xi) == sign(pop.H[i])
	# end
	fi = sign(xi) == sign(pop.H[i]) ? abs(pop.H[i]) : 0.
	# println(xi, " - ", pop.H[i], " --> ", fi)
	return fi
end


function expiring_fitness(x::Genotype, pop::Pop)
	f = 0
	for (i, xi) in enumerate(x.seq)
		f += expiring_fitness(xi, i, pop)
	end
	return f
end
function expiring_fitness(xi::Integer, i::Integer, pop::Pop)
	# if sign(xi) != sign(pop.H[i])
	# 	return 0.
	# else
	# 	return abs(pop.H[i]) * exp(- pop.param.α*pop.integrated_freq[i])
	# end
	return max(
		0., # minimal fitness at this pos
		pop.H[i]*xi * exp(- pop.param.α*pop.integrated_freq[i])
	)
end

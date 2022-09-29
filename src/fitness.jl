fitness(x::Genotype, pop::Pop) = fitness(x::Genotype, pop.fitness)

function fitness(x::Genotype, ϕ::AdditiveFitness)
	@assert length(x) == ϕ.L "Genotype $(length(x)) and fitness landscape $(ϕ.L) \
	must have the same length"
	f = 0
	for (xi, hi) in zip(x.seq, ϕ.H)
		f += xi*hi
	end
	return f
end
function fitness(xi::Integer, i::Integer, ϕ::AdditiveFitness)
	if xi > 0 && ϕ.H[i] > 0. || xi < 0 && ϕ.H[i] < 0.
		return abs(ϕ.H[i])
	else
		return -abs(ϕ.H[i])
	end
end

function fitness(x::Genotype, ϕ::ExpiringFitness)
	@assert length(x) == ϕ.L "Genotype $(length(x)) and fitness landscape $(ϕ.L) \
	must have the same length"
	f = 0
	for (xi, hi) in zip(x.seq, ϕ.H)
		f += xi * hi * exp(-ϕ.α * ϕ.integrated_freq[i])
	end
	return f
end
function fitness(xi::Integer, i::Integer, ϕ::ExpiringFitness)
	if xi > 0 && ϕ.H[i] > 0. || xi < 0 && ϕ.H[i] < 0.
		return abs(ϕ.H[i]) * exp(-ϕ.α * ϕ.integrated_freq[i])
	else
		return -abs(ϕ.H[i]) * exp(-ϕ.α * ϕ.integrated_freq[i])
	end
end


function fitness(x::Genotype, ϕ::PairwiseFitness)
	@assert length(x) == ϕ.L "Genotype $(length(x)) and fitness landscape $(ϕ.L) \
	must have the same length"
	f = 0.
	for j in 1:length(x.seq)
		if x.seq[j] > 0 && ϕ.H[j] > 0. || x.seq[j] < 0 && ϕ.H[j] < 0.
			f += abs(ϕ.H[j])
		else
			f -= abs(ϕ.H[j])
		end
		for i in (j+1):length(x.seq)
			# J[i,j] > 0 favors same state
			# J[i,j] < 0 favors different states
			if x.seq[i] == x.seq[j]
				f += ϕ.J[i,j]
			else
				f -= ϕ.J[i,j]
			end
		end
	end

	return f
end

fitness(::Genotype, ::NeutralFitness) = 0.
fitness(::Int, ::Int, ::NeutralFitness) = 0.

struct Genotype
	seq::Vector{Int8}
	id::UInt
end
function Genotype(L)
	# seq = rand((Int8(-1), Int8(1)), L)
	seq = ones(Int8, L)
	return Genotype(seq, hash(seq))
end

function Base.isequal(x::Genotype, y::Genotype)
	return x.seq == y.seq
end
Base.:(==)(x::Genotype, y::Genotype) = isequal(x,y)
Base.hash(x::Genotype, h::UInt) = hash(x.seq, h)

Base.length(x::Genotype) = length(x.seq)


struct PopParam
	N::Int # population size
	L::Int # length of genomes
	μ::Float64 # mutation rate (per gen per site)
	s::Float64 # magnitude of fitness effects
	α::Float64 # rate of decay of fitness
end

mutable struct Pop
	genotypes::Dict{UInt, Genotype}
	counts::Dict{UInt, Float64}
	H::Vector{Float64} # length L
	integrated_freq::Vector{Float64} # Summed frequency of the state favored by H
	N :: Float64 # real pop size ( can vary )
	param::PopParam
end

Base.in(x::Genotype, pop::Pop) = haskey(pop.genotypes, x.id)

function Base.push!(pop::Pop, x::Genotype)
	if in(x, pop)
		pop.counts[x.id] += 1.
	else
		pop.genotypes[x.id] = x
		pop.counts[x.id] = 1.
	end
	return pop
end

"""
	delete!(pop::Pop, x::Genotype)

Delete `x` from `pop`.
"""
function delete!(pop::Pop, x::Genotype)
	delete!(pop.genotypes, x.id)
	delete!(pop.counts, x.id)
	return pop
end
"""
	remove!(pop::Pop, x::Genotype, i::Integer)

Remove `x` `i` times from `pop`.
"""
function remove!(pop::Pop, x::Genotype, i)
	@assert in(x, pop) && pop.counts[x.id] >= i "Cannot remove genotype $(x.seq) $(i) times"
	pop.counts[x.id] -= i
	if pop.counts[x.id] == 0.
		delete!(pop, x)
	end
	return pop
end


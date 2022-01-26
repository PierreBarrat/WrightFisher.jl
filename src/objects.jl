######################################################################
############################## GENOTYPE ##############################
######################################################################

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

######################################################################
############################### FITNESS ##############################
######################################################################

abstract type FitnessLandscape end

const fitness_landscape_types = (:additive, :expiring, :pairwise)

mutable struct AdditiveFitness <: FitnessLandscape
	L::Int
	H::Vector{Float64} # H[i] > 0 --> 1 is favored at position i
	s::Float64 # overall magnitude
end
AdditiveFitness(s::Number, L::Int) = AdditiveFitness(L, s * ones(L), s)

mutable struct ExpiringFitness <: FitnessLandscape
	L::Int
	H::Vector{Float64} # H[i] > 0 --> 1 is favored at position i
	integrated_freq::Vector{Float64} # Summed frequency of the state favored by H
	s::Float64 # overall magnitude
	α::Float64 # rate of decay of fitness
end
function ExpiringFitness(s::Number, α::Number, L::Int)
	return ExpiringFitness(L, s*ones(L), zeros(Float64, L), s, α)
end

function init_fitness_landscape(fitness_type, L, s; α = 0.)
	@assert in(fitness_type, fitness_landscape_types) "Unrecognized fitness type\
	 $(fitness_type) - Allowed types $(fitness_landscape_types)."

	if fitness_type == :additive
		α != 0 && @warn "Additive fitness type and non-zero decay rate α=$α. Ignoring α."
		return AdditiveFitness(s, L)
	elseif fitness_type == :expiring
		return ExpiringFitness(s, α, L)
	elseif fitness_type == :pairwise
		return PairwiseFitness(s, L)
	end
end

mutable struct PairwiseFitness <: FitnessLandscape
	L::Int
	H::Vector{Float64}
	J::Matrix{Float64}
	s::Float64
	function PairwiseFitness(L, H, J, s)
		@assert issymmetric(J) "Coupling matrix must be symmetric."
		return new(L, H, J, s)
	end
end
function PairwiseFitness(s::Number, L::Int)
	return PairwiseFitness(L, s * ones(L), s/(L-1)*Symmetric(zeros(L,L)), s)
end

######################################################################
################################ POP #################################
######################################################################


struct PopParam
	N::Int # population size
	L::Int # length of genomes
	μ::Float64 # mutation rate (per gen per site)
end

mutable struct Pop{F<:FitnessLandscape}
	genotypes::Dict{UInt, Genotype}
	counts::Dict{UInt, Float64}
	N::Float64 # real pop size (can vary)
	fitness::F
	param::PopParam
end

"""
	Pop(;
		N = 100,
		L = 10,
		μ = 1 / N / L,
		fitness_type = :additive,
		s = 0.01,
		α = 0.,
	)

Initialize a population.
- `N`: population size
- `L`: genotype length
- `μ`: per-individual and per-length mutation rate.
- `fitness_type`: Implemented `fitness_landscape_types`.
- `s`: overall magnitude of fitness effects
- `α`: fitness decay rate for `:expiring` fitness type.
"""
function Pop(;
	N = 100,
	L = 1,
	μ = 1 / N / L,
	fitness_type = :additive,
	s = 0.,
	α = 0.,
)
	ϕ = init_fitness_landscape(fitness_type, L, s; α)
	return Pop(ϕ; N, L, μ)
end
"""
	Pop(fitness::FitnessLandscape; N = 100, L = 1, μ = 1 / N / L)

Initialize a population using a pre-built fitness landscape.
"""
function Pop(fitness::FitnessLandscape; N = 100, L = fitness.L, μ = 1 / N / L)
	@assert fitness.L == L "Fitness landscape length $(fitness.L) differs from input length $L"
	x = Genotype(L)
	param = PopParam(N, L, μ)
	return Pop(
		Dict(x.id => x), # genotypes
		Dict(x.id => Float64(N)), # counts
		Float64(N),
		fitness, # fitness landscape
		param
	)
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




######################################################################
############################## GENOTYPE ##############################
######################################################################

struct Genotype
	seq::Vector{Int8}
	id::UInt
end
function Genotype(L)
	seq = ones(Int8, L)
	return Genotype(seq, hash(seq))
end
Genotype(seq::Vector{<:Int}) = Genotype(seq, hash(seq))


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
#=
To implement
- `fitness(::Genotype, ϕ::FitnessLandscape)`
=#

"""
	const fitness_landscape_types

`(:additive, :expiring, :neutral, :pairwise)`.
"""
const fitness_landscape_types = (:additive, :expiring, :neutral, :pairwise)

"""
# Summary
```
struct NeutralFitness <: FitnessLandscape end
```

Empty struct to represent neutral evolution.
"""
struct NeutralFitness <: FitnessLandscape end

Base.length(f::NeutralFitness) = 0

"""
# Summary
```
mutable struct AdditiveFitness <: FitnessLandscape
```

# Fields
```
L::Int
H::Vector{Float32} # H[i] > 0 --> 1 is favored at position i
s::Float64 # overall magnitude
```
"""
Base.@kwdef mutable struct AdditiveFitness <: FitnessLandscape
	L::Int
	H::Vector{Float32} # H[i] > 0 --> 1 is favored at position i
end
"""
	AdditiveFitness(s::Number, L::Int)

Create an `AdditiveFitness` landscape with positive fields of magnitude `s`.
"""
AdditiveFitness(s::Number, L::Int) = AdditiveFitness(L, s * ones(L))

Base.length(f::AdditiveFitness) = f.L

"""
# Summary
```
mutable struct ExpiringFitness <: FitnessLandscape
```

# Fields
```
L::Int
H::Vector{Float32} # H[i] > 0 --> 1 is favored at position i
integrated_freq::Vector{Float64} # Summed frequency of the state favored by H
α::Float64 # rate of decay of fitness
```
"""
Base.@kwdef mutable struct ExpiringFitness <: FitnessLandscape
	L::Int
	H::Vector{Float32} # H[i] > 0 --> 1 is favored at position i
	α::Float64 # decay rate of fitness advantage
	integrated_freq::Vector{Float64} = zeros(L) # Summed frequency of the state favored by H
end
"""
	ExpiringFitness(s::Number, α::Number, L::Int, H=s*ones(L))
"""
function ExpiringFitness(s::Number, α::Number, L::Int, H=s*ones(L))
	return ExpiringFitness(L, H, α, zeros(Float64, L))
end

Base.length(f::ExpiringFitness) = f.L

"""
# Summary
```
mutable struct Pairwise <: FitnessLandscape
```

# Fields
```
L::Int
H::Vector{Float32} # H[i] > 0 --> 1 is favored at position i
J::Matrix{Float32}
s::Float64 # overall magnitude
```
"""
Base.@kwdef mutable struct PairwiseFitness <: FitnessLandscape
	L::Int
	H::Vector{Float32}
	J::Matrix{Float32}
	function PairwiseFitness(L, H, J)
		@assert issymmetric(J) "Coupling matrix must be symmetric."
		@assert sum(i->J[i,i], 1:L) == 0 "Coupling matrix must have null diagonal."
		return new(L, H, J)
	end
end
"""
	PairwiseFitness(s::Number, L::Int)

Return `PairwiseFitness` landscape with null couplings and fields equal to `s`.
"""
function PairwiseFitness(s::Number, L::Int)
	return PairwiseFitness(L, s * ones(L), s/(L-1)*Symmetric(zeros(L,L)))
end
"""
	PairwiseFitness(H::Vector{Float64}, J::Matrix{Float64})
"""
function PairwiseFitness(H::Vector{Float64}, J::Matrix{Float64})
	return PairwiseFitness(length(H), H, J)
end

Base.length(f::PairwiseFitness) = f.L


"""
	init_fitness_landscape(fitness_type::Symbol, L, s; α = 0.)

Simple initialization for fitness types in `fitness_landscape_type`.
"""
function init_fitness_landscape(fitness_type, L, s; α = 0.)
	@assert in(fitness_type, fitness_landscape_types) "Unrecognized fitness type\
	 $(fitness_type) - Allowed types $(fitness_landscape_types)."

	if fitness_type == :additive
		α != 0 && @warn "Additive fitness type and non-zero decay rate α=$α. Ignoring α."
		return AdditiveFitness(s, L)
	elseif fitness_type == :expiring
		return ExpiringFitness(s, α, L)
	elseif fitness_type == :neutral
		return NeutralFitness()
	elseif fitness_type == :pairwise
		return PairwiseFitness(s, L)
	end
end

######################################################################
################################ POP #################################
######################################################################


Base.@kwdef struct PopParam
	N::Int # population size
	L::Int # length of genomes
	μ::Float64 # mutation rate (per gen per site)

	sampling_method::Symbol = :free
end

mutable struct Pop{F<:FitnessLandscape}
	genotypes::Dictionary{UInt, Genotype}
	counts::Dictionary{UInt, Float64}
	N::Float64 # real pop size (can vary)
	fitness::F
	param::PopParam
end

"""
	Pop(
		fitness::FitnessLandscape;
		N = 100, L = fitness.L, μ = .2 / N, init=:ones
	)

Initialize a population using a pre-built fitness landscape.
"""
function Pop(
	fitness::FitnessLandscape;
	N = 100, L = length(fitness), μ = .2 / N, init=:ones, sampling_method=:free,
)
	@assert in(init, (:ones, :rand, :random)) "Unrecognized input for `init` kwarg."
	param = PopParam(N, L, μ, sampling_method)
	if init == :ones
		return ones_pop(fitness, param)
	elseif init == :rand || init == :random
		return random_pop(fitness, param)
	else
		@error "Unrecognized input for `init` kwarg."
	end
end
"""
	Pop(;
		N = 100,
		L = 10,
		μ = .2 / N,
		fitness_type = :additive,
		s = 0.01,
		α = 0.,
		init = :ones,
	)

Initialize a population.
- `N`: population size
- `L`: genotype length
- `μ`: per-individual and per-length mutation rate.
- `fitness_type`: Implemented `fitness_landscape_types`.
- `s`: overall magnitude of fitness effects
- `α`: fitness decay rate for `:expiring` fitness type.
- `init`: `:ones` or `:rand`
"""
function Pop(;
	N = 100,
	L = 1,
	μ = .2 / N,
	fitness_type = :additive,
	s = 0.,
	α = 0.,
	init = :ones,
	kwargs...
)
	ϕ = init_fitness_landscape(fitness_type, L, s; α)
	return Pop(ϕ; N, L, μ, init, kwargs...)
end

function ones_pop(fitness, param)
	x = Genotype(param.L)
	return Pop(
		Dictionary([x.id], [x]), # genotypes
		Dictionary([x.id], [Float64(param.N)]), # counts
		Float64(param.N),
		fitness, # fitness landscape
		param
	)
end
function random_pop(fitness, param)
	pop = Pop(
		Dictionary{UInt64, WrightFisher.Genotype}(),
		Dictionary{UInt64, Float64}(),
		Float64(param.N),
		fitness,
		param
	)
	for n in 1:param.N
		push!(pop, Genotype(rand([-1,1], param.L)))
	end
	return pop
end


Base.in(x::Genotype, pop::Pop) = haskey(pop.genotypes, x.id)

function Base.push!(pop::Pop, x::Genotype, n=1.)
	if in(x, pop)
		pop.counts[x.id] += n
	else
		insert!(pop.genotypes, x.id, x)
		insert!(pop.counts, x.id, n)
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

function size(pop::Pop)
	N = 0.
	for cnt in pop.counts
		N += cnt
	end
	return N
end



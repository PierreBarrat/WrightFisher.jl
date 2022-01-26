module StochasticExpiringFitness

using LinearAlgebra
using PoissonRandom
using RandomNumbers
using StatsBase

import Base: ==
import Base: hash, length, push!, in, delete!, show

const SEF = StochasticExpiringFitness
export SEF

export Pop, FitnessLandscape
export init_fitness_landscape

include("objects.jl")
include("fitness.jl")
include("evolve.jl")
include("pop_methods.jl")
include("misc.jl")

include("Tools/Tools.jl")

end # module

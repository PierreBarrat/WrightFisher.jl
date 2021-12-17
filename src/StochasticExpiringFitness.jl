module StochasticExpiringFitness

# using Distributions
using RandomNumbers
using PoissonRandom
using StatsBase

import Base: ==
import Base: hash, length, push!, in, delete!, show

const SEF = StochasticExpiringFitness
export SEF, ST

include("objects.jl")
include("fitness.jl")
include("evolve.jl")
include("pop_methods.jl")
include("misc.jl")

include("Tools/Tools.jl")

end # module

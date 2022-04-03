module WrightFisher

using Dictionaries
using Distributions
using LinearAlgebra
using PoissonRandom
using RandomNumbers
using StatsBase

import Base: ==
import Base: hash, length, push!, in, delete!, size, show
import StatsBase: sample, sample!

const WF = WrightFisher
export WF

export Pop
export AdditiveFitness, ExpiringFitness, PairwiseFitness
export evolve!
export init_fitness_landscape
export sample


include("objects.jl")
include("fitness.jl")
include("evolve.jl")
include("pop_methods.jl")
include("misc.jl")

include("Tools/Tools.jl")


end # module

function Base.show(io::IO, y::MIME"text/plain", pop::Pop)
	d = Dict(pop.genotypes[id].seq => pop.counts[id] for id in keys(pop.counts))
	show(io, y, d)
end
Base.show(pop::Pop) = show(stdout, pop)

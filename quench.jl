#using IterTools
using Random
using SparseArrays
using Combinatorics
using Optim
using JSON
using ArgParse
using DataStructures
using LinearAlgebra

densitybondtypes = [
    [(1,0), (0,1), (-1,1)]
]

spinbondtypes = [
    [(1,0), (0,1), (-1,1),],
    [(1,1), (-1,2), (-2,1),],
    [(2,0), (0,2), (-2,2),],
]

mutable struct OccupiedEmpty
    occupied ::Set{Int}
    empty ::Set{Int}

    function OccupiedEmpty(nsite ::Integer, nptl ::Integer)
        occupied = Set{Int}(1:nptl)
        empty = Set{Int}(nptl+1:nsite)
        return new(occupied, empty)
    end
end


"""
DopedSystem

# Members
- size
- densityhamiltonians: Keep the V terms
- spinhamiltonians: Keep the J1, J2, J3 terms
"""
struct DopedSystem
    size :: Tuple{Int, Int}
    densityhamiltonians ::Vector{SparseMatrixCSC{Int, Int}}
    spinhamiltonians ::Vector{SparseMatrixCSC{Int, Int}}
end

import Base: length, size

length(ds::DopedSystem) = prod(ds.size)
size(ds::DopedSystem) = ds.size

function dopedsystem(n1 ::Integer, n2 ::Integer, nparticle ::Integer)
    #s2i(i1 ::Integer, i2 ::Integer) = sub2ind((n1,n2), i1, i2)
    s2i(i1 ::Integer, i2 ::Integer) = (LinearIndices((n1,n2)))[i1, i2]
    i2s(i ::Integer) = ind2sub((n1, n2), i)

    densityhamiltonians = SparseMatrixCSC{Int, Int}[]
    spinhamiltonians = SparseMatrixCSC{Int, Int}[]

    for bonds in densitybondtypes
        rows = Int[]
        cols = Int[]
        vals = Int[]
        for i1 in 1:n1, i2 in 1:n2
            i = s2i(i1, i2)
            for (d1, d2) in bonds
                j1 = mod1(i1+d1, n1)
                j2 = mod1(i2+d2, n2)
                j = s2i(j1, j2)
                push!(rows, i)
                push!(cols, j)
                push!(vals, 1)
            end
        end
        push!(densityhamiltonians, sparse(rows, cols, vals, n1*n2, n1*n2))
    end

    for bonds in spinbondtypes
        rows = Int[]
        cols = Int[]
        vals = Int[]
        for i1 in 1:n1, i2 in 1:n2
            i = s2i(i1, i2)
            for (d1, d2) in bonds
                j1 = mod1(i1+d1, n1)
                j2 = mod1(i2+d2, n2)
                j = s2i(j1, j2)
                push!(rows, i)
                push!(cols, j)
                push!(vals, 1)
            end
        end
        push!(spinhamiltonians, sparse(rows, cols, vals, n1*n2, n1*n2))
    end

    return DopedSystem((n1, n2), densityhamiltonians, spinhamiltonians)
end


"""
FrozenDopedSystem with the locations built in
"""
struct FrozenDopedSystem
    size ::Tuple{Int, Int}
    densityenergies ::Vector{<:Real}
    spinhamiltonians ::Vector{<:AbstractArray}
end


#=
    loc: [1, 3, 4, 6, 10, ...]
    convertmatrix: (1, 1) -> 1, (3, 2) -> 1, (4, 3) -> 1, (6, 4) -> 1, ...
    
    densityhamiltonians: [(i,j) -> 1, ...]

    =>
    densityenergies

=#
function freezelocation(ds ::DopedSystem, loc::Vector{Int})
    n_ptl = length(loc)
    convertmatrix = sparse(loc, 1:n_ptl, ones(n_ptl), length(ds), n_ptl)
    #densityenergies = Int[sum(h) for h in ds.densityhamiltonians]
    densityenergies = [sum(transpose(convertmatrix) * h * convertmatrix) for h in ds.densityhamiltonians]
    spinhamiltonians = [transpose(convertmatrix) * h * convertmatrix for h in ds.spinhamiltonians]
    return FrozenDopedSystem(ds.size, densityenergies, spinhamiltonians)
end


"""
    loc2vec

# Arguments
- ds: System
- loc ::Vector{Int} : locations of the iron ions [i1, i2, i3, ..., iN]
- spins ::Vector{<:Real} : spins [S1, S2, S3, ..., SN]

# Resturn
- sparse vector [ 0, ..., 0, S1, ..., S4, ... ]
"""
function loc2vec(ds:: DopedSystem, loc ::Vector{Int}, spins ::Vector{R}) where {R <: Real}
    cols = Int[]
    vals = Int[]
    for (o, s) in zip(loc, spins)
        push!(cols, o)
        push!(vals, s)
    end
    sparsevec(cols, vals, length(ds))
end


"""
minimumenergy

Find the minimum energy of the given loc-configuration, iterating over all spin configurations

# Arguments
- ds
- V, J1, J2, J3: parameters
- loc: Locations of the ions.

# Return
- minimum energy
"""
function minimumenergy(ds ::DopedSystem, V::Vector{<:Real}, J::Vector{<:Real}, loc ::Vector{Int})
    mine = Inf
    n_ptl = length(loc)
    fds = freezelocation(ds, loc)
    for i in Iterators.product(Iterators.repeated([-1,1], n_ptl-1)...)
        spins = [1, i...]
        evec = [dot(spins, h * spins) for h in fds.spinhamiltonians]
        energy = dot(J, evec)
        if energy < mine
            mine = energy
        end
    end
    return mine + dot(V, fds.densityenergies)
end


function minimumenergy(ds ::DopedSystem, V ::Vector{<:Real}, J::Vector{<:Real}, occvec ::Vector{Bool})
    loc = [i for (i, y) in enumerate(occvec) if y]
    return minimumenergy(ds, V, J, loc)
end


function minimumenergy(ds ::DopedSystem, V ::Vector{<:Real}, J::Vector{<:Real}, occemp ::OccupiedEmpty)
    mine = Inf
    n_ptl = length(occemp.occupied)
    loc = sort(collect(occemp.occupied))
    fds = freezelocation(ds, loc)

    for i in Iterators.product(Iterators.repeated([-1,1], n_ptl-1)...)
        spins = [1, i...]
        evec = [dot(spins, h * spins) for h in fds.spinhamiltonians]
        energy = dot(J, evec)
        if energy < mine
            mine = energy
        end
    end
    return mine + dot(V, fds.densityenergies)
end


#=
function minimum_locations(ds ::DopedSystem, J1 ::Real, J2 ::Real, J3 ::Real)
    mine = Inf
    mini = 0
    for (iloc, loc) in enumerate(ds.locations)
        energy = minimum_energy(ds, J1, J2, J3, loc)
        if energy < mine
            mine = energy
            mini = iloc
        end
    end
    return (mine, mini)
end
=#

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--out", "-o"
            arg_type = String
            required = true
        "jsoninputfile"
            arg_type = String
            required = true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()

    J1J2J3s = Tuple{Float64, Float64, Float64}[]
    parameters = JSON.parsefile(args["jsoninputfile"])

    ## type can be "local" or "global"
    sweeps = [(s["count"], s["temperature"], s["type"]) ::Tuple{Int, Float64, String} for s in parameters["sweeps"]]

    size = parameters["size"]
    n1, n2 = size[1], size[2]
    nsite = n1 * n2

    nptl = parameters["num_particles"]

    printevery = parameters["print_every"]

    ds = dopedsystem(n1, n2, nptl)

    function localupdate!(x_proposal ::Vector{Bool}, x ::Vector{Bool})
        #copy!(x_proposal, x)
        copyto!(x_proposal, x)
        o = rand([i for (i, y) in enumerate(x) if y])
        e = rand([i for (i, y) in enumerate(x) if !y])
        x_proposal[o] = false
        x_proposal[e] = true
    end

    function globalupdate!(x_proposal ::Vector{Bool}, x ::Vector{Bool})
        #copy!(x_proposal, x)
        copyto!(x_proposal, x)
        sel = randperm(nsite)
        #x_proposal[:] = false
        #x_proposal[sel[1:nptl]] = true
        x_proposal[:] .= false
        x_proposal[sel[1:nptl]] .= true
    end

    output_data = []

    for tp in parameters["tuning_parameters"]
        V ::Vector{Float64} = tp["V"]
        J ::Vector{Float64} = tp["J"]
        println("V: $(V) \tJ: $(J)")

        f(occvec ::Vector{Bool}) = minimumenergy(ds, V, J, occvec)

        current = zeros(Bool, nsite)
        best = zeros(Bool, nsite)
        next = zeros(Bool, nsite)

        Random.seed!(parameters["seed"])
        let
            sel = randperm(nsite)
            current[sel[1:nptl]] .= true
        end

        f_current = f(current)
        f_best = f_current
        #copy!(best, current)
        copyto!(best, current)

        last_update = 0
        sweepiter = 0
        sweeptotal = sum(sweepcount for (sweepcount, _2, _3) in sweeps)

        for (sweepcount, sweeptemperature, sweeptype) in sweeps
            β = 1.0 / sweeptemperature
            update! = if sweeptype == "local"
                localupdate!
            elseif sweeptype == "global"
                globalupdate!
            else
                @assert(false)
            end

            for iter in 1:sweepcount
                sweepiter += 1
                update!(next, current)

                f_next = f(next)
                if f_next < f_current
                    f_current = f_next
                    #copy!(current, next)
                    copyto!(current, next)
                else
                    prob = exp(- β * (f_next - f_current))
                    if rand() < prob
                        f_current = f_next
                        #copy!(current, next)
                        copyto!(current, next)
                    end
                end

                if f_current < f_best
                    last_update = sweepiter
                    f_best = f_current
                    #copy!(best, current)
                    copyto!(best, current)
                end

                if printevery > 0 && sweepiter % printevery == 0
                    print("\rIter: $(sweepiter)/$(sweeptotal) \tCurrent: $(f_current) \tBest: $(f_best) @ $(last_update)")
                end
                #if iter - last_update > 20000
                #    break
                #end
            end
        end
        println("\rIter: $(sweeptotal)/$(sweeptotal) \tCurrent: $(f_current) \tBest: $(f_best) @ $(last_update)")

        bestloc = sort(map(i -> ind2sub((n1, n2), i), i for (i,x) in enumerate(best) if x) )

        let
            p = OrderedDict(tp...)
            p["locations"] = bestloc
            p["best_update"] = last_update
            push!(output_data, p)

            open(args["out"], "w") do file
                JSON.print(file, output_data)
            end
        end
    end
end

main()

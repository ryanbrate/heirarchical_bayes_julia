""" Restrict to a role, Restrict to a tuple count min per v+e, Restict to a specific set of entites
"""

using StatsBase
using LogExpFunctions
using Distributions
using ProgressBars
using SpecialFunctions
using DataStructures
using JLD, HDF5
using JSON
using SparseArrays

const DEBUG = true

mutable struct Hash
    i::Dict{String,Int64}
    next_i::Int64
end
Hash() = Hash(Dict(), 1)

""" Add entry,w,  to Hash object, h, 
"""
function hpush!(h, w)
    if (w in keys(h.i)) == false
        h.i[w] = h.next_i
        h.next_i += 1
    end
end

""" Save ... wrt., a specified role

    Args:
        fps: Vector of absolute filepaths
        doc2viewpoint: Dict("http://resolver..."=>"Catholic_1951")
"""
function sampler(
    fps::Vector{String},
    params::Dict,  # alpha::Float64, gamma::Float64, nu::Float64, P::Int64, Z::Int64
    doc2viewpoint::Dict,
    r::String,  # ROLE OF INTEREST, OTHERS IGNORE
    output_dir::String,
    wanted_entities::Vector{String} = [],
    threshold::Int64 = 50,  # only entites with count(word, role) >= threshold are considered
)
    ## get hyper params
    P::Int64 = params["persona_count"]
    Z::Int64 = params["topic_count"]
    alpha::Float64 = params["alpha"]
    gamma::Float64 = params["gamma"]
    nu::Float64 = params["nu"]

    ## load or build a corpus
    println("\tbuilding an organised corpus")
    (viewpoints, entities, words, roles, v2i, e2i, w2i, r2i) =
        get_corpus(fps, doc2viewpoint, wanted_entities)
    if DEBUG
        report([
            ("\t\tv2i", v2i.i),
            ("\t\te2i", e2i.i),
            ("\t\tw2i", w2i.i),
            ("\t\tr2i", r2i.i),
        ])
        report([
            ("\t\tviewpoints", viewpoints),
            ("\t\tentities", entities),
            ("\t\twords", words),
            ("\t\troles", roles),
        ])
    end

    # get corpus-specific params
    W::Int64 = w2i.next_i - 1
    V::Int64 = v2i.next_i - 1
    E::Int64 = e2i.next_i - 1

    ## init counters
    Czw = spzeros(Int64, Z, W)
    Cpz = spzeros(Int64, P, Z)
    Cvp = spzeros(Int64, V, P)

    ## ve2indices; [vi][ei] = vector of corresponding indices
    println("\tbuilding ve2indices")
    (ve2indices, ve_count, tuples_indices) =
        get_ve2indices(viewpoints, entities, roles, v2i, r2i.i[r], threshold)
    if DEBUG
        report([
            ("\t\tve2indices", ve2indices),
            ("ve_count", ve_count),
            ("len(tuples_indices)", length(tuples_indices)),
        ])
    end

    println("\trandom init. of personas")

    ## randomly init personas to viewpoints, entities
    random_vector = sample(1:P, ve_count)
    personas = spzeros(Int64, V, E)
    counter = 1
    for vi in keys(ve2indices)
        for ei in keys(ve2indices[vi])

            # assign persona
            p = random_vector[counter]
            personas[vi, ei] = p

            # update the counters
            Cvp[vi, p] += 1

            counter += 1
        end
    end

    println("\trandom init. of topics")

    ## randomly init topics for each tuple
    topics::Vector{Int64} = sample(1:Z, length(words))

    ## build Czw and Cpz (restricting ourselves to those indices matching threshold and role)
    for i in tuples_indices
        wi = words[i]
        zi = topics[i]
        ei = entities[i]
        vi = viewpoints[i]
        p = personas[vi, ei]

        Czw[zi, wi] += 1
        Cpz[p, zi] += 1
    end

    println("\tburn-in")

    i2w = Dict([wi => w for (w, wi) in w2i.i])
    i2v = Dict([vi => v for (v, vi) in v2i.i])
    i2e = Dict([ei => e for (e, ei) in e2i.i])

    ## burn-in
    for _ in ProgressBar(1:1000)

        # resample personas for each entity and update Cpz, Cvp and personas
        for vi in keys(ve2indices)
            for ei in keys(ve2indices[vi])
                update_p!(
                    vi,
                    ei,
                    topics[ve2indices[vi][ei]],
                    personas,
                    Cvp,
                    Cpz,
                    alpha,
                    nu,
                    P,
                )
            end
        end

        # resample topics
        for i in tuples_indices
            zi = topics[i]
            wi = words[i]
            vi = viewpoints[i]
            ei = entities[i]
            pi_ = personas[vi, ei]
            update_z!(i, zi, wi, vi, ei, pi_, topics, Czw, Cpz, gamma, nu, Z)
        end
    end

    # 'true' samples
    println("\ttrue samples")

    history_vep = DefaultDict(Vector{Int64})
    history_wz = DefaultDict(Vector{Int64})
    for epoch in ProgressBar(1:1000)

        # resample personas for each entity and update Cpz, Cvp and personas
        for vi in keys(ve2indices)
            for ei in keys(ve2indices[vi])
                update_p!(
                    vi,
                    ei,
                    topics[ve2indices[vi][ei]],
                    personas,
                    Cvp,
                    Cpz,
                    alpha,
                    nu,
                    P,
                )

                if epoch % 100 == 0
                    push!(history_vep[(i2v[vi], i2e[ei])], personas[vi, ei])
                end

            end
        end

        # resample topics
        for i in tuples_indices
            zi = topics[i]
            wi = words[i]
            vi = viewpoints[i]
            ei = entities[i]
            pi_ = personas[vi, ei]
            update_z!(i, zi, wi, vi, ei, pi_, topics, Czw, Cpz, gamma, nu, Z)
            if epoch % 100 == 0
                push!(history_wz[i2w[wi]], zi)
            end
        end
    end

    save_fp = joinpath(output_dir, "history_vep.json")
    mkpath(dirname(save_fp))
    open(save_fp, "w") do f
        JSON.print(f, history_vep, 4)
    end

    save_fp = joinpath(output_dir, "history_wz.json")
    mkpath(dirname(save_fp))
    open(save_fp, "w") do f
        JSON.print(f, history_wz, 4)
    end

end

""" Update Czw, Cpz, topics wrt., new topic sampled for tuple, i"""
function update_z!(
    i::Int64,
    zi::Int64,
    wi::Int64,
    vi::Int64,
    ei::Int64,
    pi_::Int64,
    topics::Vector{Int64},
    Czw::SparseMatrixCSC,
    Cpz::SparseMatrixCSC,
    gamma::Float64,
    nu::Float64,
    Z::Int64,
)

    # remove zi from count matrices
    Czw[zi, wi] -= 1
    Cpz[pi_, zi] -= 1

    # sample new topic
    weights = Weights(conditional_Z(wi, pi_, Czw, Cpz, gamma, nu))
    new_zi = sample(1:Z, weights)

    # update count matrices, an topics
    Czw[new_zi, wi] += 1
    Cpz[pi_, new_zi] += 1
    topics[i] = new_zi

end

""" Update Cpz, Cvp and personas in-place, with newly sampled persona corresponding to entity at vi, ei
"""
function update_p!(
    vi::Int64,
    ei::Int64,
    ve_topics::Vector{Int64},
    personas::SparseMatrixCSC,
    Cvp::SparseMatrixCSC,
    Cpz::SparseMatrixCSC,
    alpha::Float64,
    nu::Float64,
    P::Int64,
)

    c = collect(counter(ve_topics))
    ve_z = first.(c)  # i.e., each unique zi in ve_topics
    ve_z_count = last.(c)  #  i.e., the count for ve_z entries

    # remove the vi,ei instance person from the count matrices ready for sampling
    current_pi = personas[vi, ei]
    Cpz[current_pi, ve_z] -= ve_z_count
    Cvp[vi, current_pi] -= 1

    # sample new pi
    weights = Weights(conditional_P(vi, ve_topics, Cvp, Cpz, alpha, nu))
    new_pi = sample(1:P, weights)

    # update 
    Cpz[new_pi, ve_z] += ve_z_count
    Cvp[vi, new_pi] += 1
    personas[vi, ei] = new_pi

end

"""Return P(Z_vet=x | ...):::Vector{Float64} for all possible x
"""
function conditional_Z(
    wi::Int64,
    pi_::Int64,
    Czw_star::SparseMatrixCSC,
    Cpz_star::SparseMatrixCSC,
    gamma::Float64,
    nu::Float64,
)::Vector{Float64}

    term1n::Vector{Float64} = log.(vec((Czw_star[:, wi] .+ gamma)))  # [Z]
    term1d::Vector{Float64} = log.(vec(sum(Czw_star .+ gamma, dims = 2))) # [Z]

    term2n::Vector{Float64} = log.(vec(Cpz_star[pi_, :] .+ nu))
    # term2d::Vector{Float64} = log.(vec(sum(Cpz_star[pi_,:] .+ nu, dims=2)))

    x = term1n - term1d + term2n

    return exp.(x .- logsumexp(x))
end

"""Return P(P_ve=x | ...)::Vector{Float64} for all possible x
"""
function conditional_P(
    vi::Int64,
    ve_topics::Vector{Int64},
    Cvp_star::SparseMatrixCSC,  # i.e., Cvp removed of contribution p @ ve
    Cpz_star::SparseMatrixCSC,  # i.e., Cpz removed of contribution p @ ve
    alpha::Float64,
    nu::Float64,
)::Vector{Float64}

    term1n = log.(Cvp_star[vi, :] .+ alpha)  # [P]
    term2n = vec(sum(log.(Cpz_star .+ nu)[:, ve_topics], dims = 2))  # [P]
    term2d = log.(vec(sum(Cpz_star .+ nu, dims = 2)))  # [P]

    x = term1n + term2n - term2d

    return exp.(x .- logsumexp(x))
end



""" Return ve2indices::Dict, ve_counter::Int64, tuples_indices
"""
function get_ve2indices(viewpoints, entities, roles, v2i, ri, threshold)

    ve2indices = Dict()
    ve_counter = 0
    tuples_indices = Set()

    # indices corresponding to role of interest, r
    # THIS IS OUR NEW MASTER LIST for building ve2indices
    r_mask = get_mask(roles, ri)
    r_indices = (1:length(roles))[r_mask]  # [1, 2, 56, ...]

    for (v, vi) in v2i.i
        println("\t\t\t$(v):")

        ve2indices[vi] = Dict()

        # indices matching vi and ri
        vr_mask = get_mask(viewpoints[r_indices], vi)  # [1, 0, 0, 1, ...] wrt., r_indices
        vr_indices = r_indices[vr_mask]  # [1, 12, 117, ...]

        # vector of entities matching vi and ri
        v_entities = entities[vr_indices]

        # iterate over the viewpoint's entities
        for ei in ProgressBar(Set(v_entities))

            # indices matching vi, ri, ei
            ver_mask = get_mask(v_entities, ei)
            ver_indices = vr_indices[ver_mask]

            if length(ver_indices) >= threshold
                ve2indices[vi][ei] = ver_indices
                ve_counter += 1
                union!(tuples_indices, ver_indices)
            end

        end
    end

    return (ve2indices, ve_counter, tuples_indices)
end

function get_mask(vector, value)
    return vector .== value
end

# """ Return count::Int64 of all lines in fps.
# """
# function get_line_count(fps)::Int64

#     line_counts = Vector{Int64}(undef, length(fps))
#     Threads.@threads for i in ProgressBar(1:length(fps))
#         lines = readlines(fps[i])
#         line_counts[i] = length(lines)
#     end
#     line_count = sum(line_counts)

# end

""" Return tuples, and supporting hashed for entities in wanted_entities"""
function get_corpus(
    fps::Vector{String},
    doc2viewpoint::Dict,
    wanted_entities::Vector{String} = [],  # if blank, all entities taken
)

    ## build the following ... 
    w2i = Hash()
    r2i = Hash()
    e2i = Hash()
    v2i = Hash()

    # corresponding values from each tuple
    viewpoints = Vector{Int64}()
    entities = Vector{Int64}()
    words = Vector{Int64}()
    roles = Vector{Int64}()


    println("\tbuild vectors")

    viewpoints_set = Set(values(doc2viewpoint))
    doc_labels_set = Set(keys(doc2viewpoint))
    for fp in ProgressBar(fps)
        lines = readlines(fp)
        for i = 1:length(lines)

            # get tuple info
            # line = chomp(readline(f))
            entity, role, word, pattern, doc_label, text = split(lines[i], ", ", limit = 6)

            if (doc_label in doc_labels_set)
                if (length(wanted_entities) == 0) || (entity in wanted_entities)

                    viewpoint = doc2viewpoint[doc_label]

                    # add to hashes
                    hpush!(w2i, word)
                    hpush!(r2i, role)
                    hpush!(e2i, entity)
                    hpush!(v2i, viewpoint)

                    # record in corpus entity counts
                    ri = r2i.i[role]
                    wi = w2i.i[word]
                    ei = e2i.i[entity]
                    vi = v2i.i[viewpoint]

                    # add to vectors
                    push!(viewpoints, vi)
                    push!(entities, ei)
                    push!(words, wi)
                    push!(roles, ri)

                end
            end
        end
    end

    return (viewpoints, entities, words, roles, v2i, e2i, w2i, r2i)

end

## """ Return a Matrix{Int64} of dims D x P, where theta[d,p] = P(p|d)
## """
## function get_theta(Cdp::Matrix{Int64}, alpha::Vector{Float64})

##     # allocated + pseudo counts of personas attributed to each doc
##     dp_counts::Matrix{Float64} = Cdp .+ alpha'  # [D x P]

##     # summation of persona counts attributed to doc
##     d_counts::Matrix{Float64} = sum(dp_counts, dims = 2)  # [D x 1]

##     # theta[d,p] = count(p|d) / count(.|d)
##     return dp_counts ./ d_counts  # [D x P]

## end


## """ Return a Matrix{Int64} of dims K x V, where phi[k,w] = P(w|k)
## """
## function get_phi(Ckw::Matrix{Int64}, gamma::Vector{Float64})

##     # allocated + pseudo counts of words attributed to each topic
##     kv_counts::Matrix{Float64} = Ckw .+ gamma'  # [K x V]

##     # summation of all word counts attributed to each topic
##     k_counts::Matrix{Float64} = sum(kv_counts, dims = 2)  # [K x 1]

##     # phi[k, w] = count(w|k) / count(.|k), including pseudo counts 
##     return kv_counts ./ k_counts  # [K x V]

## end

## """ Return a Vector{Matrix{Int64}} of dims R x P x K, where psi[r][p,k] = P(k|r,p)
## """
## function get_psi(Crpk::Vector{Matrix{Int64}}, nu::Vector{Vector{Float64}})

##     R = length(Crpk)
##     P, K = size(Crpk[1])

##     psi::Vector{Matrix{Float64}} = [zeros(Float64, P, K) for r = 1:R]
##     for r = 1:R

##         # NOTE: julia interprets a Vector as a column in broadcasting

##         # allocated + pseudo topic counts attributed to each r,p combination
##         pk_counts::Matrix{Float64} = Crpk[r] .+ nu[r]'  # [P x K]

##         # summation of topics wrt., each persona
##         p_counts::Matrix{Float64} = sum(pk_counts, dims = 2)  # [P x 1]

##         # psi[r,p,k] = count[k|r,p] / count[.|r,p], including pseudo counts
##         # i.e., p(k|r,p)
##         psi[r] = pk_counts ./ p_counts  # [P x K]

##     end

##     return psi

## end

## """ P(word | theta, phi, psi) 
##         = sum_k,p P(w|k) . P(k | p, r) . P(p)
## """



## # """ Return a float64, a measurement proportional to the model joint probability

## #     i.e., return ...
## #     prod_D prod_P theta_d,p ^ (Cdp + alpha_p - 1) 
## #     x 
## #     prod_K prod_V phi_k,w ^ (Ckw + gamma_w - 1) 
## #     x 
## #     prod_R prod_P prod_K phi_r,p,w ^ (Crpk + nu_r,k - 1) 

## # """
## # function get_log_joint(
## #     Cdp::Matrix{Int64},
## #     Ckw::Matrix{Int64},
## #     Crpk::Vector{Matrix{Int64}},
## #     alpha::Vector{Float64},
## #     gamma::Vector{Float64},
## #     nu::Vector{Vector{Float64}},
## # )

## #     # useful parameters
## #     R = length(nu)

## #     # back calculate multinomials
## #     theta::Matrix{Float64} = get_theta(Cdp, alpha)  # [D x P]
## #     phi::Matrix{Float64} = get_phi(Ckw, gamma)  # [K x V]
## #     psi::Vector{Matrix{Float64}} = get_psi(Crpk, nu)  # [R x P x K]

## #     # calculate metric \prop P(joint)
## #     log_term1::Float64 = beta(alpha) + sum(log.(theta) .* (Cdp .+ alpha' .- 1))
## #     log_term2::Float64 = beta(gamma) + sum(log.(phi) .* (Ckw .+ gamma' .- 1))
## #     log_term3::Float64 = 0
## #     for r = 1:R
## #         log_term3 += beta(nu[r]) + sum(log.(psi[r]) .* (Crpk[r] .+ nu[r]' .- 1))
## #     end

## #     return log_term1 .+ log_term2 .+ log_term3

## # end

## """Return a Float64, the multinomial beta function applied to a dirichlet concentration parameter
## """
## function beta(alpha::Vector{T}) where {T<:Real}
##     exp(sum(log.(alpha)) / gamma(sum(alpha)))
## en6

function report(t, count::Int64 = 200)
    for (name, var) in t
        s = string(var)
        println("$(name) : $(s[1:min(count, length(s))])")
    end
end

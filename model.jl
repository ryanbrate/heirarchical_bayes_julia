using FromFile: @from
using JSON
using DataStructures
using DataFrames
using ProgressBars
using CSV

# load scripts in ./samplers
@from "./samplers/bamman_single.jl" import sampler

function main()

    # load the configs
    config_fp = "model_configs.json"
    configs::Vector = JSON.parsefile(config_fp)

    # load path_syns.json
    path_syns::Dict = JSON.parsefile("path_syns.json")

    # iterate over configs
    for config in configs

        ## get config options

        desc::String = config["desc"]
        println("desc=$(desc)")

        switch::Bool = config["switch"]  # run config or skip?

        # bayesian solver
        # sample = eval(Symbol(config["sampler"]))
        params::Dict = config["params"]

        wanted_entities::Vector{String} = config["wanted_entities"]

        # input locations wrt., extractTuples of interest
        input_dir::String = resolve_fp(config["input"][1], path_syns = path_syns)
        input_pattern::Regex = eval(Meta.parse(config["input"][2]))
        output_dir::String = resolve_fp(config["output_dir"], path_syns = path_syns)

        # metadata information (to build viewpoint2doc)
        viewpoints_criteria::Dict = config["viewpoints"]
        metadata_dir::String = resolve_fp(config["metadata"][1], path_syns = path_syns)
        metadata_pattern::Regex = eval(Meta.parse(config["metadata"][2]))
        label_col = config["label_col"]

        # config to be run?
        if switch == false
            println("\tconfig switched off ... skipping")
        else

            # build doc2viewpoint is doesn't exist
            doc2viewpoint_fp = joinpath(output_dir, "doc2viewpoint.json")
            if isfile(doc2viewpoint_fp) == false

                println("\tbuilding doc2viewpoint dict")
                doc2viewpoint::Dict = get_doc2viewpoint(
                    metadata_dir,
                    metadata_pattern,
                    viewpoints_criteria,
                    label_col,
                )
                # {"http://resolver...": "Catholic", ...}

                ## report number of articles per viewpoint

                # build viewpoint2doc
                println("\tbuilding viewpoint2doc")
                viewpoint2doc = DefaultDict(Vector{String})
                for (doc_label, viewpoint) in doc2viewpoint
                    push!(viewpoint2doc[viewpoint], doc_label)
                end

                # report on number of docs available for each viewpoint
                for (viewpoint, doc_labels) in viewpoint2doc
                    println("\t\t$(viewpoint): $(length(doc_labels)) docs available")
                end

                # save doc2viewpoint
                open(doc2viewpoint_fp, "w") do f
                    JSON.print(f, doc2viewpoint, 4)
                end

            else
                println("loading doc2viewpoint dict")
                doc2viewpoint = JSON.parsefile(doc2viewpoint_fp)
            end

            fps = get_fps(input_dir, pattern = input_pattern)

            # bayesian sampling
            sampler(fps, params, doc2viewpoint, "adj", output_dir, wanted_entities)

        end


        # save a copy of the config
        save_fp = joinpath(output_dir, "model_config.json")
        mkpath(dirname(save_fp))
        open(save_fp, "w") do f
            JSON.print(f, config, 4)
        end
    end
end


""" Return a dict of {doc_label: viewpoint, ...} """
function get_doc2viewpoint(
    metadata_dir::String, # abs path, user-expanded
    metadata_pattern::Regex,
    viewpoints::Dict,
    label_col,
)::Dict

    # build a metadata dataframe
    println("\tload metadata related to model corpus")
    metadata_df = build_metadata(metadata_dir, metadata_pattern)
    println("\t metadata loaded")

    # iterate over viewpoints ...
    doc2viewpoint = Dict()
    for (viewpoint, metadata_requirements) in ProgressBar(viewpoints)

        # for each viewpoint, get a mark wrt., metadata_df rows, giving
        # metadata matches
        masks = []
        for (column, p::String) in metadata_requirements
            pattern::Regex = eval(Meta.parse(p))
            mask = match.(pattern, metadata_df[:, column]) .!== nothing
            push!(masks, mask)
        end

        all_mask = [all(getindex.(masks, i)) for i = 1:length(masks[1])]

        # add to doc2viewpoint
        for doc_label in ProgressBar(metadata_df[all_mask, label_col])
            doc2viewpoint[doc_label] = viewpoint
        end

    end

    return doc2viewpoint
end


""" Return a metadata dataframe.
Combines all dataframes in dir_path , whose name matches pattern & drops duplicate rows
"""
function build_metadata(
    dir_path::String,  # abs path, user-expanded 
    pattern::Regex = r".+",
)::DataFrame

    # fps to relevant csvs containing metadata...
    csvs_fps = get_fps(dir_path, pattern = pattern)

    # # combine into a single dataframe (and drop duplicates)
    df = vcat([DataFrame(CSV.File(f)) for f in csvs_fps]...)

    return df
end



# def get_configs(config_fp_str: str, *, path_syns=None) -> list:
#     """Return the configs to run."""

#     configs_fp = resolve_fp(config_fp_str, path_syns)

#     with open(configs_fp, "rb") as f:
#         configs = orjson.loads(f.read())

#     return configs


""" Return a Vector{String} of absolute filepaths meeting regex requirements.

"""
function get_fps(
    dir_path::String; # assume abspath, user-expanded
    pattern::Regex = r".+",
    ignore_pattern = nothing,
)::Vector{String}

    # get filenames with match against patten
    fns = filter(x -> match(pattern, x) != nothing, readdir(dir_path))

    # igore filtered filenames which match ignore_pattern
    if ignore_pattern !== nothing
        fns = filter(x -> match(ignore_pattern, x) === nothing, fns)
    end
    fps = map(x -> joinpath(dir_path, x), fns)
    return fps

end

"""Return user-expanded path::String, but with path_syns replaced, 
 """
function resolve_fp(
    path::String;  # abspath, even if not user-expanded
    path_syns::Dict = Dict()::String,
)::String

    # resolve path synonyms
    if isempty(path_syns) == false
        for (fake, real) in path_syns
            path = replace(path, fake => real)
        end
    end

    # expand user
    path = expanduser(path)

    return path

end


main()

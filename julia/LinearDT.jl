module LinearDT

export fitness, gen_n_children, gen_child, Node, init_tree, gen_perf, gen_investment_list, gen_investment

using BenchmarkTools
using StatsBase: sample, Weights

# I was to lazy to pass all these parameters arond all the time.
# When they are constants it is more difficult to play around with hyper-parameter
# optimazation, since the whole code has to be rerun,
# so we should probably change this at some point.

# Constants for relevant parameters, constants compile to faster code.
const feature_cost = 0.1
const decision_cost = 0.01
const p_crossover = 0.3
const p_mutate_params = 0.7
const p_mutate_w = 0.5
const p_mutate_threshold = 0.5
const p_mutate_end = 0.3
const p_mutate_subtree = 0.3
const p_opt_decisions = 0.5
const p_opt_tree = 0.5
const p_reduce = 0.5
const p_trim = 0.5
const tourn_size = 3

Investment = Tuple{Vector{Float64}, Vector{Float64}}

"Generates a single pair of possible investments"
function gen_investment(sigmas::Vector{Float64})::Investment
    choice_1 = [randn()*sigma for sigma in sigmas]
    choice_2 = [randn()*sigma for sigma in sigmas]
    return (choice_1, choice_2)
end

"Generates a list of n choices to make"
function gen_investment_list(sigmas::Vector{Float64}, n::Int64)::Vector{Investment}
    [gen_investment(sigmas) for i in 1:n]
end

"Generates a vector of the features of the two choices"
function gen_features(x::Investment)::Vector{Float64}
    vcat(x[1], x[2])
end

"Genreates a random weight"
function rand_w()::Int64
    rand([-1,0,1])
end

"Generates a random threshold"
function rand_threshold()::Float64
    rand()*3 - 1
end

"Generates a random decsion ∈ [1,2]"
function rand_d()::Int64
    rand([1,2])
end

"The custom type for the node object"
mutable struct Node
    w_len::Int64
    sigmas::Vector{Float64}
    w::Array{Int64,}
    threshold::Float64
    left::Union{Int64,Node}
    right::Union{Int64,Node}
    function Node(sigmas::Vector{Float64})
        w_len = length(sigmas)*2
        w = [rand_w() for i in 1:w_len]
        threshold = rand_threshold()
        left = rand_d()
        right = rand_d()
        new(w_len, sigmas, w, threshold, left, right)
    end
end

"Fuction for pretty printing of trees"
function node_string(node::Union{Node,Int64}; tabs::String="")::String
    if isa(node, Node)
        tabs = string(tabs, "\t")
        return string(node.w, "|", round(node.threshold, digits=2), "\n", tabs, "right:", node_string(node.right, tabs=tabs), "\n", tabs, "left:", node_string(node.left, tabs=tabs))
    else
        return string(node)
    end
end
"Sets how tress/nodes are supposed to be printed"
Base.show(io::IO, node::Node) = print(io, string("Root: ", node_string(node)))

"Helper function for the decsion that updates costs"
function update_cost!(cost_vec::Vector{Int64}, n_decisions::Int64, w::Vector{Int64})::Int64
    for i in 1:length(w)
        if w[i] != 0
            cost_vec[i] = 1
            n_decisions += 1
        end
    end
    return n_decisions
end

"Take a decision given features and a tree"
function dt_decide(node::Union{Node,Int64}, features::Vector{Float64}, cost_vec::Vector{Int64}=Int64[], n_decisions::Int64=0)::Tuple{Int64, Vector{Int64}, Int64}
    if length(cost_vec) == 0
        cost_vec = zeros(Int64, length(features))
    end
    if isa(node, Int64)
        return (node, cost_vec, n_decisions)
    else
        n_decisions = update_cost!(cost_vec, n_decisions, node.w)
        if sum(node.w.*features) < node.threshold
            return dt_decide(node.left, features, cost_vec, n_decisions)
        else
            return dt_decide(node.right, features, cost_vec, n_decisions)
        end
    end
end

"Generates a tree of max length max_t, and exptends in each direction with probability p_extend"
function init_tree(max_t::Int64, p_extend::Float64, sigmas::Vector{Float64})::Node
    root = Node(sigmas)
    if (max_t > 1) && (rand() < p_extend)
        root.left = init_tree(max_t - 1, p_extend, sigmas)
    end
    if (max_t > 1) && (rand() < p_extend)
        root.right = init_tree(max_t - 1, p_extend, sigmas)
    end
    return root
end

"Returns a list of all nodes in a subtree"
function gen_node_list(node::Node)::Vector{Node}
    node_list = [node]
    if isa(node.left, Node)
        append!(node_list, gen_node_list(node.left))
    end
    if isa(node.right, Node)
        append!(node_list, gen_node_list(node.right))
    end
    return node_list
end

"Generates a subtree, that can be just a decision. Used by mutate_subtree"
function random_subtree(max_t::Int64, p_extend::Float64, sigmas::Vector{Float64})::Union{Node, Int64}
    if rand() < 1/max_t
        root = rand_d()
    else
        root = init_tree(max_t, p_extend, sigmas)
    end
    return root
end

"Calcualte fitness for a tree from a single decision"
function fitness_single(tree::Node, x::Investment)::Float64
    features = gen_features(x)
    choice, cost_vec, n_decisions = dt_decide(tree, features)
    # if x[1][4] > 0
    #     payoff = sum(x[choice])
    # else
    #     payoff = -sum(x[choice])
    # end
    payoff = sum(x[choice])

    cost = sum(cost_vec)*feature_cost + n_decisions*decision_cost
    return payoff - cost
end

"Calculates the fitness over a vector of decsions for a tree"
function fitness(tree::Node, x_vec::Vector{Investment})::Float64
    fitness = sum(fitness_single(tree,x) for x in x_vec)
end

"Calculates the crossover of two trees"
function cross_over(tree1::Node, tree2::Node)::Node
    new_tree = deepcopy(tree1)
    if rand() < p_crossover
        node = rand(gen_node_list(new_tree))
        new_node = deepcopy(rand(gen_node_list(tree2)))
        if rand() < 0.5
            node.left = new_node
        else
            node.right = new_node
        end
    end
    return new_tree
end

"Mutates the params of a tree"
function mutate_params(tree::Node)
    for node in gen_node_list(tree)
        for i in 1:node.w_len
            if rand() < p_mutate_w
                node.w[i] = rand_w()
            end
        end
        if rand() < p_mutate_threshold
            node.threshold = rand_threshold()
        end

        if isa(node.left, Int64) && (rand() < p_mutate_end)
            node.left = rand_d()
        end
        if isa(node.right, Int64) && (rand() < p_mutate_end)
            node.right = rand_d()
        end
    end
end

"Replaces a node/leaf with a random subtree"
function mutate_subtree(tree::Node, max_t::Int64=2, p_extend::Float64=0.5)
    node = rand(gen_node_list(tree))
    if rand() < 0.5
        node.left = random_subtree(max_t, p_extend, node.sigmas)
    else
        node.right = random_subtree(max_t, p_extend, node.sigmas)
    end
end

"Makes a local optimazation for all decision nodes"
function opt_decisions(tree::Node, x_vec::Vector{Investment})
    for node in gen_node_list(tree)
        for branch in (:left, :right)
            child = getfield(node, branch)
            if isa(child, Int64)
                fit = fitness(tree, x_vec)
                setfield!(node, branch, child == 1 ? 2 : 1)
                if fitness(tree, x_vec) <= fit
                    setfield!(node, branch, child)
                end
            end
        end
    end
end


"Makes a local optimazation for all params, w and threshold, for a node"
function opt_node_params(tree::Node, node::Node, x_vec::Vector{Investment})
    best_fitness = fitness(tree, x_vec)
    for i in 1:node.w_len
        best_val = node.w[i]
        test_vals = [-1,0,1]
        filter!(x -> x != best_val, test_vals)
        for val in test_vals
            node.w[i] = val
            fit = fitness(tree, x_vec)
            if fit > best_fitness
                best_val = val
                best_fitness = fit
            end
        end
        node.w[i] = best_val
    end

    best_threshold = node.threshold
    for a in range(-1.5,stop=1.5, length=19)
        node.threshold = a
        fit = fitness(tree, x_vec)
        if fit > best_fitness
            best_fitness = fit
            best_threshold = a
        end
    end
    node.threshold = best_threshold
end

"Iterates over all decision nodes in a tree and makes a local optimazation of parameters"
function opt_tree_params(tree::Node, x_vec::Vector{Investment})
    for node in gen_node_list(tree)
        opt_node_params(tree, node, x_vec)
    end
end

"Iteratively replaces the nodes with a decision if that improves the fitness"
function reduce_tree(tree::Node, x_vec::Vector{Investment})
    for node in gen_node_list(tree)
        if isa(node.left, Node)
            best_node = node.left
            best_fitness = fitness(tree, x_vec)
            for decision in [1,2]
                node.left = decision
                fit = fitness(tree, x_vec)
                if fit >= best_fitness
                    best_fitness = fit
                    best_node = decision
                end
            end
            node.left = best_node
        end
        if isa(node.right, Node)
            best_node = node.right
            best_fitness = fitness(tree, x_vec)
            for decision in [1,2]
                node.right = decision
                fit = fitness(tree, x_vec)
                if fit >= best_fitness
                    best_fitness = fit
                    best_node = decision
                end
            end
            node.right = best_node
        end
    end
end

"Removes nodes with two identical decisions or if weights are all 0"
function trim_tree(tree::Node)
    for node in gen_node_list(tree)
        if isa(node.left, Node) && all(node.left.w .== 0)
            if node.left.threshold >= 0
                node.left = node.left.left
            else
                node.left = node.left.right
            end
        end
        if isa(node.right, Node) && all(node.right.w .== 0)
            if node.right.threshold >= 0
                node.right = node.right.left
            else
                node.right = node.right.right
            end
        end

        if isa(node.left, Node) && isa(node.left.left, Int64) && isa(node.left.right, Int64) && (node.left.left == node.left.right)
            node.left = node.left.left
        end
        if isa(node.right, Node) && isa(node.right.left, Int64) && isa(node.right.right, Int64) && (node.right.left == node.right.right)
            node.right = node.right.left
        end
    end
end
#
"Generates a named tuple with tree at fitness for all trees in population"
function gen_perf(pop::Vector{Node}, x_vec::Vector{Investment})::Vector{NamedTuple{(:tree, :fit),Tuple{Node, Float64}}}
    [(tree=tree, fit=fitness(tree,x_vec)) for tree in pop]
end

"Returns a single tournament winner"
function sample_tourn_winner(perf::Array{NamedTuple{(:tree, :fit),Tuple{Node,Float64}},1})
    tourn_set = rand(perf, tourn_size)
    sort!(tourn_set, rev=true, by = x -> x.fit)
    return tourn_set[1].tree
end

"Selects two parents by tournament selection"
function tournament_selection(perf::Array{NamedTuple{(:tree, :fit),Tuple{Node,Float64}},1})
    tree1 = sample_tourn_winner(perf)
    tree2 = sample_tourn_winner(perf)
    return (tree1, tree2)
end

"Returns two parents by weigthed selection, used by roulette and rank selection"
function weighted_selection(perf::Array{NamedTuple{(:tree, :fit),Tuple{Node,Float64}},1}, weights::Vector{Float64})
    tree1 = sample(perf, Weights(weights)).tree
    tree2 = sample(perf, Weights(weights)).tree
    return (tree1, tree2)
end

"Selects two parents, creates a child with crossower, mutates, optimizes and trims"
function gen_child(selection_mechanism::String, perf::Array{NamedTuple{(:tree, :fit),Tuple{Node,Float64}},1}, x_vec::Vector{Investment}, weights::Vector{Float64})::Node
    if selection_mechanism == "tournament"
        tree1, tree2 = tournament_selection(perf)
    else
        tree1, tree2 = weighted_selection(perf, weights)
    end

    new_tree = cross_over(tree1, tree2)

    if rand() < p_mutate_params
        mutate_params(new_tree)
    end
    if rand() < p_mutate_subtree
        mutate_subtree(new_tree)
    end
    if rand() < p_opt_decisions
        opt_decisions(new_tree, x_vec)
    end
    if rand() < p_opt_tree
        opt_tree_params(new_tree, x_vec)
    end
    if rand() < p_reduce
        reduce_tree(new_tree, x_vec)
    end
    if rand() < p_trim
        trim_tree(new_tree)
    end

    return new_tree
end

"Iteratively generates children unitl n are generated. Not currently in use with par"
function gen_n_children(n::Int64, selection_mechanism::String, perf::Array{NamedTuple{(:tree, :fit),Tuple{Node,Float64}},1}, x_vec::Vector{Investment}, weights::Vector{Float64})::Vector{Node}
    children = [gen_child(selection_mechanism, perf, x_vec, weights) for i in 1:n]
    return children
end

end

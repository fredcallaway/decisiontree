module MemoryDT
using Logging
using Parameters
using Memoize

export Params, Problem, Node, fitness, evolve!
using StatsBase: sample, Weights

"Global parameters for problems and trees."
@with_kw struct Params
    n_mem::Int = 2
    n_feature::Int = 3
    n_choice::Int = 2
    p_branch::Float64 = 0.5
    node_cost = 1.0
    write_cost = 0.1
    sigmas = [1., 2., 4.]  # TODO enforce consistency with n_feature
    literal_ratio = 0.2
    n_problem = 1000
end

# ---------- Problem ---------- #
"An investment problem defined by a payoff matrix X."
struct Problem
    X::Matrix{Float64}
end

Problem(sigmas::Vector{Float64}, n_items::Int64) = begin
    return Problem(randn(length(sigmas), n_items) .* sigmas)
end
Problem(t::Params) = Problem(t.sigmas, t.n_choice)

function payoff(problem::Problem, choice::Int)::Float64
    sum(problem.X[:, choice])
end

function features(problem::Problem)::Vector{Float64}
    vcat(problem.X...)
end

# ---------- Decision Tree ---------- #

abstract type Node end

"""
    A Branch node updates memory and passes control to a child node.

Memory is updated with the equation ``m' = m + X φ``
The child node is determined made by the inequality ``w ⋅ m > threshold``
"""
struct Branch <: Node
    X::Matrix{Int}
    w::Vector{Int}
    threshold::Float64
    children::Tuple{Node, Node}
end

"""
    A Decision node makes a choice based on memory.

Currently, this choice is hardcoded: select the item that has
the highest accumulator value (this assumes that each slot in
memory corseponds to one item).
"""
struct Decision <: Node

end

# A mask for Branch.X matrices, see randX docstring.
@memoize function feature_mask(t::Params)
    tmp = map(1:t.n_choice) do i
        B = zeros(t.n_feature, t.n_choice)
        B[:, i] .= 1
        hcat(B...)
    end
    vcat(tmp...)
end

"""
    Generates a random Branch.X matrix.

Currently, we impose a strong prior based on the assumption
that the memory encodes value accumulators for each item.
"""
randX(t::Params) = begin
    # TODO this is probably inefficient
    X = rand(-1:1, (t.n_mem, t.n_choice * t.n_feature))
    X .*= feature_mask(t)
    X = max.(0, X)
    return X
end

randw(t::Params) = rand(-1:1, t.n_mem)
rand_threshold(t::Params) = randn() * √ sum(t.sigmas .^ 2)

Node(t::Params) = rand() < t.p_branch ?  Branch(t) : Decision(t)
Decision(t::Params) = Decision()
Branch(t::Params) = Branch(
    randX(t),
    randw(t),
    rand_threshold(t),
    (Node(t), Node(t))
)

"Fuction for pretty printing of trees"
function node_string(node::Node, tabs::String="")::String
    if node isa Decision
        return tabs * "Decision"
    end
    parent = string(tabs, node.X, "\n", tabs, node.w, " > ", round(node.threshold, digits=2))
    children = [node_string(n, tabs * '\t') for n in node.children]
    join([parent; children], "\n")
end
Base.print(node::Node) = print(node_string(node))

"The cost of executing a node."
node_cost(t::Params, node::Node) = begin
    t.write_cost * sum(abs.(node.X)) + t.node_cost
end

"Runs the decision tree algorithm starting at `node`. Returns decision and cost."
function decide(t::Params, node::Node, problem::Problem)
    @debug "deciding..."
    memory = zeros(t.n_mem)
    cost = 0.0
    phi = features(problem)
    while node isa Branch
        cost += node_cost(t, node)
        memory .+= node.X * phi
        @debug memory cost
        child = node.w'memory > node.threshold ? 1 : 2
        node = node.children[child]
    end
    decision = argmax(memory)  # NOTE hardcoded decision rule!
    @debug "decision:" decision
    (decision=decision, cost=cost)
end


# ----------- Evolution ---------- #

function fitness(t::Params, node::Node, problem::Problem)
    decision, cost = decide(t, node, problem)
    return payoff(problem, decision) - cost
end
fitness(t::Params, node::Node, problems::Vector{Problem}) = begin
    sum(fitness(t, node, p) for p in problems)
end

function mutate!(t::Params, node::Branch)
    change = false
    for i in 1:length(node.X)
        if rand() < t.p_mutate_X
            change = true
            node.X[i] = rand(-1:1)
        end
    end
    for i in 1:length(node.w)
        if rand() < t.p_mutate_w
            change = true
            node.w[i] = rand(-1:1)
        end
    end
    if rand() < t.p_mutate_threshold
        change = true
        node.threshold = rand_threshold()
end
mutate!(t::Params, node::Decision) = false

function optimize!(t::Params, node::Branch)
    for child in node.children
        optimize!(t, child)
    end
    # TODO or not TODO, that is the question
end
optimize(t::Params, node::Decision) = nothing

function evolve!(t, pop)
    problems = [Problem(t) for i in 1:t.n_problem]
    fits = [fitness(t, node, problems) for node in pop]
    pop[:] = sample(pop, Weights(fits), length(pop))
    for node in pop
        change = mutate!(t, node)
        if change
            optimize!(t, node)
        end
    end



    # ranks = sortperm(fits, rev=true)
    # pop[:] = pop[ranks]  # pop is reverse sorted
    # fits[:] = fits[ranks]
    n_keep = Int(ceil(length(pop) * t.literal_ratio))
    n_new = length(pop) - n_keep
    for i in n_keep+1:length(pop)
        pop[i] = Node(t)
    end
    return fits[ranks][1]
end

t = Params()
feature_mask(t)

# plot(1:3)
#
# t = Params(write_cost=0, n_problem=1000)
# pop = [Node(t) for i in 1:100]
# score = [evolve!(t, pop) for i in 1:100]
# print(pop[1])

end # module

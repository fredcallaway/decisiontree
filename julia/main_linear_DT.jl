using BenchmarkTools
push!(LOAD_PATH, pwd())

using LinearDT
using Distributed
using Statistics: mean
using Dates
## Setup for parallisation, the @everywhere keyword is used to maked vairables, functions etc. available for all workers
addprocs(Sys.CPU_THREADS)
# addprocs(4)
@everywhere push!(LOAD_PATH, pwd())
@everywhere using LinearDT


# %%
# All theses values have to be shared among the workers
@everywhere begin
    pop_size = 200
    n_elit = 3
    periods = 10
    n_problems = 10000
    sigmas = [1., 0.7, 0.5, 0.2]
    selection_mechanism = "tournament"
    params = (pop_size=200, n_items=3, n_attr=length(sigmas), max_t=5, p_extend=0.7, sigmas=sigmas, selection_mechanism=selection_mechanism, feature_cost=0.05, decision_cost=0.01, p_crossover=0.3, p_mutate_params=0.7, p_mutate_w=0.5, p_mutate_threshold=0.5, p_mutate_end=0.3, p_mutate_subtree=0.3, p_opt_decisions=0.5, p_opt_tree=0.5, p_reduce=0.5, p_trim=0.5, tourn_size=3)
end

# The globals are needed to work from terminal, bugish behavior
global pop = [init_tree(params.max_t, params) for i in 1:pop_size]
global prev_best = pop[1]



for i in 1:periods
    println(Dates.format(now(), "HH:MM:SS"))
    global pop = pop
    global prev_best = prev_best
    x_vec = gen_investment_list(n_problems, sigmas, params)
    prev_best_fit = fitness(prev_best, x_vec, params.feature_cost, params.decision_cost)
    @everywhere begin
        x_vec = $x_vec
        single_perf = tree -> (tree=tree, fit=fitness(tree, x_vec, params.feature_cost, params.decision_cost))
    end
    perf = pmap(single_perf, pop)
    # perf = gen_perf(pop, x_vec)
    sort!(perf, rev=true, by = x -> x.fit)

    println(string("----- Best individual in iteration: ", i, " -------"))
    println(string("Avg perf:", mean([x.fit for x in perf])))
    println(string("Prevbest:", prev_best_fit))
    println(string("Perf: ", perf[1].fit, "\n", perf[1].tree))
    if selection_mechanism == "rank"
        weights = [1/i for i in 1:pop_size]
    elseif selection_mechanism == "roulette"
        weights = [x.fit for x in perf]
        weights = weights./sum(weights)
    else
        weights = Float64[]
    end
    elit_pop = [x.tree for x in perf[1:n_elit]]
    prev_best = elit_pop[1]
    @everywhere begin
        weights = $weights
        perf = $perf
        gen_child_parallel = x-> gen_child(selection_mechanism, perf, x_vec, weights, params)
    end
    new_pop = pmap(gen_child_parallel, 1:(pop_size - n_elit))
    # new_pop = gen_n_children(pop_size - n_elit, selection_mechanism, perf, x_vec, weights)
    append!(new_pop, elit_pop)
    # global pop = new_pop
    pop = new_pop
end

# %%

# After the algorithm is done, compare all remaining trees on
# a larger set of problems to find best.
n_problems = 100000
x_vec = gen_investment_list(n_problems, sigmas, params)
@everywhere begin
    x_vec = $x_vec
    pop_fun = x -> init_tree(params.max_t, params)
    single_perf = tree -> (tree=tree, fit=fitness(tree, x_vec, params.feature_cost, params.decision_cost))
end


@everywhere pop = $pop
@time perf = pmap(single_perf, pop)
sort!(perf, rev=true, by = x -> x.fit)
best_tree = perf[1].tree;


# %%


get_decision_payoff_tuple = x -> begin
    decision = dt_decide(best_tree, gen_features(x))[1]
    (decision, decision_payoff(x, decision))
end
tree_decisions = [get_decision_payoff_tuple(x) for x in x_vec]
opt_decisions = [actual_best_decision(x, params) for x in x_vec];

avg_tree_profit = mean([x[2] for x in tree_decisions])
avg_opt_profit = mean([x[2] for x in opt_decisions])
println(string("Avg payoff from tree decisions:", avg_tree_profit))
println(string("Avg payoff from tree decisions:", avg_opt_profit))
println(string("Share of profit: ", avg_tree_profit/avg_opt_profit))
println(string("Share same decision:", sum([x[1] for x in opt_decisions] .== [x[1] for x in tree_decisions])/length(tree_decisions)))



### Old non-parallised code
# pop_size = 200
# max_t = 3
# p_extend = 0.7
# elit = 2
# periods = 20
# n_problems = 200
# sigmas = [1., 0.7, 0.2, 0.1]
#
#
#
#
# selection_mechanism = "tournament" # One of "rank" | "tournament" "roulette"
# pop = [init_tree(max_t, p_extend, sigmas) for i in 1:pop_size]
# prev_best = pop[1]
# for i in 1:periods
#     x_vec = gen_investment_list(n_problems, sigmas)
#     perf = gen_perf(pop, x_vec)
#     sort!(perf, rev=true, by = x -> x[2])
#     println(string("----- Best individual in iteration: ", i, " -------"))
#     println(string("Avg perf:", mean([x ->x.fit for x in perf])))
#     println(string("Prevbest:", fitness(prev_best, x_vec)))
#     println(string("Perf: ", perf[1].fit, "\n", perf[1].tree))
#     if selection_mechanism == "rank"
#         weights = [1/i for i in 1:pop_size]
#     elseif selection_mechanism == "roulette"
#         weights = [x.fit for x in perf]
#         weights = weights./sum(weights)
#     else
#         weights = Float64[]
#     end
#     new_pop = [x.tree for x in perf[1:elit]]
#     append!(new_pop, gen_n_children(pop_size,selection_mechanism, perf, x_vec, weights))
#     prev_best = deepcopy(perf[1].tree)
#     pop = new_pop
# end

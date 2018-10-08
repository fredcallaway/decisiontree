push!(LOAD_PATH, pwd())
using LinearDT

using BenchmarkTools
using Statistics: mean

pop_size = 3
max_t = 3
p_extend = 0.7
n_elit = 3
periods = 20
n_problems = 10
sigmas = [1., 0.7, 0.5, 0.2]
selection_mechanism = "tournament"
params = (pop_size=200, n_items=3, n_attr=length(sigmas), max_t=5, p_extend=0.7, sigmas=sigmas, selection_mechanism=selection_mechanism, feature_cost=0.05, decision_cost=0.01, p_crossover=0.3, p_mutate_params=0.7, p_mutate_w=0.5, p_mutate_threshold=0.5, p_mutate_end=0.3, p_mutate_subtree=0.3, p_opt_decisions=0.5, p_opt_tree=0.5, p_reduce=0.5, p_trim=0.5, tourn_size=3)

problems = gen_investment_list(n_problems, sigmas, params)
pop = [init_tree(max_t, params) for i in 1:pop_size]

performance(tree) = (tree=tree, fit=fitness(tree, problems, params.feature_cost, params.decision_cost))
perf = map(performance, pop)
weights = Float64[]
gen_child(selection_mechanism, perf, problems, weights, params)


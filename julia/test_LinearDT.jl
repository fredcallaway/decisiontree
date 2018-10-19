push!(LOAD_PATH, pwd())
using LinearDT

using BenchmarkTools
using Statistics: mean
using Profile

pop_size = 50
max_t = 5
p_extend = 0.7
n_elit = 3
periods = 20
n_problems = 100
sigmas = [1., 0.7, 0.5, 0.2]
n_items = 3
selection_mechanism = "tournament"
params = (pop_size=pop_size, n_items=n_items, n_attr=length(sigmas), max_t=5, p_extend=0.7, sigmas=sigmas, selection_mechanism=selection_mechanism, feature_cost=0.05, decision_cost=0.01, p_crossover=0.3, p_mutate_params=0.7, p_mutate_w=0.5, p_mutate_threshold=0.5, p_mutate_end=0.3, p_mutate_subtree=0.3, p_opt_decisions=0.5, p_opt_tree=0.5, p_reduce=0.5, p_trim=0.5, tourn_size=3)


function evolve()
    problems = gen_investment_list(n_problems, sigmas, params)
    pop = [init_tree(max_t, params) for i in 1:pop_size]
    performance(tree) = (tree=tree, fit=fitness(tree, problems, params.feature_cost, params.decision_cost))
    perf = map(performance, pop)

    weights = Float64[]
    [gen_child(selection_mechanism, perf, problems, weights, params)
           for i in 1:pop_size]
end


@profile evolve()
open("profile.txt", "w") do s
    context = IOContext(s, :displaysize => (24, 500))
    Profile.print(context, noisefloor=2, mincount=10)
end

function bench_fitness(n_problems=1000, pop_size=20)
    problems = gen_investment_list(n_problems, sigmas, params)
    pop = [init_tree(max_t, params) for i in 1:pop_size]
    # performance(tree) = (tree=tree, fit=fitness(tree, problems, params.feature_cost, params.decision_cost))
    performance(tree) = fitness(tree, problems, params.feature_cost, params.decision_cost)
    performance(pop[1])
    @time map(performance, pop)
end
# bench_fitness(1000, 100)
# tree = init_tree(max_t, params)
# problem = gen_investment(sigmas, n_items)
# features = gen_features(problem)
# dt_decide(tree, features)
# @time dt_decide(tree, features)

# evolve()
# @profile evolve()
# Profile.print(mincount=50)
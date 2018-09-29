import numpy as np
from functools import partial
import random 
import copy
import multiprocessing as mp


# Two functions for generating the datasets from a list of variances
def gen_investment(sigmas, n_item=2):
    X = np.random.randn(n_item, len(sigmas))
    return X * sigmas

def gen_investment_list(sigmas, n):
    return [gen_investment(sigmas) for i in range(n)]


# The decision trees are built recursively from the node objects. The nodes
# are either of comparision or end types. The end nodes simply have the
# decision related to it. The comparison nodes have feature weights anda
# threshold and point to two child nodes, left and right. If weights\*x <
# threshold the left child node is selected, otherwise the right child node.

# Functions to sample the parameters of the decision nodes. Used for both generation and mutation of DTs
def rand_weights():
    return np.random.randint(-1, high=2)

def rand_threshold():
    return np.random.rand()*2 -1
    # return 0

def rand_decision():
    return np.random.randint(0,2) 

# Nodes that are used for recursive construction of the DTs. Can be of either compare or end type. 
class Node:
    def __init__(self, node_type="compare", weights_len=0):
        if node_type == "compare":
            self.weights_len = weights_len
            self.type = node_type
            self.weights = np.array([rand_weights() for i in range(weights_len)])
            self.threshold = rand_threshold()
            self.left = None
            self.right = None
        elif node_type == "end":
            self.type = node_type
            self.decision = rand_decision()  
        self.depth_cost = 1

    def add_left_compare(self):
        self.left = Node(weights_len=self.weights_len)
    
    def add_right_compare(self):
        self.right = Node(weights_len=self.weights_len)
    
    def add_left_end(self):
        self.left = Node(node_type="end")
    
    def add_right_end(self):
        self.right = Node(node_type="end")
    
    # Simple function for printing the decision trees for observability
    def _str(self, tabs=""):
        if self.type == "compare":
            tabs = tabs + "\t"
            return (str(self.weights) + "|" + str(round(self.threshold,2)) + "\n" + tabs + 
                    "left:" + self.left._str(tabs) + "\n" + tabs +
                    "right:" + self.right._str(tabs))
        elif self.type=="end": 
            return str(self.decision)
        else:
            return ""

    def __str__(self):
        return "Root:" + self._str()
    
    def copy(self):
        return copy.deepcopy(self)
    
    # This is the function that performs the decision, again a recursive defintion
    def decide(self, x, cost=0):
        if self.type == "end":
            return (self.decision, cost)
        else:
            new_cost = (self.weights != 0).sum() + self.depth_cost
            if  np.dot(self.weights, x) < self.threshold:
                return self.left.decide(x, cost + new_cost)
            else:
                return self.right.decide(x, cost + new_cost)
    
    # Helper function to get a list of all non-termianl nodes. Used for mutations and crossover
    def node_list(self):
        if self.type == "end":
            return np.array([])
        else:
            return np.concatenate((np.array([self]), self.left.node_list(), self.right.node_list()))


    
    # Helper function to get a list of all end-nodes, used for mutation. 
    def end_list(self):
        if self.type == "end":
            return np.array([self])
        else:
            return np.concatenate((self.left.end_list(), self.right.end_list()))


# Generates a tree of max-depth max_t, where the probability of extending a 
# node in each direction is given by p_extend
def init_tree(max_t, p_extend, weights_len=10):
    root = Node(weights_len=weights_len)
    if (max_t > 1) and (np.random.rand() < p_extend):
        root.left = init_tree(max_t - 1, p_extend, weights_len=weights_len)
    else:
        root.add_left_end()
    if (max_t > 1) and (np.random.rand() < p_extend):
        root.right = init_tree(max_t - 1, p_extend, weights_len=weights_len)
    else:
        root.add_right_end()
    return root

# The fitness calculation
def fitness_single(tree, x, c=1):
    choice, cost = tree.decide(x.flat)
    payoff = x[choice].sum()
    cost = cost*c
    return payoff - cost

def fitness(tree, x_vec, c=1):
    return sum(fitness_single(tree, x, c) for x in x_vec)


# Crossover, with probability p one node including subtree, is replace in
# tree1 from tree2. Tree1 is then the child.
def cross_over(tree1, tree2, p):
    if  np.random.random() < p:
        new_tree = tree1.copy()
        node = np.random.choice(new_tree.node_list())
        replace_node = np.random.choice(tree2.node_list()).copy()
        if 0.5 < np.random.rand():
            node.left = replace_node
        else:
            node.right = replace_node
        return new_tree
    else:
        return tree1.copy()

# Mutates the params at the nodes
def param_mutate(tree, p_threshold, p_end):
    for node in tree.node_list():
        for i in range(len(node.weights)):
            if np.random.rand() < p_weights:
                node.weights[i] = rand_weights()
        if np.random.rand() < p_threshold:
            node.threshold = rand_threshold()
    
    for end in tree.end_list():
        if np.random.rand() < p_end:
            end.decision = rand_decision()

# Generates a new random subtree from some node
def subtree_mutate(tree, max_t=2, p_extend=0.5):
    node = np.random.choice(tree.node_list())
    if np.random.rand() < 0.5:
        node.left = init_tree(max_t, p_extend)
    else:
        node.right = init_tree(max_t, p_extend)

# Calculates the tournament winner. Used for tournament selection
def sample_turn_winner(perf, tourn):
    turn_set = [random.choice(perf) for i in range(tourn)]
    turn_set.sort(reverse=True,key= lambda x: x["fit"])
    return turn_set[0]["tree"]

def tournament_selection(perf, tourn_size):
    tree1 = sample_turn_winner(perf, tourn)
    tree2 = sample_turn_winner(perf, tourn)
    return (tree1, tree2)

def weighted_selection(perf, weigths):
    tree1 = np.random.choice(perf, p=weights)["tree"]
    tree2 = np.random.choice(perf, p=weights)["tree"]
    
def gen_child(selection_mechanism, perf, weights):
    if selection_mechanism == "tournament":
        tree1,tree2 = tournament_selection(perf, tourn_size)
    else:
        tree1 = np.random.choice(perf, p=weights)["tree"]
        tree2 = np.random.choice(perf, p=weights)["tree"]
    
    new_tree = cross_over(tree1, tree2, p_crossover)

    if np.random.rand() < param_mut_p:
        param_mutate(new_tree, p_threshold, p_end)
    if np.random.rand() < subtree_mut_p:
        subtree_mutate(new_tree, max_t=2, p_extend=p_extend)
    
    return new_tree

def gen_n_children(n, selection_mechanism, perf, weights):
    children = []
    for i in range(n):
        children.append(gen_child(selection_mechanism, perf, weights))
    return children

def gen_perf(pop, x_vec, c):
    perf = [{"tree":tree, "fit":fitness(tree, x_vec, c=c)} for tree in pop]
    return perf


# ### Running the evolutionary algorithm 
# 
# #### Set parameters

# In[35]:


param_mut_p = 1.    # Probability of mutation for children
p_weights = 0.6           # Probability of each element of weight vector to be mutated, conditional on mutation
p_threshold = 0.6           # Probability of threshold to be mutated, conditional on mutation 
p_end = 0.2         # Probability of the decision at end-nodes to be mutated, conditional on mutation
subtree_mut_p = 0.0 # Probability of subtree-mutation
p_crossover = 0.    # Probability of cross-over
pop_size = 100     # Size of population
inves_len = 500  # How many decision problems are used when calculating fitness 
elit = 5            # The number of best individuals kept in each iteration without any modification
tourn = 5       # Number of individuals in each tournament in case of tournament selection
c = 0.015           # Cognitive cost per non-zero weight and comparison
max_t = 2           # Max-depth of trees
p_extend = 0.3      # Probability of extending along each node when generating
periods = 50        # Number of periods the evolutionary algorithm runs

# params = {"param_mut_p":param_mut_p, "p_weights":p_weights, "p_threshold":p_threshold, "p_end":p_end}


selection_mechanism = "rank" # The used selection rule, one of  "rank" | "tournament" | "roulette" 


## The vector of standard deviations for the investment problem 
# sigmas = [1., 0.01, 0.01, 0.01] 
sigmas = [1., 0.5, 0.25, 0.125, 0.01] 


# Set up for parallellization
cores = mp.cpu_count() # If cores is 1, no parallelization is performed
# cores = 1
n_per_core = int((pop_size -elit)/cores)
n_extra = pop_size - n_per_core*cores - elit

# In[ ]:



def evolve(periods=50, verbose=False):
    pop = [init_tree(max_t, p_extend, weights_len=2*len(sigmas)) for i in range(pop_size)]
    for i in range(periods):
        # Generate a new set of investment problems, to avoid overfitting
        x_vec = gen_investment_list(sigmas, inves_len)
        
        # Gives a list of one dict for each tree with the tree and its performance
        if cores > 1:
            perf = []
            pool = mp.Pool(processes=cores)
            res_mp = [pool.apply_async(gen_perf, args=(pop[n*n_per_core:(n+1)*n_per_core], x_vec, c)) for n in range(cores)]
            for p in res_mp:
                perf.extend(p.get())
            pool.close()
            pool.join()
            perf.extend(gen_perf(pop[(cores+1)*n_per_core:], x_vec, c))
        else:
            perf = [{"tree":tree, "fit":fitness(tree, x_vec, c=c)} for tree in pop]
            
        perf.sort(reverse=True,key= lambda x: x["fit"])
        
        if selection_mechanism == "roulette":
            weights = np.array([x["fit"] for x in perf])
            weights = weights + perf[-1]["fit"]
            weights = weights/sum(weights)
        elif selection_mechanism == "rank":
            weights = np.array([1/(i+1) for i in range(len(perf))])
            weights = weights/weights.sum()
        else:
            weights = np.array([])
        if verbose:
            print("---- Best individual in iteration: " + str(i) + "-----")
            print("Perf: " + str(perf[0]["fit"]) + "\n" + str(perf[0]["tree"]))
        
        # Save the five best trees
        new_pop = [x["tree"] for x in perf[:elit]]
        
        if cores > 1:
            pool = mp.Pool(processes=cores)
            res_mp = [pool.apply_async(gen_n_children, args=(n_per_core, selection_mechanism, perf, weights)) 
                      for x in range(cores)]
            for p in res_mp:
                new_pop.extend(p.get())
            pool.close()
            pool.join()
            new_pop.extend(gen_n_children(n_extra, selection_mechanism, perf, weights))
        else:
            new_pop.extend(gen_n_children(pop_size - elit, selection_mechanism, perf, weights))
        
        pop = new_pop

if __name__ == '__main__':
    evolve(verbose=True)

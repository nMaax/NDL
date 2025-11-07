# %% [markdown]
# # Homework 1

# %%
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt

# %% [markdown]
# ## Exercise 1

# %% [markdown]
# Consider the network in Figure with link capacities provided over the links

# %%
G = nx.DiGraph()

nodes = ['o', 'a', 'b', 'c', 'd']
G.add_nodes_from(nodes)

edges = [
    ('o', 'a', {'capacity': 3}),
    ('a', 'd', {'capacity': 3}),
    ('o', 'b', {'capacity': 3}),
    ('b', 'c', {'capacity': 3}),
    ('c', 'd', {'capacity': 2}),
    ('a', 'b', {'capacity': 1}),
]
G.add_edges_from(edges)

pos = {
    'o': (0, 0),    # Left center
    'a': (1, 1),    # Top middle-left
    'b': (1, 0),    # Center
    'c': (1, -1),   # Bottom middle-right
    'd': (2, 0)     # Top right
}

def draw_capacited_graph(G, pos=pos):
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=500, arrowsize=20)

    edge_labels = nx.get_edge_attributes(G, 'capacity')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

draw_capacited_graph(G)

# %% [markdown]
# Compute the capacity of all the cuts and find the minimum capacity to be removed for no
# feasible flow from $o$ to $d$ to exist

# %%
nx.flow.minimum_cut(G, 'o', 'd') # Also {o, b, c} vs. {a, d}

# %% [markdown]
# You are given $x > 0$ extra units of capacity $(x ∈ Z)$. How should you distribute them in order
# to maximize the throughput that can be sent from $o$ to $d$? Plot the maximum throughput
# from o to d as a function of $x ≥ 0$

# %%
import collections

class Graph:
    """
    This class represents a directed graph using
    adjacency matrix representation.
    """

    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.row = len(graph)

    def bfs(self, s, t, parent):
        """
        Returns true if there is a path from
        source 's' to sink 't' in residual graph.
        Also fills parent[] to store the path.
        """

        # Mark all the vertices as not visited
        visited = [False] * self.row

        # Create a queue for BFS
        queue = collections.deque()

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS loop
        while queue:
            u = queue.popleft()

            # Get all adjacent vertices of the dequeued vertex u
            # If an adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if (visited[ind] == False) and (val > 0):
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        # If we reached sink in BFS starting from source, then return
        # true, else false
        return visited[t]

    # Returns the maximum flow from s to t in the given graph
    def edmonds_karp(self, source, sink):
        # This array is filled by BFS and to store path
        parent = [-1] * self.row

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.bfs(source, sink, parent):
            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow

# %%
mfGraph = Graph(np.array(
        [[0, 3, 3, 0, 0],
        [0, 0, 1, 0, 3],
        [0, 0, 0, 3, 0],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0]]
    ))
print(mfGraph.edmonds_karp(0, 4))

# %%
def intervetion(G, x=1):
    for _ in range(x):

        #! This can be optimized

        # Get current bottleneck
        mincut, bottleneck = nx.flow.minimum_cut(G, 'o', 'd')
        
        # Pick the lowest capacity outgoing edge of the bottleneck
        boundary = nx.edge_boundary(G, *bottleneck)
        min_cap = float('inf')
        for edge in boundary:
            capacity = G[edge[0]][edge[1]]['capacity']
            if capacity == min_cap:
                edge_mincap = edge_mincap.append(edge)
            if capacity < min_cap or not edge_mincap:
                min_cap = capacity
                edge_mincap = [edge]
        
        # Logging abd increase it by 1
        print(f"Increasing edges {edge_mincap} with capacity: {min_cap} by 1...")
        for edge in edge_mincap:
            G[edge[0]][edge[1]]['capacity'] += 1

    # Return the maximum flow value of the new graph
    return nx.flow.maximum_flow_value(G, 'o', 'd')

# Apply the intervention and show the result
G1 = nx.DiGraph(G)
intervetion(G1, 30)
draw_capacited_graph(G1)

# %%
G2 = nx.DiGraph(G)
num_iterations = 30
maxflow = [nx.flow.maximum_flow_value(G, 'o', 'd')] + [intervetion(G2) for i in range(1, num_iterations+1)]

plt.plot(np.arange(0, num_iterations+1), maxflow, drawstyle='steps-post')
plt.show()

# %% [markdown]
# You are given the possibility of adding to the network a directed link $e_8$ with capacity $c_8 = 1$
# and $x > 0$ extra units of capacity $(x ∈ Z)$. Where should you add the link and how should
# you distribute the additional capacity in order to maximize the throughput that can be sent
# from o to d? Plot the maximum throughput from $o$ to $d$ as a function of $x ≥ 0$.

# %% [markdown]
# > **Answer**: The most straightforward solution is to introduce a direct link between $o$ and $d$, introducing the whole capacity in it. It would intuitively act as an "highway" of flow.
# >
# > This comes directly from mincut-maxflow theorem, as by definition of "cut" we have that $\mathbb{U}$ must always contain the node o, and conversley $\mathbb{U}^{c}$ must always contain node d. As such, introducing a link between these two will always result in an out-boundary edge, being relevant for *any* cut that can be found in the network; it then act as "universal" capacity increaser, as we can simply add all capacity there and virtually increase the capacity of *all* cuts simoultaneously, including the bottleneck.
# >
# > If instead we would have proceeded by introducing the link between some other nodes, we would have, in the best cases, resolved the local cut to not be a bottleneck anymore, but then another would have appeared around the network. With the solution we propose this would have not virtually happened as such link would have been shared by all cuts, tho without effectively modifying the "ranking" of best/worse cuts.

# %% [markdown]
# Alterantively, once can still distribute flow as done according to previous *intervention()* algorithm; neverthless, the link $o-d$ will be much frequently intervened on, as it will always be part of all cuts.

# %%
G3 = nx.DiGraph(G)
G3.add_edges_from([
    ('o', 'd', {'capacity': 1}),
])

num_iterations = 26
maxflow = [nx.flow.maximum_flow_value(G, 'o', 'd')] + [intervetion(G3) for i in range(1, num_iterations+1)]

plt.plot(np.arange(0, num_iterations+1), maxflow, drawstyle='steps-post')
plt.show()

# %% [markdown]
# ### Exercise 2

# %% [markdown]
# Consider the simple graph in figure below

# %%
star = nx.star_graph(np.arange(8, 15))
star = nx.relabel_nodes(
    star,
    mapping={
        8: 9,
        9: 8,    
    }
)
star.add_node(7)
star.add_edge(8, 7)

complete = nx.complete_graph(np.arange(1, 7))
complete.add_node(15)
complete.add_edge(6, 15)

G = nx.compose(complete, star)
G.add_edge(
    6, 7   
)

pos = {
    # Star graph (left side)
    9: (0, 0),
    10: (0, 1),
    11: (-1, 1),
    12: (-1, 0), 
    13: (-1, -1),
    14: (0, -1),

    # Linking chain (middle)
    8: (2, 0),
    7: (4, 0),

    # Complete graph (right) 
    6: (6, 0),
    1: (7, -1),
    2: (9, -1),
    3: (10, 0),
    4: (7, 1),
    5: (9, 1),

    # Extra node attached to n6
    15: (5, 1)
}

nx.draw(G, with_labels=True, pos=pos)

# %% [markdown]
# Compute the Katz centrality, with $β = 0.15$ and uniform intrinsic centrality $\mu$.

# %%
def katz(G, beta=0.15, mu=None, tol=1e-5):

    N = len(G.nodes)
    W = nx.adjacency_matrix(G)
    W = W.toarray()

    w,v = np.linalg.eig(W)
    w = w.real

    lambda_max = max(w) 

    if mu is None:
        mu = np.ones((N,1))/N

    z_0 = np.ones((N,1))/N
    z_old = z_0

    while True:
        z_new = W.T @ z_old * (1-beta)/lambda_max + beta * mu
        if np.linalg.norm(z_new-z_old) < tol:
            break
        z_old=z_new

    zk_approx = z_new

    zk_approx = zk_approx / sum(zk_approx)
    return zk_approx

# %% [markdown]
# Write a distributed algorithm for the computation of Page-rank centrality, with $β = 0.15$ and
# uniform intrinsic centrality $\mu$

# %%
# **Remark**: Notice that the proposed method is **distributed**, 
# i.e., the operations at single node levels do not require a complete
# knowledge of the network. Each node $i$ updates its state $z_i(t+1)$
# by using only local information, i.e., the i-th row of $W$ and the
# state $z_j(t)$ of nodes $j$ that are adjacent to $j$.

def pagerank(G, beta=0.15, mu=None, tol=1e-5):

    # ! Is this really distributed? As it still needs to know N, P, W, D, ... 
    N = len(G.nodes)
    W = nx.adjacency_matrix(G)
    W = W.toarray()
    D = np.diag(W.sum(axis=1))
    P = np.linalg.inv(D) @ W

    w,v = np.linalg.eig(W)
    w = w.real

    if mu is None:
        mu = np.ones((N,1))/N

    z_0 = np.ones((N,1))/N
    z_old = z_0

    while True:
        z_new = P.T @ z_old * (1-beta) + beta * mu
        if np.linalg.norm(z_new-z_old) < tol:
            break
        z_old=z_new

    zb_approx = z_new

    zb_approx = zb_approx / sum(zb_approx)
    return zb_approx

# %%
def draw_centralities(G, capacities):
    nx.draw(G, pos,
            with_labels=True,
            nodelist=G.nodes,
            node_size = [d*10000 for d in capacities],  
            node_color=list(capacities),
            font_size=8,
            cmap=plt.cm.Reds,
            )

# %%
katzz = katz(G)
print(katzz)
draw_centralities(G, katzz)

# %%
pagez = pagerank(G)
print(pagez)
draw_centralities(G, pagez)

# %% [markdown]
# Explain the results of points (a) and (b), focusing on the centralities of nodes n6 and n9
# 
# > *Answer:*
# > ...

# %% [markdown]
# Compute the Page-Rank centrality with uniform intrinsic centrality $µ$ and for every $β ∈ {0, 1/4, 1/2, 3/4, 1}$

# %%
draw_centralities(G, pagerank(G, 0.0))

# %%
draw_centralities(G, pagerank(G, 0.25))

# %%
draw_centralities(G, pagerank(G, 0.50))

# %%
draw_centralities(G, pagerank(G, 0.75))

# %%
draw_centralities(G, pagerank(G, 1.0))

# %%
betas = [0.0, 0.25, 0.5, 0.75, 1.0]
n6z, n9z = [], []
for beta in betas:
    pagez = pagerank(G, beta)
    n6z.append(pagez[5])
    n9z.append(pagez[8])

print(f"{n6z=}")
print(f"{n9z=}")

plt.plot(np.arange(len(betas)), n6z, label='Node 6')
plt.plot(np.arange(len(betas)), n9z, label='Node 9')
plt.legend()
plt.show()

# %% [markdown]
# Is the difference between the centrality of node $n_6$ and node $n_9$ monotone in $β$? Explain the
# results, focusing in particular on the extreme values of $β$.
# 
# > *Answer:* ...

# %% [markdown]
# ### Exercise 3

# %%
B = sp.io.loadmat('traffic.mat')["traffic"]
l = sp.io.loadmat('traveltime.mat')["traveltime"].reshape(28,)
C = sp.io.loadmat('capacities.mat')["capacities"].reshape(28,)
f = sp.io.loadmat('flow.mat')["flow"].reshape(28,)

G = nx.DiGraph()

for edge, length, capacity in zip(B.T, l, C):
    tail_ndx = edge.argmax()
    head_ndx = edge.argmin()
    print(f"Adding link {tail_ndx+1} -> {head_ndx+1}\twith {capacity=}, {length=}")
    G.add_edge((tail_ndx+1), (head_ndx+1), capacity=int(capacity), length=length,)

nx.draw(G, pos=nx.spring_layout(G, seed=0) , with_labels=True)

# %% [markdown]
# Find the shortest path between node 1 and 17. This is equivalent to the fastest path (path
# with shortest traveling time) in an empty network

# %%
nx.shortest_path(G, 1, 17, weight='length')

# %% [markdown]
# Find the maximum flow between node 1 and 17.

# %%
nx.flow.maximum_flow_value(G, 1, 17, capacity='capacity')

# %% [markdown]
# Given the flow vector in flow.mat, compute the vector ν satisfying Bf = ν

# %%
nu = B @ f
print(nu)

# %% [markdown]
# In the following, we assume that the exogenous inflow is zero in all the nodes except for node $1$, for which $ν_1$ has the same value computed in the point (c), and node $17$, for which $ν_{17} = −ν_1$

# %%
nu_opt = np.zeros_like(nu)
nu_opt[0] = nu[0]
nu_opt[16] = -nu[0]
nu_war = np.array(nu_opt)
print(nu_opt)

# %% [markdown]
# Find the social optimum $f^∗$ with respect to the delays on the different links $τ_e(f_e)$.
# 
# For this, minimize the cost function given by:

# %%
tau = lambda f, l, c : l / (1-f/c)
phi = lambda f, l, c : f * tau(f, l, c)
phi2 = lambda f, l, c : (l * c)/(1-f/c) - l*c

# %%
import cvxpy as cp

# Construct the problem
n_edges = B.shape[1]
f_opt = cp.Variable(n_edges)
obj_opt = cp.Minimize(
    cp.sum(cp.multiply(cp.multiply(l, C), cp.power(1 - cp.multiply(f_opt, cp.inv_pos(C)), -1)) - cp.multiply(l, C))
)
const_opt = [B @ f_opt == nu_opt, f_opt >=0]
prob_opt = cp.Problem(obj_opt, const_opt)

# Solve the problem
cost_opt = prob_opt.solve()
flow_opt = f_opt.value
print("Social optimal flow:", flow_opt)
print("Optimal cost:", cost_opt)

# %% [markdown]
# Find the Wardrop equilibrium

# %%
# Construct the problem
n_edges = B.shape[1]
f_war = cp.Variable(n_edges)
obj_war = cp.Minimize(
    cp.sum(-1 * cp.multiply(cp.multiply(l, C), cp.log(1 - cp.multiply(f_war, cp.inv_pos(C)))))
) # ! Can't use ln as there is only log
const_war = [B @ f_war == nu_opt, f_war >=0]
prob_war = cp.Problem(obj_war, const_war)

# Solve the problem
cost_war = prob_war.solve()
flow_war = f_war.value
print("Wardrop optimal flow:", flow_war)
print("Optimal cost:", cost_war)

# %%

social_cost_at_wardrop = np.sum((l * C) / (1 - flow_war / C) - (l * C))

print(f"Social Optimum Cost (min total delay): {cost_opt}")
print(f"Wardrop Equilibrium's Total Delay:     {social_cost_at_wardrop}")

price_of_anarchy = social_cost_at_wardrop / cost_opt
print(f"Price of Anarchy: {price_of_anarchy}")

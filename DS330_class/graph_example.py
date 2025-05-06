import matplotlib.pyplot as plt
import networkx as nx

# make a list of nodes and edges
nodes = list(range(9)) # 0 - 8, name of nodes in this case is numbers
edges = [(1, 0), (2, 1), (3, 2), (4, 1), (5, 0),
         (0, 5), (6, 3), (7, 3), (8, 0)]

# if you want an undirected graph use nx.gGraph()
gr = nx.DiGraph()
gr.add_nodes_from(nodes)
gr.add_edges_from(edges)


pr = nx.pagerank(gr, max_iter = 1000)
print(pr)
# setting position
# spring layout starts from random numbers 
pos = nx.spring_layout(gr)

# manually draw nodes and edges
nx.draw_networkx_nodes(gr, pos = pos, node_size = [pr[node] * 300 for node in nodes]) # position from spring layout
nx.draw_networkx_edges(gr, pos = pos)
plt.show()
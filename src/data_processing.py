import pandas as pd
import networkx as nx
from itertools import combinations

base = "./data/WY/56001"

p = pd.read_csv(f"{base}/people.txt", sep="\t")
gqp = pd.read_csv(f"{base}/gq_people.txt", sep="\t")

# Person -> place edges
edges = []
for _, r in p.iterrows():
  pid = f"P:{r.sp_id}"
  edges.append((pid, f"H:{r.sp_hh_id}", "household"))
  if r.school_id != "X":
      edges.append((pid, f"S:{r.school_id}", "school"))
  if r.work_id != "X":
      edges.append((pid, f"W:{r.work_id}", "work"))

for _, r in gqp.iterrows():
  edges.append((f"P:{r.sp_id}", f"G:{r.sp_gq_id}", "gq"))

B = nx.Graph()
for u, v, t in edges:
  B.add_node(u, kind="person")
  B.add_node(v, kind="place")
  B.add_edge(u, v, rel=t)

# Optional person-person projection (shared places)
G = nx.Graph()
places = [n for n, d in B.nodes(data=True) if d["kind"] == "place"]
for place in places:
  people = [n for n in B.neighbors(place) if B.nodes[n]["kind"] == "person"]
  rel = place.split(":")[0]
  for a, b in combinations(people, 2):
      if G.has_edge(a, b):
          G[a][b]["weight"] += 1
          G[a][b]["rels"].add(rel)
      else:
          G.add_edge(a, b, weight=1, rels={rel})

print("bipartite:", B.number_of_nodes(), B.number_of_edges())
print("person-person:", G.number_of_nodes(), G.number_of_edges())

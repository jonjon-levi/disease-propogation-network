from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd


def _largest_connected_component_subgraph(graph):
    if graph.number_of_nodes() == 0:
        return graph.copy()
    largest_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_nodes).copy()


def _add_person_place_edges(people_df, gq_people_df):
    edges = []

    for _, row in people_df.iterrows():
        person_id = f"P:{row.sp_id}"
        edges.append((person_id, f"H:{row.sp_hh_id}", "household"))
        if row.school_id != "X":
            edges.append((person_id, f"S:{row.school_id}", "school"))
        if row.work_id != "X":
            edges.append((person_id, f"W:{row.work_id}", "work"))

    for _, row in gq_people_df.iterrows():
        edges.append((f"P:{row.sp_id}", f"G:{row.sp_gq_id}", "gq"))

    return edges


def build_bipartite_graph_from_dataframes(people_df, gq_people_df):
    """
    Build a person-place bipartite graph.

    Person nodes are prefixed with `P:` and place nodes with one of
    `H:`, `S:`, `W:`, `G:` for household/school/work/group-quarters.
    """
    graph = nx.Graph()
    for u, v, rel in _add_person_place_edges(people_df, gq_people_df):
        graph.add_node(u, kind="person")
        graph.add_node(v, kind="place")
        graph.add_edge(u, v, rel=rel)
    return graph


def project_people_graph(bipartite_graph):
    """
    Project a person-place bipartite graph to a weighted person-person graph.
    """
    person_graph = nx.Graph()
    place_nodes = [
        node for node, data in bipartite_graph.nodes(data=True) if data.get("kind") == "place"
    ]

    for place in place_nodes:
        people = [
            neighbor
            for neighbor in bipartite_graph.neighbors(place)
            if bipartite_graph.nodes[neighbor].get("kind") == "person"
        ]
        rel = place.split(":")[0]
        for a, b in combinations(people, 2):
            if person_graph.has_edge(a, b):
                person_graph[a][b]["weight"] += 1
                person_graph[a][b]["rels"].add(rel)
            else:
                person_graph.add_edge(a, b, weight=1, rels={rel})

    return person_graph


def load_wy_county_graphs(base_path):
    """
    Load WY county raw files and return (bipartite_graph, person_projection_graph).

    `base_path` can be absolute or relative to the project root.
    """
    base = Path(base_path)
    people_df = pd.read_csv(base / "people.txt", sep="\t")
    gq_people_df = pd.read_csv(base / "gq_people.txt", sep="\t")

    bipartite = build_bipartite_graph_from_dataframes(people_df, gq_people_df)
    person_projection = project_people_graph(bipartite)

    bipartite_lcc = _largest_connected_component_subgraph(bipartite)
    person_projection_lcc = _largest_connected_component_subgraph(person_projection)
    return bipartite_lcc, person_projection_lcc

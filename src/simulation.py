import numpy as np
import networkx as nx
import random

def simulate_sir(
    G,                      # G = (V,E) where nodes represent people and edges represent interactions
    beta,                   # transmition rate (higher beta faster spread)
    gamma,                  # recovery rate (higher gamma faster recovery)
    steps,                  # modeling time
    initial_infected,       # initial state of the disease
    weight="weight",        # strength of interaction between people
    vulnerability=None,     # risk of person to get infected
    immunized=None,         # set of nodes that start in state R (cannot be infected)
    mu=0.0,                 # death rate (probability an infected person is permanently removed)
    rng=None,               # randomness (allows usage of same seed to recreate trials)
):
    """
    Discrete-time SIR simulation on a graph.

    States:
        0 = Susceptible
        1 = Infected
        2 = Recovered (immune)
        3 = Removed / dead
    """

    if rng is None:
        rng = np.random.default_rng()

    if vulnerability is None:
        vulnerability = {}

    if immunized is None:
        immunized = set()

    # initialize all nodes as susceptible
    state = {}
    for v in G.nodes():
        state[v] = 0

    # immunized nodes start recovered
    for v in immunized:
        if v in state:
            state[v] = 2

    # initial infected
    for v in initial_infected:
        if v in state and v not in immunized:
            state[v] = 1

    def count_states():
        vals = list(state.values())
        return (
            vals.count(0),
            vals.count(1),
            vals.count(2),
            vals.count(3),
        )

    S, I, R, D = [], [], [], []
    s, i, r, d = count_states()
    S.append(s); I.append(i); R.append(r); D.append(d)

    for _ in range(steps):
        new_state = state.copy()

        # infection step
        for v in G.nodes():
            if state[v] != 0:
                continue

            pressure = 0.0
            # for every infected neighbor increment pressure
            for u in G.neighbors(v):
                if state[u] == 1:
                    pressure += G[u][v].get(weight, 1.0)

            if pressure == 0:
                continue

            vul = vulnerability.get(v, 1.0)
            # Poission Process calculated infection probability
            p_inf = 1.0 - np.exp(-beta * pressure * vul) 

            # simulates a Bernoulli trial given p_inf
            if rng.random() < p_inf:
                new_state[v] = 1

        # recovery / removal step
        for v in G.nodes():
            if state[v] != 1:
                continue

            if mu > 0 and rng.random() < mu:
                new_state[v] = 3
            elif rng.random() < gamma:
                new_state[v] = 2

        state = new_state
        s, i, r, d = count_states()
        S.append(s); I.append(i); R.append(r); D.append(d)

        if i == 0:
            break

    return {"S": S, "I": I, "R": R, "D": D}

def get_degree(pair):
    return pair[1]

def choose_immunized_by_degree(G, k):
    nodes = sorted(G.degree, key=get_degree, reverse=True)
    immunized = set()
    for node, _ in nodes[:k]:
        immunized.add(node)
    return immunized

def choose_immunized_random(G, k):
    nodes = random.sample(list(G.nodes()), k) 
    immunized = set()
    for node in nodes[:k]:
        immunized.add(node)
    return immunized

def choose_immunized_by_eigenvector_centrality(
    G,
    k,
    weight="weight",
    max_iter=1000,
    tol=1.0e-6,
):
    if k <= 0:
        return set()

    k = min(k, G.number_of_nodes())
    centrality = nx.eigenvector_centrality(
        G,
        max_iter=max_iter,
        tol=tol,
        weight=weight,
    )
    ranked_nodes = sorted(centrality.items(), key=lambda pair: pair[1], reverse=True)

    immunized = set()
    for node, _ in ranked_nodes[:k]:
        immunized.add(node)
    return immunized

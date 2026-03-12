import numpy as np
import networkx as nx
import random
import math
import hashlib

def simulate_sir(
    G,                      # G = (V,E) where nodes represent people and edges represent interactions
    beta,                   # transmission rate (higher beta faster spread)
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

    total_inf = 0

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
                total_inf += 1

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

    return {"S": S, "I": I, "R": R, "D": D, "Total_inf": total_inf}

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

def choose_immunized_by_betweenness_centrality(
    G,
    k,
    weight="weight",
    normalized=True,
):
    if k <= 0:
        return set()

    k = min(k, G.number_of_nodes())
    centrality = nx.betweenness_centrality(
        G,
        normalized=normalized,
        weight=weight,
    )
    ranked_nodes = sorted(centrality.items(), key=lambda pair: pair[1], reverse=True)

    immunized = set()
    for node, _ in ranked_nodes[:k]:
        immunized.add(node)
    return immunized

def choose_immunized_by_adaptive_degree(G, k):
    if k <= 0:
        return set()

    target = min(k, G.number_of_nodes())
    work_graph = G.copy()
    immunized = set()

    while len(immunized) < target and work_graph.number_of_nodes() > 0:
        node, _ = max(work_graph.degree, key=lambda pair: pair[1])
        immunized.add(node)
        work_graph.remove_node(node)

    return immunized

def choose_immunized_by_highest(G, k):
    nodes = list(G.nodes())
    pop = len(nodes)

    if k <= 0 or pop == 0:
        return set()

    target = min(k, pop)
    immunized = set()

    # Interview a random batch, then vaccinate the top degree nodes from that batch.
    # Re-sample until we hit the target; this guarantees progress for all k in [0, pop].
    while len(immunized) < target:
        remaining = [node for node in nodes if node not in immunized]
        if not remaining:
            break

        batch_size = max(1, math.ceil(0.1 * len(remaining)))
        batch_size = min(batch_size, len(remaining))
        batch = random.sample(remaining, batch_size)

        ranked_batch = sorted(batch, key=lambda node: G.degree[node], reverse=True)
        take = max(1, math.ceil(0.25 * batch_size))
        take = min(take, target - len(immunized))

        for node in ranked_batch[:take]:
            immunized.add(node)

    return immunized

def choose_immunized_by_neighbor_nomination(G, k):
    """
    Chain-based neighbor nomination immunization.

    Process:
    1) Pick a random person as the current nominator.
    2) Current nominator nominates one of their neighbors for immunization.
    3) The newly immunized person becomes the next nominator.
    4) Repeat until k people are immunized.

    Robustness for large k:
    - If the current nominator has no eligible neighbors, restart from a node that
      can still nominate someone new.
    - If no such node exists, fill the remaining quota from the un-immunized nodes.
    """
    nodes = list(G.nodes())
    pop = len(nodes)
    if k <= 0 or pop == 0:
        return set()

    target = min(k, pop)
    immunized = set()
    current = random.choice(nodes)

    while len(immunized) < target:
        eligible_neighbors = [
            neighbor
            for neighbor in G.neighbors(current)
            if neighbor not in immunized
        ]

        if eligible_neighbors:
            nominee = random.choice(eligible_neighbors)
            immunized.add(nominee)
            current = nominee
            continue

        restart_candidates = [
            node
            for node in nodes
            if node not in immunized
            and any(neighbor not in immunized for neighbor in G.neighbors(node))
        ]

        if restart_candidates:
            current = random.choice(restart_candidates)
            continue

        # Saturated nomination graph: everyone left is isolated from un-immunized nodes.
        # Fill remaining quota directly to guarantee we hit `target`.
        remaining = [node for node in nodes if node not in immunized]
        take = min(target - len(immunized), len(remaining))
        if take > 0:
            immunized.update(random.sample(remaining, take))
        break

    return immunized


def stable_seed(base_seed, county_id, strategy_name, k_count, trial):
    """Generates a reproducible seed for Monte Carlo trials."""
    text = f"{base_seed}_{county_id}_{strategy_name}_{k_count}_{trial}"
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2 ** 32)


def evaluate_strategy_mc(args):
    """
    Worker function to run all Monte Carlo trials for a single strategy.
    """
    (strategy_name, strategy_fn, G, k_count, county_id, n_trials,
     beta, gamma, steps, n_infected, base_seed) = args

    deterministic_strategies = ["Degree", "Eigenvector", "Betweenness", "Adaptive Degree"]

    # 1. Pre-calculate if Deterministic
    if strategy_name == "baseline":
        base_immunized = set()
    elif strategy_name in deterministic_strategies:
        seed0 = stable_seed(base_seed, county_id, strategy_name, k_count, 0)
        random.seed(seed0)
        base_immunized = strategy_fn(G, k_count)
    else:
        base_immunized = None

    strategy_curves = []
    total_outbreaks = []  # <--- New list to track total ever infected

    # 2. Run the N_TRIALS Monte Carlo loop
    for trial in range(n_trials):
        seed_trial = stable_seed(base_seed, county_id, strategy_name, k_count, trial)
        start_rng = random.Random(seed_trial + 1)

        if strategy_name == "baseline" or strategy_name in deterministic_strategies:
            immunized = base_immunized
        else:
            random.seed(seed_trial)
            immunized = strategy_fn(G, k_count)

        eligible = [node for node in G.nodes if node not in immunized]
        n_start = min(n_infected, len(eligible))
        initial_infected = start_rng.sample(eligible, n_start) if n_start > 0 else []

        out = simulate_sir(
            G,
            beta=beta,
            gamma=gamma,
            steps=steps,
            initial_infected=initial_infected,
            immunized=immunized,
            rng=np.random.default_rng(seed_trial + 2),
        )

        strategy_curves.append(out["I"])

        # Total outbreak size = (Patient Zeros) + (New Infections during simulation)
        total_outbreaks.append(len(initial_infected) + out["Total_inf"])

    # 3. Padding logic for time-series curves
    max_len = max(len(curve) for curve in strategy_curves)
    padded_curves = [np.pad(c, (0, max_len - len(c)), mode='constant') for c in strategy_curves]

    # Return the name, the mean time-series curve, AND the mean total outbreak size
    return strategy_name, np.mean(padded_curves, axis=0), np.mean(total_outbreaks)
# Disease Propagation on Networks

## Overview

This project studies how infectious diseases spread through social networks and how different immunization strategies can reduce the size of an outbreak. This project studies infectious disease spread using network-based models. We simulate diffusion processes on weighted graphs and analyze the effect
of targeted immunization using differently informed methods of degree centralization. We simulate disease propagation using a **Susceptible–Infected–Recovered (SIR)** model on a realistic synthetic social contact network 
derived from U.S. Census data and alternative sampled data.

The goal of this project is to determine which individuals should be vaccinated when vaccination is slow, costly or logisitically difficult.. In particular, we compare several strategies that target highly connected individuals in the network.

The simulations are run on a realistic contact network constructed from the **FRED U.S. Synthetic Population dataset**, which approximates real-world social interactions in areas such as households, schools, and workplaces.

---

## Structure
- `src/` – core simulation and graph code
- `data/` – datasets (not tracked if large)
- `notebooks/` – experiments and visualization

## Model

We run a stochastic discrete-time SIR pocess on a network \(G = (V,E)\), where each node represents an individual and edges represent contacts between individuals.

Each node can be in one of three states:

- **S** – Susceptible  
- **I** – Infected  
- **R** – Recovered (or immunized)

At each time step:

- A susceptible node becomes infected with probability determined by its infected neighbors and a transmission rate \( \beta \).
- An infected node recovers with probability \( \gamma \).
- Recovered nodes remain immune for the rest of the simulation.

Before the epidemic begins, a subset of nodes \(U \subseteq V\) is chosen to be immunized. These nodes are initialized in the recovered state and cannot become infected.

The objective is to choose the immunized set \(U\) to minimize the peak number of infections or the total number of infected individuals during the outbreak.

---

## Network Construction

The contact network is constructed from synthetic population data. The dataset contains information about individuals and the places they interact, such as:

- Households  
- Schools  
- Workplaces  
- Group quarters  

First, a **bipartite graph** is created connecting people to the locations they belong to. 

Next, this bipartite graph is projected into a **person–person contact network**, where two individuals are connected if they share a location. If two people share multiple locations, the edge weight between them increases.

This resulting network represents the contact structure used in the SIR simulations.

---

## Immunization Strategies

Several vaccination strategies are tested in order to compare their effectiveness.

### Random Immunization
A baseline strategy where \(k\) individuals are selected uniformly at random to be vaccinated.

### Degree Centrality
Vaccinates the \(k\) individuals with the highest number of connections in the network.

### Adaptive Degree
An extension of degree centrality where the highest degree node is vaccinated and removed from the network, then degrees are recalculated before selecting the next node.

### Eigenvector Centrality
Targets individuals that are connected to other highly connected individuals in the network.

### Neighbor Nomination
A practical strategy where randomly selected individuals nominate a contact to be vaccinated.

### Self-Polling
A subset of individuals is sampled and asked how many contacts they have. The individuals with the highest reported number of contacts are vaccinated.

---

## Simulation Setup

Simulations are implemented in **Python** using several scientific computing libraries.

Key parameters include:

- **β (beta)** – infection transmission rate  
- **γ (gamma)** – recovery probability  
- **steps** – number of simulation time steps  
- **k** – number of vaccinated individuals  

Multiple simulation runs can be used to estimate expected outcomes and compare strategies.

---

## Example Experiments

The simulations explore several scenarios:

### Baseline Epidemic
Runs the SIR model without vaccination to observe the natural spread of the disease.

### Strategy Comparison
Compares infection curves for different immunization strategies.

### Vaccination Budget Analysis
Studies how the peak number of infected individuals changes as the number of vaccinated individuals increases.

---

## Technologies Used

This project was implemented using the following tools:

- Python  
- NetworkX  
- NumPy  
- Pandas  
- Matplotlib  

---

## Running the Code

Note: Before following this, git for windows or mac must be downloaded (https://git-scm.com/install/)

1. Clone the repository
```bash
git clone https://github.com/yourusername/network-sir-simulation.git
```
2. Navigate to the project directory
```bash
cd network-sir-simulation
```
3. Install required packages
```bash
pip install numpy pandas networkx matplotlib
```
4. Run the simulation script
```bash
python sir_simulation.py
```


The program will load the dataset, build the contact network, run the SIR simulations, and generate plots showing the epidemic dynamics.

---

## Data Source

The network is based on the **FRED U.S. Synthetic Population dataset**, which is derived from U.S. Census data and other demographic sources to approximate realistic contact networks.

In this project we use a subset corresponding to **Albany County, Wyoming**.

---

## Limitations

There are several limitations to this project:

- Simulations were limited by available computational power.  
- Only a small regional population dataset was used instead of a full national dataset.  
- A limited number of Monte Carlo simulations were performed.

Future work could explore larger datasets, additional vaccination strategies, and more detailed epidemic models.

---

## Authors

E. Cruz-Martinez  
N. Delingat  
T. Eapen  
L. Fiechter  
J. Levi  

Department of Mathematics  
University of California, Los Angeles

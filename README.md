# AI_Quantum_Computing_Payments_network_Routing




from flask import Flask, request, render_template, jsonify
import numpy as np
import networkx as nx
import threading
import time
import random
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_aer import AerSimulator
import tensorflow as tf
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit, transpile

app = Flask(__name__)

# Global variables for real-time simulation
network_load = {}
simulation_running = False
simulation_thread = None
node_trust_levels = {}

def create_payment_network():
    G = nx.Graph()
    G.add_nodes_from(["NY Bank", "London Hub", "Tokyo Processor", "Sydney Node", 
                     "Singapore Exchange", "Frankfurt Center", "Mumbai Relay"])
    
    # Add edges with cost, latency, and base attributes
    G.add_edge("NY Bank", "London Hub", cost=5.0, latency=1.5, base_traffic=0.3)
    G.add_edge("NY Bank", "Tokyo Processor", cost=8.0, latency=3.0, base_traffic=0.5)
    G.add_edge("NY Bank", "Frankfurt Center", cost=6.0, latency=1.8, base_traffic=0.4)
    G.add_edge("London Hub", "Tokyo Processor", cost=4.0, latency=2.0, base_traffic=0.6)
    G.add_edge("London Hub", "Sydney Node", cost=7.0, latency=2.5, base_traffic=0.2)
    G.add_edge("London Hub", "Frankfurt Center", cost=2.0, latency=0.8, base_traffic=0.7)
    G.add_edge("London Hub", "Mumbai Relay", cost=5.5, latency=2.2, base_traffic=0.3)
    G.add_edge("Tokyo Processor", "Sydney Node", cost=3.0, latency=1.0, base_traffic=0.4)
    G.add_edge("Tokyo Processor", "Singapore Exchange", cost=2.5, latency=1.2, base_traffic=0.5)
    G.add_edge("Sydney Node", "Singapore Exchange", cost=2.8, latency=1.1, base_traffic=0.3)
    G.add_edge("Singapore Exchange", "Mumbai Relay", cost=3.2, latency=1.3, base_traffic=0.2)
    G.add_edge("Frankfurt Center", "Mumbai Relay", cost=4.8, latency=1.9, base_traffic=0.1)
    
    # Initialize trust levels for nodes
    global node_trust_levels
    node_trust_levels = {
        "NY Bank": {"level": "High", "score": 0.9},
        "London Hub": {"level": "High", "score": 0.95},
        "Tokyo Processor": {"level": "Medium", "score": 0.7},
        "Sydney Node": {"level": "High", "score": 0.85},
        "Singapore Exchange": {"level": "Medium", "score": 0.75},
        "Frankfurt Center": {"level": "High", "score": 0.9},
        "Mumbai Relay": {"level": "Low", "score": 0.5}
    }
    
    # Initialize network load
    global network_load
    for u, v, data in G.edges(data=True):
        network_load[(u, v)] = data['base_traffic']
        network_load[(v, u)] = data['base_traffic']  # Undirected graph
    
    return G

def update_network_load():
    """Simulate real-time changes in network load"""
    global network_load, simulation_running
    G = create_payment_network()
    
    while simulation_running:
        # For each edge, update its load randomly but with some memory of previous load
        for u, v, data in G.edges(data=True):
            # Previous load has 70% influence, new random component has 30% influence
            current_load = network_load.get((u, v), data['base_traffic'])
            random_factor = random.uniform(-0.2, 0.2)  # Random change in load
            new_load = max(0.1, min(0.9, current_load * 0.7 + (data['base_traffic'] + random_factor) * 0.3))
            network_load[(u, v)] = new_load
            network_load[(v, u)] = new_load  # Undirected graph
        
        # Sleep for a few seconds before updating again
        time.sleep(5)

def get_real_time_network():
    """Get the network with real-time load affecting latency and cost"""
    G = create_payment_network()
    G_rt = G.copy()
    
    # Update edge attributes based on current network load
    for u, v, data in G_rt.edges(data=True):
        load = network_load.get((u, v), data['base_traffic'])
        # Latency increases with load
        data['current_latency'] = data['latency'] * (1 + load)
        # Cost can also be affected by congestion
        data['current_cost'] = data['cost'] * (1 + load * 0.5)
        # Add current load as attribute
        data['current_load'] = load
    
    return G_rt

def ai_predict_weights(G, optimization_mode="balanced"):
    """
    Predict edge weights using AI model with multiple objectives:
    - Cost
    - Latency
    - Trust/Security
    - Current network load
    """
    # Create a larger synthetic dataset based on network properties
    # Collect real edge data from graph
    edges_data = []
    for u, v, data in G.edges(data=True):
        # Get current metrics
        cost = data.get('current_cost', data['cost'])
        latency = data.get('current_latency', data['latency'])
        load = data.get('current_load', data.get('base_traffic', 0.5))
        
        # Calculate trust scores for this edge (average of both nodes)
        source_trust = node_trust_levels.get(u, {"score": 0.7})["score"]
        target_trust = node_trust_levels.get(v, {"score": 0.7})["score"]
        edge_trust = (source_trust + target_trust) / 2
        
        edges_data.append([cost, latency, load, edge_trust])
    
    edges_data = np.array(edges_data)
    
    # Calculate statistics to generate more realistic data
    mean_values = np.mean(edges_data, axis=0)
    std_values = np.std(edges_data, axis=0)
    
    # Generate synthetic data with similar properties to our real edges
    num_synthetic = 100
    np.random.seed(42)  # For reproducibility
    
    synthetic_data = np.zeros((num_synthetic, 4))
    for i in range(4):
        synthetic_data[:, i] = np.random.normal(mean_values[i], std_values[i], num_synthetic)
    
    # Ensure positive values and proper ranges
    synthetic_data[:, 0] = np.maximum(0.5, synthetic_data[:, 0])  # cost
    synthetic_data[:, 1] = np.maximum(0.3, synthetic_data[:, 1])  # latency
    synthetic_data[:, 2] = np.clip(synthetic_data[:, 2], 0.1, 0.9)  # load
    synthetic_data[:, 3] = np.clip(synthetic_data[:, 3], 0.1, 1.0)  # trust
    
    # Create target weights based on the selected optimization mode
    if optimization_mode == "cost_focused":
        # Prioritize cost over other factors
        weight_factors = {
            'base_cost': 1.0,
            'cost_factor': 2.0,
            'latency_factor': 0.5,
            'load_factor': 0.8,
            'trust_factor': -0.7  # Negative because higher trust should reduce weight
        }
    elif optimization_mode == "speed_focused":
        # Prioritize latency/speed
        weight_factors = {
            'base_cost': 1.0,
            'cost_factor': 0.5,
            'latency_factor': 2.0,
            'load_factor': 1.5,
            'trust_factor': -0.7
        }
    elif optimization_mode == "security_focused":
        # Prioritize security/trust
        weight_factors = {
            'base_cost': 1.0,
            'cost_factor': 0.5,
            'latency_factor': 0.8,
            'load_factor': 0.7,
            'trust_factor': -2.0
        }
    else:  # "balanced" is the default
        # Balanced approach
        weight_factors = {
            'base_cost': 1.0,
            'cost_factor': 1.0,
            'latency_factor': 1.0,
            'load_factor': 1.0,
            'trust_factor': -1.0
        }
    
    # Calculate synthetic weights
    synthetic_weights = (
        weight_factors['base_cost'] + 
        (weight_factors['cost_factor'] * synthetic_data[:, 0]) + 
        (weight_factors['latency_factor'] * synthetic_data[:, 1]) +
        (weight_factors['load_factor'] * synthetic_data[:, 2]) +
        (weight_factors['trust_factor'] * synthetic_data[:, 3])
    )
    
    # Ensure all weights are positive
    synthetic_weights = np.maximum(0.1, synthetic_weights)
    
    # Create training data
    X = synthetic_data
    y = synthetic_weights
    
    # Build and train model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu")  # ReLU ensures positive outputs
    ])
    
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=50, verbose=0, validation_split=0.2)
    
    # Predict weights for actual graph edges and store metadata for explainability
    weights = {}
    edge_metadata = {}
    
    for u, v, data in G.edges(data=True):
        # Get edge metrics
        cost = data.get('current_cost', data['cost'])
        latency = data.get('current_latency', data['latency'])
        load = data.get('current_load', data.get('base_traffic', 0.5))
        
        # Calculate trust scores for this edge (average of both nodes)
        source_trust = node_trust_levels.get(u, {"score": 0.7})["score"]
        target_trust = node_trust_levels.get(v, {"score": 0.7})["score"]
        edge_trust = (source_trust + target_trust) / 2
        
        # Make prediction
        features = np.array([[cost, latency, load, edge_trust]])
        pred = model.predict(features, verbose=0)[0][0]
        pred = max(0.1, pred)  # Ensure minimum positive weight
        
        # Store weights and metadata
        weights[(u, v)] = pred
        weights[(v, u)] = pred  # Graph is undirected
        
        # Store metadata for explainability
        edge_metadata[(u, v)] = {
            'cost': cost,
            'latency': latency,
            'load': load,
            'trust': edge_trust,
            'weight': pred,
            'factors': {
                'cost_contribution': weight_factors['cost_factor'] * cost / pred,
                'latency_contribution': weight_factors['latency_factor'] * latency / pred,
                'load_contribution': weight_factors['load_factor'] * load / pred,
                'trust_contribution': weight_factors['trust_factor'] * edge_trust / pred
            }
        }
        edge_metadata[(v, u)] = edge_metadata[(u, v)]  # Copy for undirected
    
    return weights, edge_metadata

def classical_dijkstra_route(G, weights, start, end):
    """Find route using classical Dijkstra's algorithm"""
    # Create a copy of the graph with the predicted weights
    G_weighted = G.copy()
    for u, v in G.edges():
        G_weighted[u][v]['weight'] = weights.get((u, v), 10.0)
    
    try:
        # Get shortest path
        path = nx.shortest_path(G_weighted, source=start, target=end, weight='weight')
        total_cost = sum(weights.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
        return path, total_cost
    except nx.NetworkXNoPath:
        return [start, end], weights.get((start, end), 10.0)  # Direct connection if no path found

def quantum_route_optimization(G, weights, start, end):
    """Find route using quantum optimization"""
    # For debugging
    print(f"Finding route from {start} to {end} using quantum optimization")
    
    # Calculate shortest path using Dijkstra's algorithm as a fallback
    try:
        fallback_path, fallback_cost = classical_dijkstra_route(G, weights, start, end)
        print(f"Fallback path: {fallback_path}, cost: {fallback_cost}")
    except Exception as e:
        print(f"Error in classical fallback: {e}")
        fallback_path = [start, end]  # Direct connection if failed
        fallback_cost = weights.get((start, end), 10.0)  # Default high cost
    
    try:
        # Try quantum optimization approach
        qp = QuadraticProgram()
        nodes = list(G.nodes)
        
        # Create binary variables for each edge in the graph
        edge_vars = {}
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if G.has_edge(u, v):
                    var_name = f"x_{i}_{j}"
                    qp.binary_var(var_name)
                    edge_vars[(u, v)] = var_name
        
        # Objective function: minimize the total weight
        obj_linear = {}
        for (u, v), var_name in edge_vars.items():
            obj_linear[var_name] = weights.get((u, v), 0)
        qp.minimize(linear=obj_linear)
        
        # Constraints: flow conservation
        # For each node except start and end: incoming flow = outgoing flow
        for node in nodes:
            if node != start and node != end:
                flow_constraint = {}
                # Incoming edges
                for u in G.predecessors(node):
                    var_name = edge_vars.get((u, node), None)
                    if var_name:
                        flow_constraint[var_name] = 1
                # Outgoing edges
                for v in G.successors(node):
                    var_name = edge_vars.get((node, v), None)
                    if var_name:
                        flow_constraint[var_name] = -1
                
                if flow_constraint:  # Only add constraint if we have variables
                    qp.linear_constraint(flow_constraint, "==", 0)
        
        # Start node must have one outgoing edge
        start_constraint = {}
        for v in G.successors(start):
            var_name = edge_vars.get((start, v), None)
            if var_name:
                start_constraint[var_name] = 1
        
        if start_constraint:
            qp.linear_constraint(start_constraint, "==", 1)
        
        # End node must have one incoming edge
        end_constraint = {}
        for u in G.predecessors(end):
            var_name = edge_vars.get((u, end), None)
            if var_name:
                end_constraint[var_name] = 1
        
        if end_constraint:
            qp.linear_constraint(end_constraint, "==", 1)
        
        # Convert to QUBO and solve with QAOA
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        operator, offset = qubo.to_ising()
        
        # Set up QAOA with Sampler
        optimizer = SPSA(maxiter=50)
        sampler = Sampler()
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2)  # Increased reps for better results
        result = qaoa.compute_minimum_eigenvalue(operator)
        
        # Get the most probable bitstring
        if hasattr(result, 'eigenstate'):
            bitstring = max(result.eigenstate, key=result.eigenstate.get)
        elif hasattr(result, 'samples') and len(result.samples) > 0:
            # Alternative way to get result depending on Qiskit version
            bitstring = result.samples[0].x
        else:
            print("Could not get solution from QAOA result")
            return fallback_path, fallback_cost
        
        # Reconstruct the path
        selected_edges = []
        for (u, v), var_name in edge_vars.items():
            var_idx = list(qp.variables).index(qp.get_variable(var_name))
            if var_idx < len(bitstring) and bitstring[var_idx] == 1:
                selected_edges.append((u, v))
        
        # Construct the path from selected edges
        if not selected_edges:
            print("No edges selected in quantum solution")
            return fallback_path, fallback_cost
            
        # Try to construct a path using selected edges
        path = [start]
        current = start
        visited = set([start])
        max_iterations = len(nodes) * 2  # Avoid infinite loops
        iteration = 0
        
        while current != end and iteration < max_iterations:
            iteration += 1
            for u, v in selected_edges:
                if u == current and v not in visited:
                    path.append(v)
                    visited.add(v)
                    current = v
                    break
            else:
                # No valid next step found
                if iteration == 1:  # Failed on first step
                    print("Could not construct path from quantum solution")
                    return fallback_path, fallback_cost
                else:
                    # We started a path but couldn't complete it
                    # Try to find a way to the end node using shortest path
                    try:
                        G_weighted = G.copy()
                        for u, v in G.edges():
                            G_weighted[u][v]['weight'] = weights.get((u, v), 10.0)
                        completion_path = nx.shortest_path(G_weighted, source=current, target=end, weight='weight')
                        # Add all but the first node (we already have that)
                        path.extend(completion_path[1:])
                        break
                    except:
                        print("Could not complete partial quantum path")
                        return fallback_path, fallback_cost
        
        if current != end:
            print("Path construction did not reach the end node")
            return fallback_path, fallback_cost
            
        # Calculate the total cost using positive weights
        total_cost = sum(weights.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
        print(f"Quantum path: {path}, cost: {total_cost}")
        
        # If we have a valid path, return it
        if len(path) >= 2 and path[0] == start and path[-1] == end and total_cost > 0:
            return path, total_cost
        else:
            # Fallback to classical solution
            return fallback_path, fallback_cost
        
    except Exception as e:
        print(f"Error in quantum optimization: {e}")
        return fallback_path, fallback_cost

def generate_routes(G, start, end, amount, optimization_mode="balanced"):
    """Generate routes using different methods for comparison"""
    # Get real-time network with updated loads
    G_rt = get_real_time_network()
    
    # Use AI to predict weights
    weights, edge_metadata = ai_predict_weights(G_rt, optimization_mode)
    
    # Generate routes using different methods
    results = {}
    
    # Classical route
    classical_path, classical_cost = classical_dijkstra_route(G_rt, weights, start, end)
    results["classical"] = {
        "path": classical_path,
        "cost": classical_cost,
        "display": " -> ".join(classical_path)
    }
    
    # Quantum route
    quantum_path, quantum_cost = quantum_route_optimization(G_rt, weights, start, end)
    results["quantum"] = {
        "path": quantum_path,
        "cost": quantum_cost,
        "display": " -> ".join(quantum_path)
    }
    
    # Generate path explanations
    explanations = {}
    for path_type, path_data in results.items():
        path = path_data["path"]
        path_explanation = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = edge_metadata.get((u, v), {})
            
            if edge_data:
                # Find primary factor (highest absolute contribution)
                factors = edge_data.get('factors', {})
                if factors:
                    primary_factor = max(factors.items(), key=lambda x: abs(x[1]))
                    
                    if primary_factor[0] == 'cost_contribution':
                        reason = f"Low cost route"
                    elif primary_factor[0] == 'latency_contribution':
                        reason = f"Low latency connection"
                    elif primary_factor[0] == 'load_contribution':
                        reason = f"Low traffic route"
                    elif primary_factor[0] == 'trust_contribution':
                        reason = f"High security connection"
                    else:
                        reason = "Optimal balance of factors"
                else:
                    reason = "Optimal route segment"
            else:
                reason = "Direct connection"
                
            path_explanation.append({
                'from': u,
                'to': v,
                'reason': reason,
                'metrics': {
                    'cost': edge_data.get('cost', 0),
                    'latency': edge_data.get('latency', 0),
                    'load': edge_data.get('load', 0),
                    'trust': edge_data.get('trust', 0),
                }
            })
        
        explanations[path_type] = path_explanation
    
    return results, explanations, weights, edge_metadata

@app.route("/", methods=["GET", "POST"])
def home():
    G = create_payment_network()
    nodes = list(G.nodes)
    
    # Start network load simulation if not already running
    start_simulation()
    
    # Get trust levels for the UI
    trust_info = {node: info for node, info in node_trust_levels.items()}
    
    if request.method == "POST":
        start = request.form["start"]
        end = request.form["end"]
        amount = float(request.form["amount"])
        optimization_mode = request.form.get("optimization_mode", "balanced")
        routing_method = request.form.get("routing_method", "quantum")

        # Don't allow same start and end
        if start == end:
            return render_template("result.html", 
                                  amount=amount, 
                                  path="Error: Start and end nodes must be different", 
                                  total_cost=0.0)

        # Generate routes
        routes, explanations, weights, edge_metadata = generate_routes(G, start, end, amount, optimization_mode)
        
        # Get the selected route based on user preference
        if routing_method in routes:
            selected_route = routes[routing_method]
            path_explanation = explanations[routing_method]
        else:
            # Default to quantum
            selected_route = routes["quantum"]
            path_explanation = explanations["quantum"]
        
        # Get network snapshot for visualization
        network_snapshot = {}
        for (u, v), load in network_load.items():
            if u < v:  # Only include each edge once
                network_snapshot[(u, v)] = {
                    'load': load,
                    'weight': weights.get((u, v), 0)
                }
        
        return render_template("result.html", 
                              amount=amount, 
                              path=selected_route["display"], 
                              total_cost=selected_route["cost"],
                              routes=routes,
                              explanations=explanations,
                              selected_method=routing_method,
                              optimization_mode=optimization_mode,
                              trust_levels=trust_info,
                              network_snapshot=network_snapshot)
                              
    return render_template("index.html", 
                          nodes=nodes, 
                          trust_levels=trust_info)

@app.route("/get_network_status", methods=["GET"])
def get_network_status():
    """API endpoint to get current network status for real-time updates"""
    network_status = {}
    for (u, v), load in network_load.items():
        if u < v:  # Only include each edge once for undirected graph
            network_status[f"{u}_{v}"] = {
                'load': load,
                'color': get_load_color(load)
            }
    
    return jsonify(network_status)

def get_load_color(load):
    """Convert load to a color from green (low) to red (high)"""
    # Green to yellow to red
    if load < 0.3:
        return "#4CAF50"  # Green
    elif load < 0.6:
        return "#FFC107"  # Yellow/Amber
    else:
        return "#F44336"  # Red

def start_simulation():
    """Start the network load simulation if not already running"""
    global simulation_running, simulation_thread
    
    if not simulation_running:
        simulation_running = True
        simulation_thread = threading.Thread(target=update_network_load)
        simulation_thread.daemon = True  # Thread will exit when main program exits
        simulation_thread.start()

def stop_simulation():
    """Stop the network load simulation"""
    global simulation_running
    simulation_running = False

@app.route("/toggle_method/<method>", methods=["POST"])
def toggle_method(method):
    """Toggle between different routing methods"""
    # This is handled by the form submission, but we could add AJAX support here
    return jsonify({"status": "success", "method": method})

@app.route("/html_templates")
def html_templates():
    """Return HTML templates for the application"""
    templates = {
        "index": """
<!DOCTYPE html>
<html>
<head>
    <title>AI-Quantum Risk-Aware Payment Router</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .node {
            padding: 10px;
            margin: 5px;
            border-radius: 5px;
        }
        .trust-high {
            background-color: rgba(0, 255, 0, 0.2);
            border: 1px solid green;
        }
        .trust-medium {
            background-color: rgba(255, 255, 0, 0.2);
            border: 1px solid orange;
        }
        .trust-low {
            background-color: rgba(255, 0, 0, 0.2);
            border: 1px solid red;
        }
        .network-status {
            height: 10px;
            width: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        .timer {
            font-size: 0.8rem;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4">AI-Quantum Risk-Aware Payment Router</h1>
        
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5>Network Status <span class="badge bg-info">Real-Time</span></h5>
                        <div class="timer">Updating: <span id="countdown">5</span>s</div>
                    </div>
                    <div class="card-body">
                        <h6>Node Trust Levels:</h6>
                        <div class="d-flex flex-wrap">
                            {% for node, info in trust_levels.items() %}
                            <div class="node trust-{{ info.level|lower }}">
                                {{ node }} <span class="badge bg-secondary">{{ info.level }}</span>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-body">
                <h2 class="card-title mb-4">Configure Payment Route</h2>
                <form method="POST" action="/">
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="start" class="form-label">Start Node:</label>
                            <select name="start" id="start" class="form-select" required>
                                {% for node in nodes %}
                                <option value="{{ node }}">{{ node }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label for="end" class="form-label">End Node:</label>
                            <select name="end" id="end" class="form-select" required>
                                {% for node in nodes %}
                                <option value="{{ node }}">{{ node }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-md-4">
                            <label for="amount" class="form-label">Amount:</label>
                            <input type="number" name="amount" id="amount" class="form-control" value="10000" min="1" required>
                        </div>
                        <div class="col-md-4">
                            <label for="optimization_mode" class="form-label">Optimization Priority:</label>
                            <select name="optimization_mode" id="optimization_mode" class="form-select">
                                <option value="balanced">Balanced</option>
                                <option value="cost_focused">Cost Optimized</option>
                                <option value="speed_focused">Speed Optimized</option>
                                <option value="security_focused">Security Focused</option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label for="routing_method" class="form-label">Routing Method:</label>
                            <select name="routing_method" id="routing_method" class="form-select">
                                <option value="quantum">Quantum

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, deque
from typing import Tuple, List

def simulate_financial_avalanches(num_nodes: int = 2000, num_steps: int = 20000, m_edges: int = 2) -> Tuple[List[int], nx.Graph]:
    """
    Simulates a financial contagion model (SOC) on a scale-free network.
    """
    # 1. Establish CONNECTIVITY
    G = nx.barabasi_albert_graph(num_nodes, m_edges)
    
    # 2. Initialize System State
    stress = np.zeros(num_nodes)
    threshold = {node: degree for node, degree in G.degree()}
    avalanche_sizes = []

    print("Simulating market dynamics...")
    for step in range(num_steps):
        # 3. NOISE: Exogenous shock
        target = np.random.randint(0, num_nodes)
        stress[target] += 1
        
        # 4. AVALANCHE DYNAMICS
        current_avalanche_size = 0
        
        # Optimized Queue System using deque and a tracking set
        to_process = deque([target])
        in_queue = {target}
        
        while to_process:
            node = to_process.popleft()
            in_queue.remove(node) # Remove from tracking set
            
            # If stress exceeds risk tolerance, the trader panic sells
            if stress[node] >= threshold[node]:
                stress[node] -= threshold[node] 
                current_avalanche_size += 1
                
                # Stress is transferred to connected peers
                for neighbor in G.neighbors(node):
                    stress[neighbor] += 1
                    
                    # Add neighbor to queue if they cross threshold AND aren't already queued
                    if stress[neighbor] >= threshold[neighbor] and neighbor not in in_queue:
                        to_process.append(neighbor)
                        in_queue.add(neighbor)
                        
        if current_avalanche_size > 0:
            avalanche_sizes.append(current_avalanche_size)
            
    return avalanche_sizes, G

def analyze_avalanches(avalanches: List[int]):
    """
    Analyzes and plots the avalanche size distribution using logarithmic binning
    for a more accurate power-law fit.
    """
    if not avalanches:
        print("No avalanches occurred.")
        return

    # Logarithmic binning for cleaner tail distribution
    max_size = max(avalanches)
    min_size = min(avalanches)
    
    if max_size == min_size:
        print("Not enough variance to plot a distribution.")
        return

    # Create bins spaced evenly on a log scale
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=20)
    counts, bin_edges = np.histogram(avalanches, bins=bins)
    
    # Calculate bin centers and probabilities
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Normalize by total avalanches and bin width to get probability density
    probabilities = counts / (sum(counts) * bin_widths)
    
    # Filter out empty bins
    valid = counts > 0
    valid_centers = bin_centers[valid]
    valid_probs = probabilities[valid]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_centers, valid_probs, alpha=0.8, edgecolors='k', color='darkred', label='Log-Binned Data')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Probability Density of Market Avalanches (Log-Log Scale)", fontsize=14)
    plt.xlabel("Avalanche Size $S$", fontsize=12)
    plt.ylabel("Probability Density $P(S)$", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Fit a line to the log-binned data
    if len(valid_centers) > 2:
        log_s = np.log10(valid_centers)
        log_p = np.log10(valid_probs)
        slope, intercept = np.polyfit(log_s, log_p, 1)
        
        trend_s = np.linspace(min(valid_centers), max(valid_centers), 100)
        trend_p = (10**intercept) * (trend_s**slope)
        
        plt.plot(trend_s, trend_p, color='blue', linestyle='--', 
                 label=f'Power-Law Fit: $P(S) \propto S^{{{slope:.2f}}}$')
                 
    plt.legend()
    plt.show()

# Run execution
if __name__ == "__main__":
    avalanches, network = simulate_financial_avalanches(num_nodes=2000, num_steps=50000, m_edges=2)
    analyze_avalanches(avalanches)

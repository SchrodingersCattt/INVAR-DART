import re
import ast
import stat_pareto
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from copy import deepcopy 
import logging
import glob
from time import sleep

def parse_nsga_log(log_file):
    """
    Parse NSGA-II log file to extract optimization progress
    Returns:
        generations: List of generation numbers
        front_sizes: List of Pareto front sizes for each generation
        avg_objectives: List of average objective values for each generation
        best_individuals: List of best individuals for each generation
        elements: List of elements being optimized
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    generations = []
    front_sizes = []
    avg_objectives = []
    best_individuals_list = []
    best_individuals = {}
    elements = []
    pred_tec_means = []
    pred_density_means = []
    pred_density_stds = []
    pred_tec_stds = []
    targets = []

    # Patterns for parsing
    gen_pattern = re.compile(r"Generation (\d+)")
    front_pattern = re.compile(r"First front average objectives: \[([^\]]+)\]")
    elements_pattern = re.compile(r"Elements: (\[.+\])")
    phys_pattern = re.compile(
        r"Physical values - TEC mean: ([\d\.]+), TEC std: ([\d\.]+), "
        r"Density mean: ([\d\.]+), Density std: ([\d\.]+)"
    )
    best_ind_pattern = re.compile(r"Best Individual: \[([^\]]+)\], Best Objectives: \[([^\]]+)\]")

    for line in lines:
        # Get elements list
        if "Elements: " in line and not elements:
            elements_match = elements_pattern.search(line)
            if elements_match:
                elements = ast.literal_eval(elements_match.group(1))
        
        # Get generation number
        gen_match = gen_pattern.search(line)
        if gen_match:
            generation = int(gen_match.group(1))
            if generation not in generations:
                generations.append(generation)
                # Store the best individual when a new generation starts
                if best_individuals:
                    best_individuals_list.append(deepcopy(best_individuals))
        
        # Get Pareto front average objectives
        front_match = front_pattern.search(line)
        if front_match:
            objectives = list(map(float, front_match.group(1).split()))
            avg_objectives.append(objectives)
            front_sizes.append(1)  # Default to 1
        
        # Get physical values - only store if we have a matching generation
        phys_match = phys_pattern.search(line)
        if phys_match and generations:
            tec_mean = float(phys_match.group(1))
            tec_std = float(phys_match.group(2))
            density_mean = float(phys_match.group(3))
            density_std = float(phys_match.group(4))
            
            pred_tec_means.append(tec_mean)
            pred_tec_stds.append(tec_std)
            pred_density_means.append(density_mean)
            pred_density_stds.append(density_std)
        
        # Get best individual at the end
        best_match = best_ind_pattern.search(line)
        if best_match and generations:
            ind = list(map(float, best_match.group(1).split()))
            objs = list(map(float, best_match.group(2).split()))
            best_individuals = dict(zip(elements, ind))

    # Append the last best individual after the loop ends
    if best_individuals:
        best_individuals_list.append(deepcopy(best_individuals))

    # Make sure all arrays have the same length by truncating to minimum length
    min_len = min(len(generations), len(pred_tec_means), len(pred_density_means), 
                  len(pred_tec_stds), len(pred_density_stds), len(best_individuals_list)) if generations else 0
    
    if min_len > 0:
        generations = generations[:min_len]
        pred_tec_means = pred_tec_means[:min_len]
        pred_density_means = pred_density_means[:min_len]
        pred_tec_stds = pred_tec_stds[:min_len]
        pred_density_stds = pred_density_stds[:min_len]
        best_individuals_list = best_individuals_list[:min_len]

    return {
        'generations': generations,
        'front_sizes': front_sizes,
        'avg_objectives': avg_objectives,
        'elements': elements,
        'pred_tec_means': pred_tec_means,
        'pred_density_means': pred_density_means,
        'pred_tec_stds': pred_tec_stds,
        'pred_density_stds': pred_density_stds,
        'best_individuals_list': best_individuals_list
    }

def plot_nsga_results(data, log_name):
    """
    Plot NSGA-II optimization results with the same style as GA monitor
    """
    if not data['generations'] or len(data['generations']) == 0:
        logging.warning(f"No valid data to plot for {log_name}")
        return

    generations = data['generations']
    tec = data['pred_tec_means']
    density = data['pred_density_means']
    pred_tec_stds = data['pred_tec_stds']
    pred_density_stds = data['pred_density_stds']
    
    # Ensure all arrays have the same length
    min_len = min(len(generations), len(tec), len(density), len(pred_tec_stds), len(pred_density_stds))
    
    if min_len == 0:
        logging.warning(f"Not enough data points to plot for {log_name}")
        return
    
    # Truncate all arrays to the same length
    generations = generations[:min_len]
    tec = tec[:min_len]
    density = density[:min_len]
    pred_tec_stds = pred_tec_stds[:min_len]
    pred_density_stds = pred_density_stds[:min_len]
    
    # Convert inputs to numpy arrays for convenience
    g = np.array(generations)
    tec = np.array(tec)
    density = np.array(density)
    pred_tec_stds = np.array(pred_tec_stds)
    pred_density_stds = np.array(pred_density_stds)
    
    plt.figure(figsize=(19, 4))
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Arial'
    plt.suptitle(log_name)

    # Subplot 1: TEC over generations
    plt.subplot(141)
    plt.plot(g, tec)
    plt.fill_between(g, tec - pred_tec_stds, tec + pred_tec_stds, alpha=0.2)
    plt.ylabel("TEC")
    
    # Inset for TEC standard deviation
    ax_inset_tec = inset_axes(plt.gca(), width="30%", height="30%", loc="lower right")
    ax_inset_tec.plot(g, pred_tec_stds, color='orange', label="TEC STD")
    ax_inset_tec.set_ylabel("STD")
    
    # Subplot 2: Density over generations
    plt.subplot(142)
    plt.plot(g, density)
    plt.fill_between(g, density - pred_density_stds, density + pred_density_stds, alpha=0.2)
    plt.ylabel("Density")
    plt.xlabel("Generation")
    
    # Inset for Density standard deviation
    ax_inset_density = inset_axes(plt.gca(), width="30%", height="30%", loc="lower right")
    ax_inset_density.plot(g, pred_density_stds, color='orange')
    ax_inset_density.set_ylabel("STD")
    
    # Subplot 3: Placeholder for target (NSGA doesn't use target)
    plt.subplot(143)
    plt.plot(g, np.zeros_like(g))
    plt.ylabel("Target (N/A for NSGA)")

    # Subplot 4: TEC vs Density scatter plot
    plt.subplot(144)
    plt.xlabel("TEC")
    plt.ylabel("Density")
    plt.grid()
    
    try:
        data_file = "dataset/Data_base_DFT_Thermal.xlsx"
        raw_tec, raw_den, pareto_front, _ = stat_pareto.get_pareto_front(data_file)
        
        # Plot the original Pareto front
        plt.scatter(raw_tec, raw_den, c='#9999AA', label="Orig. Data", alpha=0.5)
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='#882233', label="Orig. Pareto Front.", marker='s')
    except Exception as e:
        logging.warning(f"Could not load original Pareto front: {e}")
    
    # Use the new method to get the colormap
    cmap = plt.get_cmap('viridis')
    
    # Create a ScalarMappable object with a dummy array for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=g.min(), vmax=g.max()))
    sm.set_array([])  # Provide an empty array for the ScalarMappable object
    
    # Get the current axis
    ax = plt.gca()
    cbar = ax.figure.colorbar(sm, ax=ax, label="Generation")
    sc = ax.scatter(tec, density, c=g, linestyle='-', label="New Points", marker='x', s=2, cmap=cmap)
    ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout(pad=0.2)
    plt.savefig(f"{log_name}.png", dpi=300)
    plt.close()

def regularize_precision(x):    
    if isinstance(x, dict):
        return {k: round(v, 4) for k, v in x.items()}
    elif isinstance(x, list):
        if not x:
            return x
        if isinstance(x[0], float):
            return [round(y, 4) for y in x]
        elif isinstance(x[0], dict):
            return [regularize_precision(y) for y in x]
        else:
            return x
    elif isinstance(x, (int, float)):
        return round(x, 4)
    else:
        return x

def save_results_to_csv(data, log_name):
    """Save parsed data to CSV files for further analysis with GA-style format"""
    if not data['generations']:
        logging.warning(f"No data to save for {log_name}")
        return
    
    generations = data['generations']
    pred_tec_means = data['pred_tec_means']
    pred_density_means = data['pred_density_means']
    pred_tec_stds = data['pred_tec_stds']
    pred_density_stds = data['pred_density_stds']
    best_individuals_list = data['best_individuals_list']
    elements = data['elements']
    
    # Ensure all arrays have the same length
    min_len = min(len(generations), len(pred_tec_means), len(pred_density_means), 
                  len(pred_tec_stds), len(pred_density_stds), len(best_individuals_list))
    
    if min_len == 0:
        logging.warning(f"Not enough data points to save for {log_name}")
        return
    
    # Truncate all arrays to the same length
    generations = generations[:min_len]
    pred_tec_means = pred_tec_means[:min_len]
    pred_density_means = pred_density_means[:min_len]
    pred_tec_stds = pred_tec_stds[:min_len]
    pred_density_stds = pred_density_stds[:min_len]
    best_individuals_list = best_individuals_list[:min_len]
    
    # Regularize precision
    generations = regularize_precision(generations)
    pred_tec_means = regularize_precision(pred_tec_means)
    pred_density_means = regularize_precision(pred_density_means)
    pred_tec_stds = regularize_precision(pred_tec_stds)
    pred_density_stds = regularize_precision(pred_density_stds)
    best_individuals_list = regularize_precision(best_individuals_list)
    
    new_paretos = {}
    
    # Apply the same mask as in monitor_ga.py
    tec_array = np.array(pred_tec_means)
    density_array = np.array(pred_density_means)
    mask = np.logical_and(tec_array < 5, density_array < 8100)
    
    for idx, g in enumerate(generations):
        if not g in np.array(generations)[mask]:
            continue
        new_paretos[g] = {
            'Pred Tec Means': pred_tec_means[idx], 
            'Pred Density Means': pred_density_means[idx], 
            'Best Individuals': best_individuals_list[idx] if idx < len(best_individuals_list) else {}
        }
        
    df = pd.DataFrame(new_paretos).T
    
    # Check if 'Pred Tec Means' column exists and handle appropriately
    if not df.empty and 'Pred Tec Means' in df.columns:
        df_sorted = df.sort_values(by='Pred Tec Means', ascending=True)
        df_sorted.to_csv(f"{log_name}.csv")
        logging.info(f"{log_name}:\n {df_sorted.to_markdown()}")
    else:
        logging.warning(f"{log_name}: No 'Pred Tec Means' column found or DataFrame is empty")
        df.to_csv(f"{log_name}.csv")
        logging.info(f"{log_name}:\n {df.to_markdown()}")

def monitor_nsga_logs(log_pattern="nsga_*.log"):
    """Monitor NSGA-II log files and generate reports"""
    logs = glob.glob(log_pattern)
    
    for log_file in logs:
        try:
            logging.info(f"Processing log file: {log_file}")
            log_name = log_file.split('.log')[0]
            
            # Parse the log file
            data = parse_nsga_log(log_file)
            
            if not data['generations']:
                logging.warning(f"No generations found in {log_file}")
                continue
                
            # Generate plots
            plot_nsga_results(data, log_name)
            
            # Save data to CSV
            save_results_to_csv(data, log_name)
            
            logging.info(f"Successfully processed {log_file}")
            
        except Exception as e:
            logging.error(f"Error processing {log_file}: {str(e)}")
            continue

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Monitor logs continuously (every 60 seconds)
    while True:
        monitor_nsga_logs()
        sleep(60)
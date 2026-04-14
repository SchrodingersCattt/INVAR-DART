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


TEC_THRESHOLD = 5
DENSITY_THRESHOLD = 8200


def parse_nsga_log(log_file):
    """
    Parse NSGA-II log file to extract generation data including all individuals
    """
    data = {
        'generations': [],
        'best_individuals_list': [],
        'all_individuals': [],  # Store all individuals for each generation
        'all_objectives': [],   # Store all objectives for each generation
        'elements': []
    }
    
    # Generation tracking
    current_generation = -1
    current_gen_individuals = []
    current_gen_objectives = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    # Get elements from the log
    for line in lines:
        if "Elements:" in line:
            # Extract elements list
            elements_match = re.search(r"Elements: \[(.*?)\]", line)
            if elements_match:
                elements_str = elements_match.group(1)
                data['elements'] = [e.strip().strip("'") for e in elements_str.split(",")]
            break
    
    # Parse the log file
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for generation info
        gen_match = re.search(r"Generation (\d+)", line)
        if gen_match:
            new_generation = int(gen_match.group(1))
            if new_generation != current_generation:
                # Save previous generation data if exists
                if current_generation >= 0 and current_gen_objectives:
                    data['generations'].append(current_generation)
                    data['all_individuals'].append(current_gen_individuals.copy())
                    data['all_objectives'].append(current_gen_objectives.copy())
                    # For now, use first individual as "best" - in reality, NSGA-II has a front of best solutions
                    data['best_individuals_list'].append(current_gen_individuals[0] if current_gen_individuals else {})
                
                # Reset for new generation
                current_generation = new_generation
                current_gen_individuals = []
                current_gen_objectives = []
        
        # Check for physical values
        phys_match = re.search(r"Physical values - TEC mean: ([\d\.eE+-]+), TEC std: ([\d\.eE+-]+), Density mean: ([\d\.eE+-]+), Density std: ([\d\.eE+-]+)", line)
        if phys_match and current_generation >= 0:
            tec_mean = float(phys_match.group(1))
            tec_std = float(phys_match.group(2))
            density_mean = float(phys_match.group(3))
            density_std = float(phys_match.group(4))
            
            # Store the individual's objectives
            current_gen_objectives.append([tec_mean, tec_std, density_mean, density_std])
            
            # Try to find the composition that led to these physical values
            # Look backwards for the composition
            individual = {}
            for j in range(min(10, i)):  # Check up to 10 lines back
                prev_line = lines[i-j-1]
                comp_match = re.search(r"Evaluating objectives for composition: \[(.*?)\]", prev_line)
                if comp_match:
                    comp_str = comp_match.group(1)
                    comp_values = [float(x.strip()) for x in comp_str.split()]
                    individual = {data['elements'][k]: comp_values[k] for k in range(len(comp_values)) if k < len(data['elements'])}
                    break
            
            current_gen_individuals.append(individual)
        
        # Check for first front average objectives (for best individuals tracking)
        first_front_match = re.search(r"First front average objectives: \[(.*?)\]", line)
        if first_front_match and current_generation >= 0:
            # This line contains the average of the first Pareto front
            # We could use this information to better select "best" individuals if needed
            pass
            
        i += 1
    
    # Don't forget the last generation
    if current_generation >= 0 and current_gen_objectives:
        data['generations'].append(current_generation)
        data['all_individuals'].append(current_gen_individuals.copy())
        data['all_objectives'].append(current_gen_objectives.copy())
        data['best_individuals_list'].append(current_gen_individuals[0] if current_gen_individuals else {})
    
    return data


def plot_nsga_results(data, log_name):
    """
    Plot NSGA-II optimization results using only individual data points
    """
    if not data['generations'] or len(data['generations']) == 0:
        logging.warning(f"No valid data to plot for {log_name}")
        return

    generations = data['generations']
    all_objectives = data['all_objectives']
    
    # Ensure we have data to plot
    if len(all_objectives) == 0:
        logging.warning(f"Not enough data points to plot for {log_name}")
        return
    
    plt.figure(figsize=(15, 4))
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Arial'
    plt.suptitle(log_name)

    # Subplot 1: TEC over generations (using all individuals)
    plt.subplot(131)
    # Collect all TEC values and their corresponding generations
    all_tec_values = []
    all_gen_values = []
    
    for gen_idx, gen_objectives in enumerate(all_objectives):
        for objectives in gen_objectives:
            if len(objectives) >= 4:
                all_tec_values.append(objectives[0])  # TEC mean
                all_gen_values.append(generations[gen_idx] if gen_idx < len(generations) else gen_idx)
    
    # Plot all individual points
    plt.scatter(all_gen_values, all_tec_values, alpha=0.6, s=1)
    plt.ylabel("TEC")
    
    # Inset for TEC standard deviation from log data
    ax_inset_tec = inset_axes(plt.gca(), width="30%", height="30%", loc="lower right")
    
    # Collect TEC std values from log (second value in objectives)
    tec_std_values = []
    gen_numbers_for_std = []
    
    for gen_idx, gen_objectives in enumerate(all_objectives):
        for objectives in gen_objectives:
            if len(objectives) >= 4:
                tec_std_values.append(abs(objectives[1]))  # TEC std (already negative in log)
                gen_numbers_for_std.append(generations[gen_idx] if gen_idx < len(generations) else gen_idx)
    
    # Plot TEC std values
    if tec_std_values:
        ax_inset_tec.scatter(gen_numbers_for_std, tec_std_values, s=1, alpha=0.6)
        ax_inset_tec.set_ylabel("TEC STD", fontsize=8)
        ax_inset_tec.tick_params(axis='both', which='major', labelsize=6)
    else:
        ax_inset_tec.text(0.5, 0.5, "N/A", transform=ax_inset_tec.transAxes, 
                          horizontalalignment='center', verticalalignment='center')
        ax_inset_tec.set_ylabel("TEC STD", fontsize=8)
    
    # Subplot 2: Density over generations (using all individuals)
    plt.subplot(132)
    # Collect all Density values and their corresponding generations
    all_density_values = []
    all_gen_values = []
    
    for gen_idx, gen_objectives in enumerate(all_objectives):
        for objectives in gen_objectives:
            if len(objectives) >= 4:
                all_density_values.append(objectives[2])  # Density mean
                all_gen_values.append(generations[gen_idx] if gen_idx < len(generations) else gen_idx)
    
    # Plot all individual points
    plt.scatter(all_gen_values, all_density_values, alpha=0.6, s=1)
    plt.ylabel("Density")
    plt.xlabel("Generation")

    # Inset for Density standard deviation from log data
    ax_inset_density = inset_axes(plt.gca(), width="30%", height="30%", loc="lower right")
    
    # Collect Density std values from log (fourth value in objectives)
    density_std_values = []
    gen_numbers_for_std = []
    
    for gen_idx, gen_objectives in enumerate(all_objectives):
        for objectives in gen_objectives:
            if len(objectives) >= 4:
                density_std_values.append(abs(objectives[3]))  # Density std (already negative in log)
                gen_numbers_for_std.append(generations[gen_idx] if gen_idx < len(generations) else gen_idx)
    
    # Plot Density std values
    if density_std_values:
        ax_inset_density.scatter(gen_numbers_for_std, density_std_values, s=1, alpha=0.6)
        ax_inset_density.set_ylabel("Density STD", fontsize=8)
        ax_inset_density.tick_params(axis='both', which='major', labelsize=6)
    else:
        ax_inset_density.text(0.5, 0.5, "N/A", transform=ax_inset_density.transAxes, 
                              horizontalalignment='center', verticalalignment='center')
        ax_inset_density.set_ylabel("Density STD", fontsize=8)

    # Subplot 3: TEC vs Density scatter plot
    plt.subplot(133)
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
    
    # Plot ALL individuals from all generations
    cmap = plt.get_cmap('viridis')
    
    # Collect all TEC and Density values from all individuals
    all_tec_values = []
    all_density_values = []
    all_generations = []
    tec_errors = []  # For error bars
    colors_for_errorbars = []  # To store colors for error bars
    
    for gen_idx, gen_objectives in enumerate(all_objectives):
        for objectives in gen_objectives:
            # Assuming TEC is the first objective and Density is the third (based on NSGA-II objectives)
            if len(objectives) >= 4:
                all_tec_values.append(objectives[0])  # TEC mean
                all_density_values.append(objectives[2])  # Density mean
                all_generations.append(generations[gen_idx] if gen_idx < len(generations) else gen_idx)
                tec_errors.append(abs(objectives[1]))  # TEC std for error bars
                
    # Convert to numpy arrays
    all_tec_values = np.array(all_tec_values)
    all_density_values = np.array(all_density_values)
    all_generations = np.array(all_generations)
    tec_errors = np.array(tec_errors)
    
    if len(all_generations) > 0:
        # Normalize generation values for color mapping
        norm = plt.Normalize(vmin=np.min(all_generations), vmax=np.max(all_generations))
        cmap = plt.get_cmap('viridis')
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax = plt.gca()
        cbar = ax.figure.colorbar(sm, ax=ax, label="Generation")
        
        # Scatter plot with colors based on generation
        sc = ax.scatter(all_tec_values, all_density_values, 
                        c=all_generations, linestyle='-', label="New Points", marker='x', s=2, cmap=cmap)
        
        # Add error bars with matching colors
        colors = cmap(norm(all_generations))
        for i in range(len(all_tec_values)):
            ax.errorbar(all_tec_values[i], all_density_values[i], xerr=tec_errors[i], 
                        fmt='none', ecolor=colors[i], alpha=0.3, elinewidth=0.5)
        
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
    best_individuals_list = data['best_individuals_list']
    all_individuals = data['all_individuals']
    all_objectives = data['all_objectives']
    elements = data['elements']
    
    # Ensure all arrays have the same length
    min_len = min(len(generations), len(best_individuals_list),
                  len(all_individuals), len(all_objectives))
    
    if min_len == 0:
        logging.warning(f"Not enough data points to save for {log_name}")
        return
    
    # Truncate all arrays to the same length
    generations = generations[:min_len]
    best_individuals_list = best_individuals_list[:min_len]
    all_individuals = all_individuals[:min_len]
    all_objectives = all_objectives[:min_len]
    
    # Regularize precision
    generations = regularize_precision(generations)
    best_individuals_list = regularize_precision(best_individuals_list)
    
    # Collect all individuals with their TEC and Density values
    all_data = []
    
    # Create a list to store the filtered data with element compositions
    filtered_data = []
    
    # Add all individuals in this generation
    for gen_idx, g in enumerate(generations):
        if gen_idx < len(all_objectives):
            for ind_idx, objectives in enumerate(all_objectives[gen_idx]):
                # Check if this individual meets the criteria
                if len(objectives) >= 4:
                    tec_val = objectives[0]  # TEC mean
                    tec_std_val = abs(objectives[1])  # TEC std
                    density_val = objectives[2]  # Density mean
                    
                    # Check if TEC < 5 or TEC - TEC_STD < 5
                    meets_criteria = tec_val < TEC_THRESHOLD or (tec_val - tec_std_val) < TEC_THRESHOLD
                    
                    if meets_criteria and (density_val < DENSITY_THRESHOLD):
                        ind_data = {
                            'Generation': g,
                            'Type': 'Individual',
                            'TEC': tec_val,
                            'Density': density_val,
                            'TEC_STD': tec_std_val,
                            'Density_STD': None,
                            'Individual': all_individuals[gen_idx][ind_idx] if gen_idx < len(all_individuals) and ind_idx < len(all_individuals[gen_idx]) else {}
                        }
                        all_data.append(ind_data)
                        
                        # Add to filtered data with element compositions
                        individual_composition = all_individuals[gen_idx][ind_idx] if gen_idx < len(all_individuals) and ind_idx < len(all_individuals[gen_idx]) else {}
                        if individual_composition:  # Only add if we have composition data
                            filtered_row = {
                                'Generation': g,  # Include generation number
                                'TEC': tec_val,
                                'TEC_STD': tec_std_val,
                                'TEC-STD': tec_val - tec_std_val,  # TEC - TEC_STD value
                                'Density': density_val
                            }
                            # Add element compositions
                            for element in elements:
                                filtered_row[element] = individual_composition.get(element, 0.0)
                            filtered_data.append(filtered_row)
    
    # Create DataFrame with all data
    df_all = pd.DataFrame(all_data)
    
    # Save all data to CSV
    df_all.to_csv(f"{log_name}_all_individuals.csv", index=False)
    # Create and save the filtered data with element compositions
    if filtered_data:
        df_filtered = pd.DataFrame(filtered_data)
        print(df_filtered.to_markdown())
        df_filtered.to_csv(f"{log_name}_filtered_with_compositions.csv", index=False)
        logging.info(f"Filtered data with compositions saved to {log_name}_filtered_with_compositions.csv")

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
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

def parse_log(log_file, mode):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    generations = []
    pred_tec_means = []
    pred_density_means = []
    pred_density_stds = []
    pred_tec_stds = []
    targets = []
    best_individuals_list = []
    best_individuals = {}
    
    # Parameters for STD_FACTOR calculation
    a_param = None
    b_param = None
    c_param = None

    # Define regex patterns
    gen_pattern = re.compile(r"- Generation (\d+)")
    tec_pattern = re.compile(r"pred_tec_mean:\s*([-+]?\d*\.\d+|\d+)")
    tec_std_pattern = re.compile(r"tec_std:\s*([-+]?\d*\.\d+|\d+)")
    density_pattern = re.compile(r"pred_density_mean:\s*([-+]?\d*\.\d+|\d+)")
    density_std_pattern = re.compile(r"density_std:\s*([-+]?\d*\.\d+|\d+)")
    target_pattern = re.compile(r"target:\s*([-+]?\d*\.\d+|\d+)")
    # Parameter patterns - fixed to match the actual log format
    param_pattern = re.compile(r"a:\s*([-+]?\d*\.\d+|\d+),\s*b:\s*([-+]?\d*\.\d+|\d+),\s*c:\s*([-+]?\d*\.\d+|\d+)")

    read_blocks = False
    best_individual_str = ""
    current_generation = None

    for line in lines:
        if "Elements: " in line:
            element_list = ast.literal_eval(line.split("Elements: ")[-1])
        # Parse a, b, c parameters
        param_match = param_pattern.search(line)
        if param_match:
            a_param = float(param_match.group(1))
            b_param = float(param_match.group(2))
            c_param = float(param_match.group(3))
        if "====" in line:
            read_blocks = True
            continue
        if "----" in line:
            read_blocks = False
            continue
        if read_blocks:
            # Match Generation
            gen_match = gen_pattern.search(line)
            if gen_match:
                generation = int(gen_match.group(1))
                if generation not in generations:
                    generations.append(generation)
                    # Only store the best individual when a new generation starts
                    if best_individuals:
                        best_individuals_list.append(deepcopy(best_individuals))
                current_generation = generation

            # Match pred_tec_mean
            tec_match = tec_pattern.search(line)
            if tec_match:
                pred_tec_means.append(float(tec_match.group(1)))            
            # Match pred_density_mean
            density_match = density_pattern.search(line)
            if density_match:
                pred_density_means.append(float(density_match.group(1)))            
            # Match tec_std            
            tec_std_match = tec_std_pattern.search(line)
            if tec_std_match:
                pred_tec_stds.append(float(tec_std_match.group(1)))            
            # Match density_std
            density_std_match = density_std_pattern.search(line)
            if density_std_match:
                pred_density_stds.append(float(density_std_match.group(1)))            
            # Match target
            target_match = target_pattern.search(line)
            if target_match:
                targets.append(float(target_match.group(1)))

        if mode == "ga":
            if " - Best Individual: [" in line and not "]" in line:
                best_individual_str = line.split("Best Individual: [")[-1]  
            elif "]" in line and best_individual_str:
                best_individual_str += line.split("]")[0] 
                best_individual = list(map(float, best_individual_str.split()))
                for idx, e in enumerate(element_list):
                    best_individuals[e] = best_individual[idx]
                best_individual_str = ""
            if " - Best Individual: [" in line and '[' in line and ']' in line: 
                s = line.split("Best Individual: [")[-1].split("]")[0]
                best_individual = list(map(float, s.split()))
                for idx, e in enumerate(element_list):
                    best_individuals[e] = best_individual[idx]
            
        if mode == 'bo':
            if " - Normalized Compositions: [" in line and not "]" in line:
                best_individual_str = line.split(" - Normalized Compositions: [")[-1]        
            elif "]" in line and best_individual_str:
                best_individual_str += line.split("]")[0] 
                best_individual = list(map(float, best_individual_str.split()))
                for idx, e in enumerate(element_list):
                    best_individuals[e] = best_individual[idx]            
                best_individual_str = "" 
            elif " - Normalized Compositions: [" in line and '[' in line and ']' in line:            
                s = line.split(" - Normalized Compositions: [")[-1].split("]")[0]
                best_individual = list(map(float, s.split(','))) if ',' in s else list(map(float, s.split()))
                for idx, e in enumerate(element_list):
                    best_individuals[e] = best_individual[idx]

    # Append the last best individual after the loop ends
    if best_individuals:
        best_individuals_list.append(deepcopy(best_individuals))

    return generations, pred_tec_means, pred_density_means, targets, pred_tec_stds, pred_density_stds, best_individuals_list, a_param, b_param, c_param



def plot_ga(g, tec, density, target, pred_tec_stds, pred_density_stds, log_name, std_factor=None):
    # Convert inputs to numpy arrays for convenience
    g = np.array(g)
    tec = np.array(tec)
    density = np.array(density)
    target = np.array(target)
    pred_tec_stds = np.array(pred_tec_stds)
    pred_density_stds = np.array(pred_density_stds)
    
    # Apply STD_FACTOR to tec standard deviation if provided
    if std_factor is not None:
        pred_tec_stds = pred_tec_stds * std_factor
    print(f"STD_FACTOR: {std_factor}")
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
    
    # Subplot 3: Target over generations
    plt.subplot(143)
    plt.plot(g, target)
    plt.ylabel("Target")


    plt.subplot(144)
    plt.xlabel("TEC")
    plt.ylabel("Density")
    # plt.xlim(-2, 5)
    plt.ylim(7500, 8500)
    plt.text(0.05, 0.05, f"Error bars represent {std_factor} times raw std." if std_factor is not None else "Error bars represent raw std.", transform=plt.gca().transAxes)
    plt.grid()
    data_file = "dataset/Data_base_DFT_Thermal.xlsx"
    raw_tec, raw_den, pareto_front, _, _ = stat_pareto.get_pareto_front(data_file)
    
    # Plot the original Pareto front
    plt.scatter(raw_tec, raw_den, s=5, c='#9999AA', label="Orig. Data", alpha=0.5)
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], s=5, c='#882233', label="Orig. Pareto Front.", marker='s')
    
    # Use the new method to get the colormap
    cmap = plt.get_cmap('viridis')
    
    # Create a ScalarMappable object with a dummy array for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=g.min(), vmax=g.max()))
    sm.set_array([])  # Provide an empty array for the ScalarMappable object
    
    # Get the current axis
    ax = plt.gca()
    cbar = ax.figure.colorbar(sm, ax=ax, label="Generation")
    sc = ax.scatter(tec, density, c=g, linestyle='-', label="New Points", marker='x', s=2, cmap=cmap)
    # Add horizontal error bars for TEC standard deviation
    ax.errorbar(tec, density, xerr=pred_tec_stds, linestyle='None', color='black', alpha=0.5, capsize=2, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)

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


def main():
    # Parse the log file
    import glob
    import stat_pareto
    import logging
    import traceback
    from time import sleep
    logging.basicConfig(level=logging.INFO)

    logs = glob.glob("2025*.log")
    
    for log in logs:
        if '_bo' in log:
            mode = "bo"
        elif '_ga' in log:
            mode = "ga"
        else:
            print(Exception(f"Unknown mode {log}"))
            #continue
        print(f"=========Parsing {log}")
        # Updated to get a, b, c parameters
        generations, pred_tec_means, pred_density_means, targets, pred_tec_stds, pred_density_stds, best_individuals, a_param, b_param, c_param = parse_log(log, mode)
        
        # Calculate STD_FACTOR as b/a if both parameters are available
        std_factor = None
        if a_param is not None and b_param is not None and a_param != 0:
            std_factor = b_param / a_param

        log_name = log.split(".log")[0]
        new_paretos = {}
        # Print results
        '''print(f"{log} Generations:", generations)
        print(f"{log} Pred Tec Means:", pred_tec_means)
        print(f"{log} Pred Density Means:", pred_density_means)
        print(f"{log} Best Individuals:", best_individuals)
        print(f"{log} Targets:", targets)
        print(f"{log} a parameter:", a_param)
        print(f"{log} b parameter:", b_param)
        print(f"{log} c parameter:", c_param)
        print(f"{log} STD_FACTOR (b/a):", std_factor)'''

        try:
            ## Plot the results
            plt.figure(figsize=(19,4))
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'Arial'
            # Pass std_factor to plot function
            plot_ga(generations, pred_tec_means, pred_density_means, targets, pred_tec_stds, pred_density_stds, log_name, std_factor)
            plt.tight_layout(pad=0.2)
            plt.savefig(f"{log_name}.png", dpi=300)

            # Apply additional constraint: Al > 0 or Ti > 0
            element_mask = np.array([(ind.get('Al', 0) > 8e-3 or ind.get('Ti', 0) > 8e-3) for ind in best_individuals])


            ## HACK
            if std_factor >=10:
                continue
            # If no individuals meet primary criteria, use secondary criteria
            property_mask_1 = np.logical_and(np.array(pred_tec_means) < 5, 
                                    np.array(pred_density_means) < 8200)
            property_mask_2 = np.logical_and(np.array(pred_tec_means) < 12, 
                                    np.array(pred_density_means) < 8000)
            property_mask = np.logical_or(property_mask_1, property_mask_2)     
            # Apply additional constraint: Al > 0 or Ti > 0
            mask = np.logical_and(property_mask, element_mask)
            
            best_individuals = regularize_precision(best_individuals)
            pred_tec_means  = regularize_precision(pred_tec_means)
            pred_density_means  = regularize_precision(pred_density_means)
            pred_tec_stds = regularize_precision(pred_tec_stds)
            targets = regularize_precision(targets)
            for idx, g in enumerate(generations):
                if not g in np.array(generations)[mask]:
                    continue
                new_paretos[g] = {
                    'Pred Tec Means': pred_tec_means[idx], 
                    'Pred Tec Stds': pred_tec_stds[idx],
                    'Pred Density Means': pred_density_means[idx], 
                    'Targets': targets[idx],
                    'Best Individuals': best_individuals[idx]
                }
                
            df = pd.DataFrame(new_paretos).T
            # Check if 'Pred Tec Stds' column exists and handle appropriately
            # if not df.empty and 'Pred Tec Stds' in df.columns:
            #     df_sorted = df.sort_values(by='Pred Tec Stds', ascending=False)
            #     df_sorted.to_csv(f"{log_name}.csv")
            #     logging.info(f"{log}:\n {df_sorted.to_markdown()}")
            # else:
            #     logging.warning(f"{log}: No 'Pred Tec Stds' column found or DataFrame is empty")
            #     df.to_csv(f"{log_name}.csv")
            #     logging.info(f"{log}:\n {df.to_markdown()}")

            # print()

            ## Sort by target values
            if not df.empty:
                df_sorted = df.sort_values(by="Targets", ascending=False)
                
                df_sorted.to_csv(f"{log_name}_target.csv")
                logging.info(f"{log} (Target sorted):\n {df_sorted.to_markdown()}")
            # ## UCB
            # if not df.empty:
            #     kappa = 10  # 探索权重，可以调
            #     # 定义一个新的 UCB 列
            #     df["UCB_Score"] = df["Pred Tec Means"] - kappa * df["Pred Tec Stds"]
                
            #     # 由于是最小化 TEC，这里按 UCB 升序（越低越好）
            #     df_sorted = df.sort_values(by="UCB_Score", ascending=True)
                
            #     df_sorted.to_csv(f"{log_name}_ucb.csv")
            #     logging.info(f"{log} (UCB sorted):\n {df_sorted.to_markdown()}")
        except Exception as e:
            logging.error(f"Error plotting {log}: {e, traceback.print_exc()}")
            print(f"Error plotting {log}: {e}")


if __name__ == "__main__":
    from time import sleep
    while True:
        main()
        sleep(10)
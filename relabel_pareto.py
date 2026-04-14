from stat_distribution import ExcelParser
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from deepmd.pt.infer.deep_eval import DeepProperty
from target import comp2struc, pred, get_packing

# 加载原子质量和转换函数
import json
from constraints_utils import mass_to_molar as mass_to_molar_converter

with open("constant/atomic_mass.json", 'r') as f:
    atomic_mass = json.load(f)

STD_FACTOR = 2

def is_dominated(target, others):
    for other in others:
        if all(o <= t for o, t in zip(other, target)) and any(o < t for o, t in zip(other, target)):
            return True
    return False

def extract_data(parser):
    tec = []
    density = []
    packing = []
    Fe, Ni, Co, Cr, V, Cu = [], [], [], [], [], []

    for row in tqdm(parser.iterate_rows()):
        tec.append(parser.get_tec(row))
        density.append(parser.get_density(row))
        Fe.append(parser.get_Fe(row))
        Ni.append(parser.get_Ni(row))
        Co.append(parser.get_Co(row))
        Cr.append(parser.get_Cr(row))
        V.append(parser.get_V(row))
        Cu.append(parser.get_Cu(row))
        packing.append((parser.get_phase(row)).replace("'", ""))

    return np.array(tec), np.array(density), np.array(Fe), np.array(Ni), np.array(Co), np.array(Cr), np.array(V), np.array(Cu), packing

def predict_tec_with_uncertainty(composition_dict, fraction_type='mass'):
    """
    Predict TEC for given composition with uncertainty using models
    
    Parameters:
    composition_dict (dict): Dictionary with element symbols as keys and proportions as values
    fraction_type (str): Either 'mass' or 'molar' to indicate the type of proportions
    
    Returns:
    tuple: (mean_tec, std_tec, all_predictions) - Mean TEC, standard deviation, and all predictions
    """
    
    # Extract elements and proportions from dictionary
    elements = list(composition_dict.keys())
    proportions = list(composition_dict.values())
    
    # Convert to molar fractions if needed
    if fraction_type == 'mass':
        # Convert mass fractions to molar fractions
        mass_array = np.array(proportions)
        molar_proportions = mass_to_molar_converter(mass_array, elements)
        proportions = molar_proportions.tolist()
    elif fraction_type != 'molar':
        raise ValueError("fraction_type must be either 'mass' or 'molar'")
    
    # Get packing structure
    packing = get_packing(elements, proportions)
    
    # Generate structure
    struct_list = comp2struc(elements, proportions, packing=packing)
    
    # Load all TEC models
    print(glob.glob('models/tec*.pt'), glob.glob('models/tec*.pt')[0])
    tec_model_paths = glob.glob('models/tec*.pt')[0:]
    if not tec_model_paths:
        raise FileNotFoundError("No TEC model found in models/tec*.pt")
    
    tec_models = [DeepProperty(model_path) for model_path in tec_model_paths]
    
    # Predict TEC for all structures and models
    tec_predictions = []
    for model in tqdm(tec_models, desc="models"):
        for struct in tqdm(struct_list, desc="structs"):
            pred_tec = pred(model, struct)
            tec_predictions.append(pred_tec)
    
    # Return mean, standard deviation, and all predictions
    mean_tec = np.mean(tec_predictions)
    std_tec = np.std(tec_predictions)
    return mean_tec, std_tec, tec_predictions

def get_pareto_front(data_file, output=False, relabel=False):
    parser = ExcelParser(data_file)
    
    if relabel:
        # Relabeled version - predict TEC values using models
        tec_pred = []
        density = []
        Fe, Ni, Co, Cr, V, Cu = [], [], [], [], [], []
        packing = []

        for i, row in tqdm(enumerate(parser.iterate_rows())):
            # Get composition as mass fractions
            fe, ni, co, cr, v, cu = parser.get_Fe(row), parser.get_Ni(row), parser.get_Co(row), parser.get_Cr(row), parser.get_V(row), parser.get_Cu(row)
            
            # Create composition dictionary (mass fractions)
            composition_dict = {
                "Fe": fe, "Ni": ni, "Co": co, 
                "Cr": cr, "V": v, "Cu": cu
            }
            
            # Predict TEC with uncertainty
            print(f"Processing {i}")
            tec_mean, tec_std, tec_all = predict_tec_with_uncertainty(composition_dict, fraction_type='mass')
            
            tec_pred.append(tec_mean)
            density.append(parser.get_density(row))
            Fe.append(fe)
            Ni.append(ni)
            Co.append(co)
            Cr.append(cr)
            V.append(v)
            Cu.append(cu)
            packing.append((parser.get_phase(row)).replace("'", ""))

        tec = np.array(tec_pred)
        Fe, Ni, Co, Cr, V, Cu = np.array(Fe), np.array(Ni), np.array(Co), np.array(Cr), np.array(V), np.array(Cu)
    else:
        # Original version - use existing TEC values from Excel
        tec, density, Fe, Ni, Co, Cr, V, Cu, packing = extract_data(parser)

    points = np.column_stack((tec, density))

    pareto_indices = []
    for i, point in enumerate(points):
        if not is_dominated(point, np.delete(points, i, axis=0)):
            pareto_indices.append(i)
            
    if output and relabel:
        output_results(pareto_indices, Fe, Ni, Co, Cr, V, Cu, packing)
    elif output and not relabel:
        output_results(pareto_indices, Fe, Ni, Co, Cr, V, Cu, packing)
        
    return tec, density, points[pareto_indices], pareto_indices

def output_results(pareto_indices, Fe, Ni, Co, Cr, V, Cu, packing):
    pareto_elements = {
        "Fe": np.array(Fe)[pareto_indices],
        "Ni": np.array(Ni)[pareto_indices],
        "Co": np.array(Co)[pareto_indices],
        "Cr": np.array(Cr)[pareto_indices],
        "V": np.array(V)[pareto_indices],
        "Cu": np.array(Cu)[pareto_indices],
        "packing": np.array(packing)[pareto_indices]
    }

    pareto_df = pd.DataFrame(pareto_elements)
    print(pareto_df.to_markdown())
    print([list(x[:-1]) for x in pareto_df.to_numpy()])

def plot_results(tec_orig, density_orig, pareto_front_orig, tec_relabel, density_relabel, pareto_front_relabel):
    plt.figure(figsize=(8,5))
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = "Arial"
    
    # Plot original data and Pareto front
    plt.scatter(tec_orig, density_orig, label="Orig. data", alpha=0.5, c='#848484')
    
    # Identify outliers: TEC<3 and density<8200
    outlier_mask = (pareto_front_orig[:, 0] < 3) & (pareto_front_orig[:, 1] < 8200)
    outliers = pareto_front_orig[outlier_mask]
    non_outliers = pareto_front_orig[~outlier_mask]
    
    # Sort non-outliers for plotting the Pareto front line
    sorted_pareto_front = non_outliers[non_outliers[:, 0].argsort()]
    plt.plot(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1], 'r--s', label="Orig. Pareto front")
    
    # Plot outliers as hollow circles
    if len(outliers) > 0:
        plt.scatter(outliers[:, 0], outliers[:, 1], facecolors='none', edgecolors='red', s=50, label="Outliers", linewidths=1.5)

    # Plot relabeled data and Pareto front
    plt.scatter(tec_relabel, density_relabel, label="Relabeled data", alpha=0.5, c='#3da4ff')
    
    # Identify outliers in relabeled data: TEC<3 and density<8200
    outlier_mask_relabeled = (pareto_front_relabel[:, 0] < 3) & (pareto_front_relabel[:, 1] < 8200)
    outliers_relabeled = pareto_front_relabel[outlier_mask_relabeled]
    non_outliers_relabeled = pareto_front_relabel[~outlier_mask_relabeled]
    
    # Sort non-outliers for plotting the Pareto front line
    sorted_pareto_front_relabel = non_outliers_relabeled[non_outliers_relabeled[:, 0].argsort()]
    plt.plot(sorted_pareto_front_relabel[:, 0], sorted_pareto_front_relabel[:, 1], 'b--d', label="Relabeled Pareto front")
    
    # Plot outliers as hollow circles
    if len(outliers_relabeled) > 0:
        plt.scatter(outliers_relabeled[:, 0], outliers_relabeled[:, 1], 
                   facecolors='none', edgecolors='#3da4ff', s=50, label="Relabeled Outliers", linewidths=1.5)

    plt.text(0.05, 0.95, f"The error bars represent {STD_FACTOR}×std", transform=plt.gca().transAxes)
    plt.xlabel("TEC")
    plt.ylabel("Density")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    plt.tight_layout()

    plt.savefig("pareto_front_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    data_file = "dataset/Data_base_DFT_Thermal.xlsx"
    
    # Get original Pareto front
    tec_orig, density_orig, pareto_front_orig, pareto_indices_orig = get_pareto_front(data_file, relabel=False)
    
    # Get relabeled Pareto front
    tec_relabel, density_relabel, pareto_front_relabel, pareto_indices_relabel = get_pareto_front(data_file, relabel=True)
    
    # Plot comparison
    plot_results(tec_orig, density_orig, pareto_front_orig, tec_relabel, density_relabel, pareto_front_relabel)
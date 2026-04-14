#!/usr/bin/env python3
"""
Element sensitivity analysis for Pareto front compositions.
Adds small perturbations of various elements to Pareto compositions and predicts resulting TEC values.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import json
from tqdm import tqdm
from relabel_pareto import get_pareto_front
from quick_predict_tool import predict_tec_with_uncertainty, calculate_density_weighted_avg
from constraints_utils import mass_to_molar, molar_to_mass

# Load atomic masses for conversions
with open("constant/atomic_mass.json", 'r') as f:
    atomic_mass = json.load(f)

# Elements to test for sensitivity analysis
TEST_ELEMENTS = ['Li', 'Mg', 'Al', 'Si', 'Y', 'Ti']
PERTURBATION_LEVEL = 0.01  # 1% perturbation

def perturb_composition(composition_dict, element, perturbation_level):
    """
    Add a small amount of an element to a composition and renormalize.
    
    Parameters:
    composition_dict (dict): Original composition in mass fractions
    element (str): Element to add
    perturbation_level (float): Amount of element to add (as molar fraction)
    
    Returns:
    dict: Perturbed composition in mass fractions
    """
    # Create a copy of the composition
    original_comp = composition_dict.copy()
    
    # Get the list of elements in the composition
    element_list = list(original_comp.keys())
    
    # Convert mass fractions to molar fractions
    mass_values = np.array([original_comp[elem] for elem in element_list])
    molar_values = mass_to_molar(mass_values, element_list)
    molar_comp = dict(zip(element_list, molar_values))
    
    # Add the new element if not present
    if element not in molar_comp:
        molar_comp[element] = 0.0
        element_list.append(element)
    
    # Get total of original elements
    total_molar = sum(molar_comp[elem] for elem in element_list if elem in molar_comp)
    
    # Add perturbation in molar fraction
    molar_comp[element] += perturbation_level * total_molar
    
    # Renormalize molar fractions to 100%
    new_total_molar = sum(molar_comp[elem] for elem in element_list if elem in molar_comp)
    for elem in element_list:
        if elem in molar_comp:
            molar_comp[elem] = molar_comp[elem] * total_molar / new_total_molar
    
    # Convert back to mass fractions
    molar_values = np.array([molar_comp[elem] for elem in element_list])
    mass_values = molar_to_mass(molar_values, element_list)
    perturbed_comp = dict(zip(element_list, mass_values))
        
    return perturbed_comp

def analyze_sensitivity():
    """
    Perform sensitivity analysis on Pareto front compositions.
    """
    # Get Pareto front data
    data_file = "dataset/Data_base_DFT_Thermal.xlsx"
    tec, density, pareto_points, pareto_indices = get_pareto_front(data_file, relabel=False)
    
    # Get the compositions of Pareto points
    parser = None
    from stat_distribution import ExcelParser
    parser = ExcelParser(data_file)
    
    pareto_compositions = []
    for i, row in enumerate(parser.iterate_rows()):
        if i in pareto_indices:
            # Create composition dictionary (mass fractions)
            composition_dict = {
                "Fe": parser.get_Fe(row), 
                "Ni": parser.get_Ni(row), 
                "Co": parser.get_Co(row), 
                "Cr": parser.get_Cr(row), 
                "V": parser.get_V(row), 
                "Cu": parser.get_Cu(row)
            }
            pareto_compositions.append(composition_dict)
    
    # Get all data points for background plotting
    tec_all, density_all, _, _ = get_pareto_front(data_file, relabel=False)
    
    # Number of Pareto points
    n_pareto = len(pareto_compositions)
    
    # Sort Pareto points by TEC values to ensure continuity when plotting
    # Create array with TEC, density, and original index
    pareto_data = np.column_stack((pareto_points[:, 0], pareto_points[:, 1], np.arange(len(pareto_points))))
    sorted_indices = np.argsort(pareto_data[:, 0])  # Sort by TEC values
    sorted_pareto_data = pareto_data[sorted_indices]
    sorted_pareto_points = sorted_pareto_data[:, :2]
    sorted_original_indices = sorted_pareto_data[:, 2].astype(int)
    
    # Reorder compositions and other data according to sorted indices
    sorted_pareto_compositions = [pareto_compositions[i] for i in sorted_original_indices]
    sorted_pareto_tec = [tec[pareto_indices[i]] for i in sorted_original_indices]
    sorted_pareto_density = [density[pareto_indices[i]] for i in sorted_original_indices]
    
    # Create subplots - arrange in 3 rows
    n_cols = (n_pareto + 2) // 3  # Calculate number of columns needed for 3 rows
    fig, axes = plt.subplots(3, n_cols, figsize=(6*n_cols, 5*3))
    if n_pareto == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Colors for different test elements
    colors = plt.cm.tab10(np.linspace(0, 1, len(TEST_ELEMENTS)))
    
    # Process each Pareto point
    for idx, (ax, composition) in enumerate(zip(axes, sorted_pareto_compositions)):
        print(f"Processing Pareto point {idx+1}/{n_pareto}")
        
        # Plot all data points in gray
        ax.scatter(tec_all, density_all, c='#848484', alpha=0.5, label='All data')
        
        # Plot original Pareto front (sorted for continuity)
        ax.plot(sorted_pareto_points[:, 0], sorted_pareto_points[:, 1], 'r--s', label='Pareto front')
        
        # Mark the current Pareto point with a black square (experimental values)
        ax.scatter(sorted_pareto_tec[idx], sorted_pareto_density[idx], 
                  s=100, facecolors='none', edgecolors='black', marker='s', linewidths=2,
                  label=f'Pareto point {idx+1} (raw)')
        
        # Predict TEC for original composition
        try:
            original_tec_mean, original_tec_std, _ = predict_tec_with_uncertainty(composition, fraction_type='mass')
            original_density = calculate_density_weighted_avg(composition, fraction_type='mass')
            
            # Mark the current Pareto point with model prediction (blue filled circle)
            ax.scatter(original_tec_mean, original_density, 
                      s=100, color='blue', marker='o', linewidths=2,
                      label=f'Pareto point {idx+1} (pred)')
        except Exception as e:
            print(f"    Error predicting TEC for original composition: {e}")
            original_tec_mean, original_density = None, None
        
        # Test each element
        for elem_idx, element in enumerate(TEST_ELEMENTS):
            print(f"  Testing element {element}")
            
            # Perturb the composition
            perturbed_comp = perturb_composition(composition, element, PERTURBATION_LEVEL)
            
            # Predict TEC for perturbed composition
            try:
                tec_mean, tec_std, _ = predict_tec_with_uncertainty(perturbed_comp, fraction_type='mass')
                
                # Calculate density for perturbed composition
                perturbed_density = calculate_density_weighted_avg(perturbed_comp, fraction_type='mass')
                
                # Plot the perturbed point
                ax.scatter(tec_mean, perturbed_density, 
                          c=[colors[elem_idx]], s=60, marker='o',
                          label=f'+{PERTURBATION_LEVEL*100:.0f} at.% {element}')
            except Exception as e:
                print(f"    Error predicting TEC for {element}: {e}")
                continue
        
        ax.set_xlabel('TEC')
        ax.set_ylabel('Density')
        ax.set_title(f'Pareto Point {idx+1} Sensitivity')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_pareto, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('element_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Sensitivity analysis completed. Results saved to 'element_sensitivity_analysis.png'")

if __name__ == "__main__":
    analyze_sensitivity()
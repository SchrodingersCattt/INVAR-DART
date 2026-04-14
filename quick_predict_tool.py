# file:/aisi-nas/guomingyu/PROPERTIES_PREDICTION/INVAR-2025/iter01.GA/dpa_v3_1/quick_predict_tool.py

import numpy as np
import glob
import json
from deepmd.pt.infer.deep_eval import DeepProperty
from target import comp2struc, pred, get_packing
from constraints_utils import mass_to_molar as mass_to_molar_converter, molar_to_mass

# Load atomic masses for conversions
with open("constant/atomic_mass.json", 'r') as f:
    atomic_mass = json.load(f)

# Load densities for weighted average calculation
with open("constant/densities.json", 'r') as f:
    densities_dict = json.load(f)

def calculate_density_weighted_avg(composition_dict, fraction_type='molar'):
    """
    Calculate weighted average density based on composition
    
    Parameters:
    composition_dict (dict): Dictionary with element symbols as keys and proportions as values
    fraction_type (str): Either 'mass' or 'molar' to indicate the type of proportions
    
    Returns:
    float: Weighted average density
    """
    
    # Extract elements and proportions from dictionary
    elements = list(composition_dict.keys())
    proportions = list(composition_dict.values())
    # if any composition > 1, normalize
    proportions = [c / sum(proportions) for c in proportions]
    # Convert to molar fractions if needed
    if fraction_type == 'mass':
        # Convert mass fractions to molar fractions
        mass_array = np.array(proportions)
        molar_proportions = mass_to_molar_converter(mass_array, elements)
        proportions = molar_proportions.tolist()
    elif fraction_type != 'molar':
        raise ValueError("fraction_type must be either 'mass' or 'molar'")
    
    # Calculate weighted average density
    density = 0
    for i, e in enumerate(elements):
        c = proportions[i]
        density += c * densities_dict[e]
    
    return density


def normalize_composition_dict(composition_dict):
    """
    Normalize composition dictionary so that sum equals 1.0
    
    Parameters:
    composition_dict (dict): Dictionary with element symbols as keys and proportions as values
    
    Returns:
    dict: Normalized composition dictionary
    """
    total = sum(composition_dict.values())
    if total == 0:
        raise ValueError("Composition sum cannot be zero")
    return {k: v/total for k, v in composition_dict.items()}


def predict_tec_with_uncertainty(composition_dict, fraction_type='molar'):
    """
    Predict TEC for given composition with uncertainty
    
    Parameters:
    composition_dict (dict): Dictionary with element symbols as keys and proportions as values
    fraction_type (str): Either 'mass' or 'molar' to indicate the type of proportions
    
    Returns:
    tuple: (mean_tec, std_tec, all_predictions) - Mean TEC, standard deviation, and all predictions
    """
    
    # Normalize composition
    composition_dict = normalize_composition_dict(composition_dict)
    
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
    tec_model_paths = glob.glob('models/tec*.pt')
    if not tec_model_paths:
        raise FileNotFoundError("No TEC model found in models/tec*.pt")
    
    tec_models = [DeepProperty(model_path) for model_path in tec_model_paths]
    
    # Predict TEC for all structures and models
    tec_predictions = []
    for model in tec_models:
        for struct in struct_list:
            pred_tec = pred(model, struct)
            tec_predictions.append(pred_tec)
    
    # Return mean, standard deviation, and all predictions
    mean_tec = np.mean(tec_predictions)
    std_tec = np.std(tec_predictions)
    return mean_tec, std_tec, tec_predictions

def format_composition(composition_dict):
    """Format composition dictionary as a string"""
    return ", ".join([f"{e}:{v:.4f}" for e, v in composition_dict.items()])

def compare_tec_multiple(compositions, fraction_type='mass', group_names=None):
    """
    Compare TEC predictions for multiple compositions with clean output
    
    Parameters:
    compositions (list): List of composition dictionaries or list of lists of composition dictionaries
    fraction_type (str): Either 'mass' or 'molar' to indicate the type of proportions
    group_names (list): Optional list of names for each group
    
    Returns:
    list: List of dictionaries containing comparison results
    """
    
    results = []
    
    # Handle both cases: list of compositions or list of groups of compositions
    if isinstance(compositions[0], dict):
        # Single group case - all compositions in one group
        compositions = [compositions]
        if group_names and len(group_names) > 0:
            group_names = [group_names[0]]
        else:
            group_names = ["Group 1"]
    
    for i, composition_group in enumerate(compositions):
        # Generate group name if not provided
        group_name = group_names[i] if group_names and i < len(group_names) else f"Group {i+1}"
        
        # Process all compositions in the group
        tec_results = []
        density_results = []
        formatted_comps = []
        converted_comps = []
        
        # Get all elements from all compositions in the group
        all_elements = set()
        for comp in composition_group:
            all_elements.update(comp.keys())
        all_elements = list(all_elements)
        
        # Convert compositions if needed for display and predict TEC
        for j, comp in enumerate(composition_group):
            # Normalize composition first
            normalized_comp = normalize_composition_dict(comp)
            
            # Convert compositions if needed for display
            if fraction_type == 'mass':
                # Convert mass to molar for display
                mass_array = np.array(list(normalized_comp.values()))
                elements = list(normalized_comp.keys())
                
                molar_array = mass_to_molar_converter(mass_array, elements)
                comp_molar = dict(zip(elements, molar_array))
                
                comp_display = format_composition(normalized_comp)  # Normalized mass
                comp_converted = format_composition(comp_molar)  # Converted to molar
            else:
                # For molar fractions, convert to mass for display
                molar_array = np.array(list(normalized_comp.values()))
                elements = list(normalized_comp.keys())
                
                mass_array = molar_to_mass(molar_array, elements)
                
                comp_mass = dict(zip(elements, mass_array))
                
                comp_display = format_composition(normalized_comp)   # Normalized molar
                comp_converted = format_composition(comp_mass)  # Converted to mass
            
            formatted_comps.append(comp_display)
            converted_comps.append(comp_converted)
            
            # Predict TEC with uncertainties using normalized composition
            tec_mean, tec_std, tec_all = predict_tec_with_uncertainty(normalized_comp, fraction_type=fraction_type)
            tec_results.append({
                "mean": tec_mean,
                "std": tec_std,
                "all": tec_all
            })
            
            # Calculate density using normalized composition
            density = calculate_density_weighted_avg(normalized_comp, fraction_type=fraction_type)
            density_results.append(density)
        
        results.append({
            "group": group_name,
            "elements": all_elements,
            "compositions_original": formatted_comps,
            "compositions_converted": converted_comps,
            "tec_results": tec_results,
            "density_results": density_results
        })
    
    # Print clean results table
    print(f"{'Group':<25}", end="")
    for i in range(len(results[0]['compositions_original'])):
        print(f" {'TEC'+str(i+1)+'±Std':<15}", end="")
    print(" {'Diff Range':<15}")
    print("-" * (25 + 16 * len(results[0]['compositions_original']) + 16))
    
    for result in results:
        print(f"{result['group']:<25}", end="")
        tec_values = []
        for tec_result in result['tec_results']:
            tec_str = f"{tec_result['mean']:.4f}±{tec_result['std']:.4f}"
            print(f" {tec_str:<15}", end="")
            tec_values.append(tec_result['mean'])
        
        # Calculate difference range
        if len(tec_values) > 1:
            diff_range = max(tec_values) - min(tec_values)
            print(f" {diff_range:<15.4f}")
        else:
            print(f" {'N/A':<15}")
    
    # Print detailed compositions if needed
    print("\nDetailed Compositions (Normalized):")
    print("=" * 100)
    for result in results:
        print(f"\n{result['group']}:")
        for j, (comp_orig, comp_conv) in enumerate(zip(result['compositions_original'], result['compositions_converted'])):
            if fraction_type == 'mass':
                print(f"  Comp {j+1} (mass_frac, normalized): {comp_orig}")
            else:
                print(f"  Comp {j+1} (molar%, normalized): {comp_orig}")
            
            # Also show converted values
            if fraction_type == 'mass':
                print(f"  Comp {j+1} (molar%, from mass): {comp_conv}")
            else:
                print(f"  Comp {j+1} (mass_frac, from molar): {comp_conv}")
                
            # Print density
            print(f"  Density: {result['density_results'][j]:.2f} kg/m³")
                
        for j, tec_result in enumerate(result['tec_results']):
            print(f"  TEC{j+1}: {tec_result['mean']:.4f} ± {tec_result['std']:.4f}")
    
    return results

def compare_tec_clean(composition_pairs, fraction_type='mass', group_names=None):
    """
    Compare TEC predictions for pairs of compositions with clean output
    
    Parameters:
    composition_pairs (list): List of tuples, each containing two composition dictionaries
    fraction_type (str): Either 'mass' or 'molar' to indicate the type of proportions
    group_names (list): Optional list of names for each composition pair
    
    Returns:
    list: List of dictionaries containing comparison results
    """
    
    # Convert tuples to lists for the new function
    composition_lists = []
    for pair in composition_pairs:
        if isinstance(pair, tuple):
            composition_lists.append(list(pair))
        else:
            composition_lists.append([pair])
    
    # Use the new multiple comparison function
    return compare_tec_multiple(composition_lists, fraction_type, group_names)

if __name__ == "__main__":
    # Example usage with the compositions you provided
    
    # Define composition pairs as dictionaries
    # composition_pairs = [
    #     # Group 1 - Iter00 Candidate 1
    #     (
    #         {"Fe": 0.6133, "Ni": 0.2896, "Co": 0.0698, "Cr": 0.0124, "V": 0, "Al": 0.0149, "Ti": 0},
    #     ),
    #     # Group 2 - Iter00 Candidate 2
    #     (
    #         {"Fe": 0.5671, "Ni": 0.2714, "Co": 0.1049, "Cr": 0.0111, "V": 0, "Al": 0, "Ti": 0.0455},
    #     ),
    #     # Group 3 - Iter00 Candidate 3
    #     (
    #         {"Fe": 0.5409, "Ni": 0.2137, "Co": 0.1601, "Cr": 0.0589, "V": 0, "Al": 0.0264, "Ti": 0},
    #     ),
    #     # Group 4 - Iter01 Candidate 1
    #     (
    #         {"Fe": 0.5886, "Ni": 0.2290, "Co": 0.1192, "Cr": 0.0580, "Ti": 0.0050, "Al": 0, "V": 0},
    #     ),
    #     # Group 5 - Iter01 Candidate 2
    #     (
    #        {"Fe": 0.6338, "Ni": 0.2854, "Co": 0.0566, "Cr": 0.0156, "Ti": 0, "Al": 0.0088, "V": 0},
    #     ),
    #     # Group 6 - Iter01 Candidate 3
    #     (
    #         {"Fe": 0.4017, "Ni": 0.0410, "Co": 0.4490, "Cr": 0.0990, "Ti": 0.0020, "Al": 0.0033, "V": 0.0040},
    #     ),

    #     # Group 1 - Iter02 Candidate 1
    #     (
    #         {"Fe": 64.4, "Ni": 28.2, "Co": 4.67, "Cr": 0.0, "V": 0.4, "Ti": 0.0, "Al": 2.3, "Cu": 0.0},
    #     ),
    #     # Group 2 - Iter02 Candidate 2
    #     (
    #         {"Fe": 60.57, "Ni": 1.3, "Co": 31.23, "Cr": 0.0, "V": 0.0, "Ti": 3.43, "Al": 3.53, "Cu": 0.0},
    #     ),
    #     # Group 3 - Iter02 Candidate 3
    #     (
    #         {"Fe": 69.93, "Ni": 5.9, "Co": 19.27, "Cr": 0.0, "V": 0.0, "Ti": 1.77, "Al": 3.17, "Cu": 0.0},
    #     ),
    #     # Group 4 - Iter02 Candidate 4
    #     (
    #         {"Fe": 77.1, "Ni": 0.47, "Co": 14.93, "Cr": 0.63, "V": 0.9, "Ti": 0.0, "Al": 5.97, "Cu": 0.0},
    #     ),
    #     # Group 5 - Iter02 Candidate 5
    #     (
    #         {"Fe": 17.7, "Ni": 19.9, "Co": 19.9, "Cr": 19.5, "V": 0.0, "Ti": 11.2, "Al": 11.8, "Cu": 0.0},
    #     )
    # ]
    # composition_pairs = [
    #     # C11
    #     (
    #         {"Fe": 42.36, "Ni": 9.37, "Co": 39.4, "Cr": 8.87, "V": 0.0, "Ti": 0.0, "Al": 0.0, "Cu": 0.0},
    #     ),
    #     # C12
    #     (
    #         {"Fe": 54.63, "Ni": 21.47, "Co": 17.03, "Cr": 6.87, "V": 0.0, "Ti": 0.0, "Al": 0.0, "Cu": 0.0},
    #     ),
    #     # C13
    #     (
    #         {"Fe": 49.17, "Ni": 17.03, "Co": 21.8, "Cr": 6.97, "V": 0.0, "Ti": 0.0, "Al": 0.0, "Cu": 5.0},
    #     ),
    #     # C14
    #     (
    #         {"Fe": 40.6, "Ni": 6.73, "Co": 36.6, "Cr": 10.27, "V": 0.0, "Ti": 0.0, "Al": 0.0, "Cu": 5.73},
    #     )
    # ]
    
    composition_pairs = [
        # Iter03 Candidate 1
        (
            {"Fe": 62.09, "Ni": 27.68, "Co": 7.25, "Cr": 1.6, "V": 0.89, "Ti": 0.0, "Al": 0.49, "Cu": 0.0},
        ),
        # Iter03 Candidate 2
        (
            {"Fe": 60.78, "Ni": 28.9, "Co": 3.11, "Cr": 0.39, "V": 6.19, "Ti": 0.0, "Al": 0.63, "Cu": 0.0},
        ),
        # Iter03 Candidate 3
        (
            {"Fe": 64.36, "Ni": 30.54, "Co": 4.06, "Cr": 0.0, "V": 0.0, "Ti": 0.0, "Al": 0.87, "Cu": 0.0},
        ),
        # Iter03 Candidate 4
        (
            {"Fe": 63.41, "Ni": 29.52, "Co": 4.77, "Cr": 0.92, "V": 1.38, "Ti": 0.0, "Al": 0.0, "Cu": 0.0},
        )
    ]

    # Optional group names
    group_names = [
        # "iter0_test1",
        # "iter0_test2",
        # "iter0_test3",
        
        # "iter1_test1",
        # "iter1_test2",
        # "iter1_test3",

        # "iter2_test1",
        # "iter2_test2",
        # "iter2_test3",
        # "iter2_test4",
        # "iter2_test5"

        "iter3_test1",
        "iter3_test2",
        "iter3_test3",
    ]

    
    # Compare TEC values
    results = compare_tec_multiple(composition_pairs, fraction_type='molar', group_names=group_names)
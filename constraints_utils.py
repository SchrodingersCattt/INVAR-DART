import logging
import json
import numpy as np

CLIPPING_FACTOR = 100
print("CLIPPING_FACTOR:", CLIPPING_FACTOR)
# constraints
def parse_constraints(constraints_str):
    constraints = {}
    if constraints_str:
        for constraint in constraints_str.split(','):
            constraint = constraint.strip()
            if '(' in constraint and ')' in constraint:
                # Handle sum constraints
                elements_part = constraint[constraint.find('(')+1:constraint.find(')')]
                elements = [e.strip() for e in elements_part.split('+')]
                condition = constraint[constraint.find(')')+1]
                value = float(constraint[constraint.find(')')+2:])
                constraints[tuple(elements)] = f"{condition}{value}"
            else:
                # Handle single element constraints
                if '<' in constraint:
                    element, condition = constraint.split('<')
                    constraints[element.strip()] = f"<{condition}"
                elif '>' in constraint:
                    element, condition = constraint.split('>')
                    constraints[element.strip()] = f">{condition}"
                elif '=' in constraint:
                    element, condition = constraint.split('=')
                    constraints[element.strip()] = f"={condition}"
    return constraints


def calculate_constraint_penalty(compositions, elements, constraints, lambda_=1000, eq_tol=1e-6):
    """
    Calculate penalty for constraint violations instead of directly modifying compositions.
    Returns a penalty value that can be added to the fitness function.
    """
    penalty = 0.0
    
    # sum constraints
    for elements_tuple, condition_str in constraints.items():
        if isinstance(elements_tuple, tuple):
            indices = [elements.index(e) for e in elements_tuple]
            current_sum = sum(compositions[i] for i in indices)
            condition, value = condition_str[0], float(condition_str[1:])

            if condition == '<' and current_sum > value:
                penalty += lambda_ * max(0.0, current_sum - value) ** 2
            elif condition == '>' and current_sum < value:
                penalty += lambda_ * max(0.0, value - current_sum) ** 2
            elif condition == '=' and abs(current_sum - value) > eq_tol:
                penalty += lambda_ * (current_sum - value) ** 2

    # single element constraints
    for element, condition_str in constraints.items():
        if isinstance(element, str):
            i = elements.index(element)
            condition, value = condition_str[0], float(condition_str[1:])

            if condition == '<' and compositions[i] > value:
                penalty += lambda_ * max(0.0, compositions[i] - value) ** 2
            elif condition == '>' and compositions[i] < value:
                penalty += lambda_ * max(0.0, value - compositions[i]) ** 2
            elif condition == '=' and abs(compositions[i] - value) > eq_tol:
                penalty += lambda_ * (compositions[i] - value) ** 2

    return penalty


def apply_constraints(compositions, elements, constraints):
    """
    Apply soft constraints by adjusting compositions toward feasible region 
    without hard clipping. Uses scaling approaches that maintain relative 
    proportions where possible.
    """
    modified_compositions = np.array(compositions).copy()
    
    # First handle sum constraints
    for elements_tuple, condition_str in constraints.items():
        if isinstance(elements_tuple, tuple):
            indices = [elements.index(e) for e in elements_tuple]
            current_sum = sum(modified_compositions[i] for i in indices)
            condition, value = condition_str[0], float(condition_str[1:])
            
            if condition == '<' and current_sum > value:
                # Scale down the elements proportionally
                scale = value / current_sum
                for i in indices:
                    modified_compositions[i] *= scale
                    
    # Then handle single element constraints with soft adjustments
    for element, condition_str in constraints.items():
        if isinstance(element, str):
            i = elements.index(element)
            condition, value = condition_str[0], float(condition_str[1:])
            
            if condition == '<' and modified_compositions[i] > value:
                # Move value toward upper limit with some softness
                # Instead of hard clip, use a sigmoid-like adjustment
                excess = modified_compositions[i] - value
                adjustment_factor = 1.0 / (1.0 + excess * CLIPPING_FACTOR)  # Soft transition
                modified_compositions[i] = value + excess * adjustment_factor
            elif condition == '>' and modified_compositions[i] < value:
                # Move value toward lower limit with some softness
                deficit = value - modified_compositions[i]
                adjustment_factor = 1.0 / (1.0 + deficit * CLIPPING_FACTOR)  # Soft transition
                modified_compositions[i] = value - deficit * adjustment_factor
            elif condition == '=' and modified_compositions[i] != value:
                # Move value toward target with some softness
                deviation = modified_compositions[i] - value
                adjustment_factor = 1.0 / (1.0 + abs(deviation) * CLIPPING_FACTOR)  # Soft transition
                modified_compositions[i] = value + deviation * adjustment_factor
                
    # Renormalize to ensure mole fractions sum to 1
    # But only if there are significant violations
    total = np.sum(modified_compositions)
    if total <= 0 or abs(total - 1.0) > 1e-3:
        modified_compositions = np.abs(modified_compositions)  # Ensure non-negative
        modified_compositions /= np.sum(modified_compositions)
    
    return modified_compositions


## atomic mass and molar mass conversion
atoms_mass_file = "constant/atomic_mass.json"
with open(atoms_mass_file, 'r') as atoms_mass_file:
    atomic_mass = json.load(atoms_mass_file)

def mass_to_molar(mass_comp: np.ndarray, element_list: list) -> np.ndarray:
    mass_comp = np.array(mass_comp)
    molar_compositions = np.array([
        mass_comp[i] / atomic_mass[element_list[i]] 
        for i in range(len(element_list))
    ])
    return molar_compositions / np.sum(molar_compositions)

def molar_to_mass(molar_comp: np.ndarray, element_list: list) -> np.ndarray:
    molar_comp = np.array(molar_comp)
    mass_compositions = np.array([
        molar_comp[i] * atomic_mass[element_list[i]] 
        for i in range(len(element_list))
    ])
    return mass_compositions / np.sum(mass_compositions)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
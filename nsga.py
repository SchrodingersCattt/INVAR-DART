import argparse
import logging
import json
import numpy as np
from target import comp2struc, z_core, norm2orig, pred
from constraints_utils import apply_constraints, parse_constraints, mass_to_molar, molar_to_mass, sigmoid
from tqdm import tqdm
from deepmd.pt.infer.deep_eval import DeepProperty
import glob


atomic_mass_file = "constant/atomic_mass.json"
density_file = "constant/densities.json"
with open(density_file, 'r') as f:
    densities_dict = json.load(f)

class NSGAII:
    def __init__(self, elements, population_size=10, generations=100, crossover_rate=0.8, mutation_rate=0.1,
                 init_population=None, constraints={}, a=0.9, b=0.1, c=0.9, d=0.1,
                 get_density_mode='weighted_avg'):
        self.elements = elements
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.constraints = constraints
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.get_density_mode = get_density_mode
        self.population_size = population_size
        tec_models = glob.glob('models/tec*.pt')
        self.tec_models = [DeepProperty(model) for model in tec_models]

        # Handle population initialization
        if init_population:
            logging.info("Initial population provided, manipulating sizes if necessary.")
            _manipulated_population = self.manipulate_population_size(init_population, population_size)
            self.population = [mass_to_molar(ind, self.elements) for ind in _manipulated_population]
        else:
            logging.info("Initial population not provided, using population size to randomize populations.")
            self.population = self.initialize_population(population_size)

        logging.info(f"Population size: {self.population_size}")

    def manipulate_population_size(self, population, population_size):
        manipulated_population = []

        # Adjust individual sizes (fill or truncate) based on the elements count
        for individual in population:
            if len(individual) < len(self.elements):
                individual = np.pad(individual, (0, len(self.elements) - len(individual)), mode='constant')
                logging.info(f"Padded individual: {individual}")
            elif len(individual) > len(self.elements):
                individual = individual[:len(self.elements)]
                logging.info(f"Truncated individual: {individual}")

            # Normalize to ensure mole fractions sum to 1
            individual = np.array(individual)
            individual = individual / np.sum(individual)
            manipulated_population.append(individual)

        # If the population size is greater than initial population size, add random compositions
        if population_size > len(manipulated_population):
            logging.info(f"Population size {population_size} is greater than initial population size {len(manipulated_population)}.")
            remaining_size = population_size - len(manipulated_population)
            manipulated_population.extend([self.random_composition() for _ in range(remaining_size)])

        # If the population size is less than or equal to the initial population size, truncate it
        elif population_size < len(manipulated_population):
            logging.info(f"Population size {population_size} is less than initial population size {len(manipulated_population)}.")
            manipulated_population = manipulated_population[:population_size]

        return manipulated_population

    def initialize_population(self, population_size):
        logging.info("Initializing population.")
        population = [self.random_composition() for _ in range(population_size)]
        if self.constraints:
            population = [apply_constraints(ind, self.elements, self.constraints) for ind in population]
        if not population:
            raise ValueError("Population initialization failed: population is empty.")
        return population

    def random_composition(self):
        logging.info("Generating random composition.")
        # Generate random mole fractions using Dirichlet distribution
        molar_comp = np.random.dirichlet(np.ones(len(self.elements)), size=1)[0]
        if self.constraints:
            molar_comp = apply_constraints(molar_comp, self.elements, self.constraints)
        return molar_comp

    def evaluate_objectives(self, comp):
        """
        Evaluate multiple objectives for NSGA-II:
        1. TEC mean (minimize)
        2. TEC std (maximize) 
        3. Density mean (minimize)
        4. Density std (maximize)
        """
        logging.info(f"Evaluating objectives for composition: {comp}")
        if self.constraints:
            # Apply constraints in mole fraction
            molar_comp = apply_constraints(comp, self.elements, self.constraints)
        else:
            molar_comp = comp
            
        # Get the individual objective values
        tec_mean, tec_std, density_mean, density_std = self.get_objective_values(self.elements, molar_comp)
        
        # Apply weights to objectives
        weighted_tec_mean = self.a * tec_mean
        weighted_tec_std = self.b * (-tec_std)  # Negative because we want to maximize std but NSGA-II minimizes
        weighted_density_mean = self.c * density_mean
        weighted_density_std = self.d * (-density_std)  # Negative because we want to maximize std but NSGA-II minimizes
        
        # Return as array (these are all to be minimized)
        return np.array([weighted_tec_mean, weighted_tec_std, weighted_density_mean, weighted_density_std])

    def get_objective_values(self, elements, compositions):
        """
        Get the individual objective values from the target function
        """
        packing = 'fcc'  # Using default packing as in target.py
        
        # These values are from the target.py file
        tec_mean_orig = 9.76186694677871
        tec_std_orig = 4.3042156360248125
        density_mean_orig = 8331.903892865434
        density_std_orig = 182.21803336559455

        # Evaluate the objectives separately
        tec_models = self.tec_models
        
        struct_list = comp2struc(elements, compositions, packing=packing)
        
        # TEC values
        pred_tec = [z_core(pred(m, s), mean=tec_mean_orig, std=tec_std_orig) for m in tec_models for s in tqdm(struct_list)]
        pred_tec_mean = np.mean(pred_tec)
        pred_tec_std = np.std(pred_tec)
        
        # Convert back to original scale for logging
        pred_tec_mean_orig = norm2orig(pred_tec_mean, mean=tec_mean_orig, std=tec_std_orig)
        pred_tec_std_orig = pred_tec_std * tec_std_orig  # std scales only with std
        
        # Density values
        if self.get_density_mode == "weighted_avg":
            density = 0
            for i, e in enumerate(elements):
                c = compositions[i]
                density += c * densities_dict[e]
            pred_density_mean = z_core(density, mean=density_mean_orig, std=density_std_orig)
            pred_density_std = 0  # For weighted average, std is 0
            pred_density_mean_orig = norm2orig(pred_density_mean, mean=density_mean_orig, std=density_std_orig)
            pred_density_std_orig = pred_density_std * density_std_orig
            

        # Apply penalties for TEC > 10 or density > 8200
        penalty = 0
        tec_penalty_factor = 1000  # Large penalty factor for TEC > 10
        density_penalty_factor = 10  # Large penalty factor for density > 8200
        
        if pred_tec_mean_orig > 10:
            # Penalty increases quadratically with how much TEC exceeds 10
            tec_excess = pred_tec_mean_orig - 10
            penalty += tec_penalty_factor * (tec_excess ** 2)
            
        if pred_density_mean_orig > 8500:
            # Penalty increases quadratically with how much density exceeds 8200
            density_excess = pred_tec_mean_orig - 8500
            penalty += density_penalty_factor * (density_excess ** 2)
        
        # Apply penalties to the normalized values
        pred_tec_mean += penalty
        pred_density_mean += penalty
            
        # Log physical meaningful values
        logging.info(f"Physical values - TEC mean: {pred_tec_mean_orig}, TEC std: {pred_tec_std_orig}, "
                    f"Density mean: {pred_density_mean_orig}, Density std: {pred_density_std_orig}")
        if penalty > 0:
            logging.info(f"Applied penalty: {penalty}")
       
        return pred_tec_mean, pred_tec_std, pred_density_mean, pred_density_std

    def non_dominated_sort(self, objective_values):
        """
        Perform non-dominated sorting on the population
        Returns fronts: list of lists, where each sublist contains indices of individuals in that front
        """
        fronts = [[]]
        domination_counts = np.zeros(len(objective_values))
        dominated_solutions = [[] for _ in range(len(objective_values))]
        
        # For each solution, check which solutions it dominates
        for p in range(len(objective_values)):
            for q in range(len(objective_values)):
                if self.dominates(objective_values[p], objective_values[q]):
                    dominated_solutions[p].append(q)
                elif self.dominates(objective_values[q], objective_values[p]):
                    domination_counts[p] += 1
                    
            if domination_counts[p] == 0:
                fronts[0].append(p)
                
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
            
        # Remove the last empty front
        fronts.pop()
        return fronts

    def dominates(self, obj1, obj2):
        """
        Check if obj1 dominates obj2 (minimization problem)
        """
        # obj1 dominates obj2 if:
        # 1. obj1 is not worse than obj2 in all objectives
        # 2. obj1 is strictly better than obj2 in at least one objective
        not_worse = all(obj1[i] <= obj2[i] for i in range(len(obj1)))
        strictly_better = any(obj1[i] < obj2[i] for i in range(len(obj1)))
        return not_worse and strictly_better

    def calculate_crowding_distance(self, objective_values):
        """
        Calculate crowding distance for each individual
        """
        population_size = len(objective_values)
        num_objectives = len(objective_values[0]) if population_size > 0 else 0
        
        # Initialize distances
        distances = np.zeros(population_size)
        
        # For each objective
        for obj_idx in range(num_objectives):
            # Sort by objective value
            sorted_indices = sorted(range(population_size), key=lambda x: objective_values[x][obj_idx])
            
            # Assign infinite distance to boundary points
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate range for normalization
            obj_min = objective_values[sorted_indices[0]][obj_idx]
            obj_max = objective_values[sorted_indices[-1]][obj_idx]
            obj_range = obj_max - obj_min if obj_max != obj_min else 1
            
            # Calculate distances for intermediate points
            for i in range(1, population_size - 1):
                prev_idx = sorted_indices[i - 1]
                curr_idx = sorted_indices[i]
                next_idx = sorted_indices[i + 1]
                
                distances[curr_idx] += (
                    (objective_values[next_idx][obj_idx] - objective_values[prev_idx][obj_idx]) / obj_range
                )
                
        return distances

    def tournament_selection(self, population, objective_values, fronts, crowding_distances):
        """
        Tournament selection based on dominance and crowding distance
        """
        tournament_size = 2
        selected = []
        
        for _ in range(len(population)):
            # Select tournament participants
            participants = np.random.choice(len(population), tournament_size, replace=False)
            
            # Select the best participant
            best = self.select_best_from_tournament(
                participants, objective_values, fronts, crowding_distances
            )
            selected.append(population[best])
            
        return selected

    def select_best_from_tournament(self, participants, objective_values, fronts, crowding_distances):
        """
        Select the best individual from tournament participants
        """
        # Get front indices for participants
        participant_fronts = []
        for p in participants:
            for front_idx, front in enumerate(fronts):
                if p in front:
                    participant_fronts.append(front_idx)
                    break
                    
        # Find the best participant
        best = participants[0]
        best_front = participant_fronts[0]
        best_distance = crowding_distances[best]
        
        for i in range(1, len(participants)):
            p = participants[i]
            front_idx = participant_fronts[i]
            distance = crowding_distances[p]
            
            # Better if in a better front (lower index)
            if front_idx < best_front:
                best = p
                best_front = front_idx
                best_distance = distance
            # If in same front, better if has larger crowding distance
            elif front_idx == best_front and distance > best_distance:
                best = p
                best_distance = distance
                
        return best

    def crossover(self, parent1, parent2):
        logging.info("Crossover.")
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(self.elements) - 1)
            offspring1 = np.concatenate((parent1[:point], parent2[point:]))
            offspring2 = np.concatenate((parent2[:point], parent1[point:]))
            # Normalize offspring
            offspring1 /= np.sum(offspring1)
            offspring2 /= np.sum(offspring2)
            if self.constraints:
                # Apply constraints in mole fraction
                offspring1 = apply_constraints(offspring1, self.elements, self.constraints)
                offspring2 = apply_constraints(offspring2, self.elements, self.constraints)
            return offspring1, offspring2
        return parent1, parent2

    def mutate(self, individual, stepsize=1.0):
        logging.info("Mutating.")
        if np.random.rand() < self.mutation_rate:
            for _ in range(np.random.randint(1, len(self.elements) // 2 + 1)):
                point = np.random.randint(len(self.elements))
                individual[point] += np.random.uniform(0.01, stepsize)
                individual = np.clip(individual, a_min=0, a_max=1)
                individual /= np.sum(individual)
            if self.constraints:
                # Apply constraints in mole fraction
                individual = apply_constraints(individual, self.elements, self.constraints)
        individual = np.clip(individual, a_min=0, a_max=1)
        return individual

    def evolve(self):
        logging.info("Evolving with NSGA-II.")
        
        for generation in range(self.generations):
            logging.info(f"Generation {generation}")
            
            # Create offspring population
            offspring_population = []
            
            # Evaluate objectives for current population
            objective_values = [self.evaluate_objectives(ind) for ind in self.population]
            
            # Non-dominated sorting
            fronts = self.non_dominated_sort(objective_values)
            
            # Calculate crowding distances
            crowding_distances = self.calculate_crowding_distance(objective_values)
            
            # Selection
            selected_population = self.tournament_selection(
                self.population, objective_values, fronts, crowding_distances
            )
            
            # Make sure we have even number of individuals for pairing
            if len(selected_population) % 2 != 0:
                selected_population.pop()
                
            # Generate offspring
            for i in range(0, len(selected_population), 2):
                parent1, parent2 = selected_population[i], selected_population[i + 1]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                if self.constraints:
                    offspring1 = apply_constraints(offspring1, self.elements, self.constraints)
                    offspring2 = apply_constraints(offspring2, self.elements, self.constraints)
                    
                offspring_population.append(offspring1)
                offspring_population.append(offspring2)
                
            # Combine parent and offspring populations
            combined_population = self.population + offspring_population
            combined_objectives = [self.evaluate_objectives(ind) for ind in combined_population]
            
            # Environmental selection - select best individuals for next generation
            combined_fronts = self.non_dominated_sort(combined_objectives)
            new_population = []
            
            # Fill new population with fronts until we exceed population size
            front_index = 0
            while len(new_population) + len(combined_fronts[front_index]) <= self.population_size:
                # Add entire front
                for idx in combined_fronts[front_index]:
                    new_population.append(combined_population[idx])
                front_index += 1
                
                if front_index >= len(combined_fronts):
                    break
                    
            # If we haven't filled the population, sort the next front by crowding distance
            if len(new_population) < self.population_size and front_index < len(combined_fronts):
                front = combined_fronts[front_index]
                front_objectives = [combined_objectives[idx] for idx in front]
                front_distances = self.calculate_crowding_distance(front_objectives)
                
                # Sort by crowding distance (descending)
                sorted_indices = sorted(range(len(front)), key=lambda x: front_distances[x], reverse=True)
                
                # Add individuals until population is full
                for i in range(min(len(front), self.population_size - len(new_population))):
                    idx_in_front = sorted_indices[i]
                    idx_in_combined = front[idx_in_front]
                    new_population.append(combined_population[idx_in_combined])
                    
            self.population = new_population[:self.population_size]
            
            # Log information about the first front (best solutions)
            if fronts and len(fronts) > 0:
                first_front = fronts[0]
                first_front_objectives = [objective_values[i] for i in first_front]
                avg_objectives = np.mean(first_front_objectives, axis=0)
                logging.info(f"Generation {generation} - First front average objectives: {avg_objectives}")
                
        # Return the best solution from the final population (first front)
        final_objective_values = [self.evaluate_objectives(ind) for ind in self.population]
        final_fronts = self.non_dominated_sort(final_objective_values)
        
        if final_fronts and len(final_fronts) > 0 and len(final_fronts[0]) > 0:
            best_idx = final_fronts[0][0]  # First individual in first front
            best_individual = self.population[best_idx]
            best_objectives = final_objective_values[best_idx]
            return best_individual, best_objectives
        else:
            # Fallback if something went wrong
            best_individual = self.population[0]
            best_objectives = final_objective_values[0]
            return best_individual, best_objectives


def run_nsga(output, elements, init_mode, population_size, 
             constraints, get_density_mode, a, b, c, d, crossover_rate, mutation_rate, init_population):

    logging.basicConfig(filename=output, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("===----Starting NSGA-II----===")
    logging.info("Elements: %s", elements)
    logging.info(f"Constraints: {constraints}")

    if init_mode == "random":
        init_population = None

    nsga = NSGAII(
        elements=elements,
        population_size=population_size,
        generations=8000,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        init_population=init_population,
        constraints=constraints,
        a=a, b=b, c=c, d=d,
        get_density_mode=get_density_mode)

    best_individual, best_objectives = nsga.evolve()

    logging.info(f"Best Individual: {best_individual}, Best Objectives: {best_objectives}")
    print("Best Composition:", best_individual)
    print("Best Objectives:", best_objectives)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSGA-II for Element Optimization")
    parser.add_argument("-o", "--output", type=str, default="nsga_debug.log", help="Log filename (default: nsga.log)")
    parser.add_argument("-e", "--elements", type=str, default="Fe,Ni,Co,Cr,V,Cu,Al,Ti",
                        help="Comma-separated list of elements (default: predefined list)")
    parser.add_argument("-m", "--init_mode", type=str, default="random",
                        help="Choose between 'random' and 'init_population'")
    parser.add_argument("-p", "--population_size", type=int, default=10, help="Population size (default: 10)")
    parser.add_argument("-i", "--init_population", type=str, default=None, help="Initial population (default: None)")
    parser.add_argument("--constraints", type=str, default=None, help="Element-wise constraints (e.g., 'Fe<0.5, Al<0.1')")
    parser.add_argument("--get_density_mode", type=str, default="weighted_avg", help="Mode for density calculation (e.g. pred, relax, default: 'weighted_avg').")

    # Arguments for a, b, c, d
    parser.add_argument("--a", type=float, default=0.9, help="Weight for TEC mean (default: 0.9)")
    parser.add_argument("--b", type=float, default=0.1, help="Weight for TEC std (default: 0.1)")
    parser.add_argument("--c", type=float, default=0.9, help="Weight for density mean (default: 0.9)")
    parser.add_argument("--d", type=float, default=0.1, help="Weight for density std (default: 0.1)")
    
    # GA parameters
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="Crossover rate (default: 0.8)")
    parser.add_argument("--mutation_rate", type=float, default=0.3, help="Mutation rate (default: 0.3)")

    args = parser.parse_args()

    # Define initial population
    init_population_data = [
        [0.636, 0.286, 0.064, 0.014, 0.0, 0.0],
        [0.621, 0.286, 0.079, 0.014, 0.0, 0.0],
        [0.485, 0.2, 0.225, 0.0, 0.09, 0.0],
        [0.605, 0.3, 0.075, 0.0, 0.02, 0.0],
        [0.635, 0.3, 0.025, 0.0, 0.04, 0.0],
        [0.635, 0.305, 0.06, 0.0, 0.0, 0.0]
    ]

    params = {
        "output": args.output,
        "elements": args.elements.split(","),
        "init_mode": args.init_mode,
        "population_size": args.population_size,
        "constraints": parse_constraints(args.constraints),
        "get_density_mode": args.get_density_mode,
        "a": args.a,
        "b": args.b,
        "c": args.c,
        "d": args.d,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "init_population": init_population_data
    }

    run_nsga(**params)
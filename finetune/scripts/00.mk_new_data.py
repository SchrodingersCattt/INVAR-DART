import json
import os
import copy
import pandas as pd
import numpy as np
import pymatgen
from pymatgen.core.structure import Structure, Element, Lattice
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import dpdata
def mass_to_molar(mass_dict: dict):
    molar_composition = []
    for kk in mass_dict.keys():
        molar_composition.append(mass_dict[kk] / Element(kk).atomic_mass)
    return molar_composition

def normalize_composition(composition: list, total: int) -> list:
    """
    Normalize composition to a given total while maintaining integer values.
    Ensure that non-zero elements remain at least 1.
    """
    if not composition or total <= 0:
        print("Warning: Invalid input. Returning None.")
        return None

    total_composition = sum(composition)
    if total_composition == 0:
        print("Warning: Composition is all zeros. Returning None.")
        return None

    # 初步按比例分配 (浮点数)
    norm_composition_float = [c / total_composition * total for c in composition]
    norm_composition = [int(round(x)) for x in norm_composition_float]

    # 先强制非零的元素至少为1
    for i, c in enumerate(composition):
        if c > 0 and norm_composition[i] == 0:
            norm_composition[i] = 1

    # 重新检查总和是否超过或不足 total
    diff = sum(norm_composition) - total

    # 如果总和不对，需要修正（只改最大元素）
    while diff != 0:
        # 找到当前数量最多的元素
        max_idx = norm_composition.index(max(norm_composition))

        # diff > 0: 总数超了，要减
        if diff > 0:
            if norm_composition[max_idx] > 1:
                norm_composition[max_idx] -= 1
                diff -= 1
            else:
                # 如果最大元素已经=1，找下一个能减的
                candidates = [i for i, x in enumerate(norm_composition) if x > 1]
                if not candidates:
                    print("Warning: Can't normalize further, no elements > 1")
                    break
                max_idx = candidates[0]
                norm_composition[max_idx] -= 1
                diff -= 1

        # diff < 0: 总数不足，要加
        elif diff < 0:
            norm_composition[max_idx] += 1
            diff += 1

    if sum(norm_composition) != total:
        print(f"Warning: Normalization failed. Sum: {sum(norm_composition)}, Target: {total}")
        return None

    return norm_composition



class StructureWithProperties:
    def __init__(self, structure: pymatgen.core.Structure, properties, type_map):
        self.structure = structure
        self.properties = properties
        self.type_map = type_map
        self.dp_sys = dpdata.System(self.structure, fmt='pymatgen/structure', type_map=self.type_map)
        self.forces = np.zeros_like(self.dp_sys.data['coords'])  ## redundant
        self.dp_labeled_sys = dpdata.LabeledSystem()

    def add_prop(self):
        self.dp_sys.data['energies'] = np.array([self.properties])
        self.dp_sys.data['forces'] = np.array(self.forces)
        self.dp_labeled_sys = dpdata.LabeledSystem(data=self.dp_sys.data, type_map=self.type_map)
    
    def save_npy(self, output_path, add_prop=True):
        if add_prop:
            self.add_prop()
        self.dp_labeled_sys.to_deepmd_npy(output_path)
    
    def convert_to_npy(self, add_prop=True):
        if add_prop:
            self.add_prop()
        return self.dp_labeled_sys
    
    def trn_valid_split(self, train_ratio=0.8):
        raise NotImplementedError

def mk_template_supercell(packing: str):
    if "fcc" in packing:
        s = Structure.from_file("struct_template/fcc-Ni_mp-23_conventional_standard.cif")
        return s.make_supercell([5,5,5])
    elif "bcc" in packing:
        s = Structure.from_file("struct_template/bcc-V_mp-146_conventional_standard.cif")
        return s.make_supercell([6,6,6])
    elif "hcp" in packing:
        s = Structure.from_file("struct_template/hcp-Co_mp-54_conventional_standard.cif")
        return s.make_supercell([6,6,6])
    else:
        raise ValueError(f"{packing} not supported")


def get_mass_map(type_map, **kwargs):
    with open(atomic_mass_file, 'r') as ff:
        masses_data = json.load(ff)

    mass_map = []

    for tt in type_map:
        mass = masses_data[tt]
        mass_map.append(mass)
    if kwargs['verbose']:
        print('[' + ', '.join([f'"{s}"' for s in type_map]) + ']')
        print(mass_map)
        
    return mass_map

def save_dp(d, data, idx, mixed_type=True, type_map=None, **kwargs):
    packing = data.get_phase(d)
    supercell = mk_template_supercell(packing)
    pmg_elements = [Element(e) for e in type_map]
    molar_composition = data.get_molar_composition(d)
    
    tec = data.get_tec(d)
    atom_num = len(supercell)
    normalized_composition = normalize_composition(copy.deepcopy(molar_composition), atom_num)
    print(molar_composition, normalized_composition)
    
    if kwargs.get('normalize'):
        print(f"Normalizing {tec}")
        tec = z_core(tec)
        print(f"Normalized {tec}")

    if normalized_composition is None:
        print("Skipping row:", d)
        return

    MAX = 10
    if mixed_type:
        ms = dpdata.MultiSystems()

    for rand_seed in range(MAX):
        np.random.seed(rand_seed)
        replace_mapping = zip(pmg_elements, normalized_composition)
        atom_range = np.array(range(atom_num))
        selected_indices = []

        for ii, (element, num) in enumerate(replace_mapping):
            if num == 0:
                continue
            available_indices = np.setdiff1d(atom_range, selected_indices)

            if len(available_indices) < num:
                raise ValueError("Insufficient atoms to replace")

            chosen_idx = np.random.choice(available_indices, num, replace=False)
            selected_indices.extend(chosen_idx)
            for jj in chosen_idx:
                ss = supercell.replace(jj, element)

        file_name = "_".join(ss.formula.split())
        save_dir = f"new_data/{idx}_{file_name}_{packing}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ss_prop = StructureWithProperties(ss, tec, type_map=type_map)
        if mixed_type:
            ms.append(ss_prop.convert_to_npy())
        else:
            ss_prop.save_npy(save_dir, add_prop=True)

    if mixed_type:
        print(ms)
        ms.to_deepmd_npy_mixed(save_dir)



class JsonParser:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def iterate_rows(self):
        for entry in self.data:
            yield entry

    def get_molar_composition(self, entry):
        return [entry['composition'].get(e, 0.0) for e in type_map]

    def get_mass_composition(self, entry):
        return [entry['composition'].get(e, 0.0) for e in type_map]

    def get_phase(self, entry):
        return entry['phase']

    def get_tec(self, entry):
        return entry['TEC']

type_map = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc",
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv",
    "Ts", "Og"
]

if __name__ == "__main__":
    data_file = "./new_compositions.json"
    data = JsonParser(data_file)
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Compositions from {data_file} are **MOLAR**. Must have a double check.")

    
    normalize = False
    for idx, d in enumerate(data.iterate_rows()):
        save_dp(d, data, idx, type_map=type_map, normalize=normalize)
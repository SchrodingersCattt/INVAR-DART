from stat_distribution import ExcelParser
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def get_pareto_front(data_file, output=False):
    parser = ExcelParser(data_file)
    tec, density, Fe, Ni, Co, Cr, V, Cu, packing = extract_data(parser)

    points = np.column_stack((tec, density))

    pareto_indices = []
    for i, point in enumerate(points):
        if not is_dominated(point, np.delete(points, i, axis=0)):
            pareto_indices.append(i)
    
    # 获取初始帕累托前沿
    initial_pareto_front = points[pareto_indices]
    
    # 识别并去除异常值（TEC<3且density<8200）
    outlier_mask = (initial_pareto_front[:, 0] < 3) & (initial_pareto_front[:, 1] < 8200)
    non_outliers_mask = ~outlier_mask
    
    # 过滤掉异常值
    filtered_pareto_points = initial_pareto_front[non_outliers_mask]
    
    # 重新获取这些点在原始数据中的索引
    filtered_pareto_indices = [pareto_indices[i] for i in range(len(pareto_indices)) if non_outliers_mask[i]]
    
    # 获取异常值点的索引
    outlier_indices = [pareto_indices[i] for i in range(len(pareto_indices)) if outlier_mask[i]]
    
    if output:
        output_results(filtered_pareto_indices, Fe, Ni, Co, Cr, V, Cu, packing)
    
    return tec, density, filtered_pareto_points, filtered_pareto_indices, outlier_indices

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

def plot_results(tec, density, pareto_front, outlier_indices):
    plt.figure(figsize=(8, 5))
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = "Arial"
    plt.scatter(tec, density, label="Orig. data", alpha=0.5, c='#848484')
    
    # 绘制异常值点
    if len(outlier_indices) > 0:
        outlier_tec = tec[outlier_indices]
        outlier_density = density[outlier_indices]
        plt.scatter(outlier_tec, outlier_density, facecolors='none', edgecolors='#d62728', s=50, marker='o', label="Outliers", linewidths=1.5)
    
    # 已经去除了异常值，直接排序绘制帕累托前沿
    sorted_pareto_front = pareto_front[pareto_front[:, 0].argsort()]
    plt.plot(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1], 'r--s', label="Filtered Pareto front")
    
    # Plot ITER_0 data points
    # iter_0_pred_tec = ITER_0_pred[:, 0]
    # iter_0_pred_tec_std = ITER_0_pred[:, 1]
    # iter_0_pred_density = ITER_0_pred[:, 2]
    # plt.scatter(iter_0_pred_tec, iter_0_pred_density, marker="<", s=60, label="Iter.0 pred.", c='#ff7f0e')
    # plt.errorbar(iter_0_pred_tec, iter_0_pred_density, xerr=iter_0_pred_tec_std*STD_FACTOR, fmt='none', elinewidth=1, capsize=3, c='#ff7f0e')
    
    # iter_0_exp_tec = ITER_0_exp[:, 0]
    # iter_0_exp_density = ITER_0_exp[:, 1]
    # plt.scatter(iter_0_exp_tec, iter_0_exp_density, marker=">", s=60, label="Iter.0 real", c='#8b4513')

    # Plot ITER_1 data points
    # iter_1_pred_tec = ITER_1_pred[:, 0]
    # iter_1_pred_tec_std = ITER_1_pred[:, 1]
    # iter_1_pred_density = ITER_1_pred[:, 2]
    # plt.scatter(iter_1_pred_tec, iter_1_pred_density, marker="<", s=60, label="Iter.1 pred.", c='#2cf0ac')
    # plt.errorbar(iter_1_pred_tec, iter_1_pred_density, xerr=iter_1_pred_tec_std*STD_FACTOR, fmt='none', elinewidth=1, capsize=3, c='#9467bd')
    
    # iter_1_exp_tec = ITER_1_exp[:, 0]
    # iter_1_exp_density = ITER_1_exp[:, 1]
    # plt.scatter(iter_1_exp_tec, iter_1_exp_density, marker=">", s=60, label="Iter.1 real", c='#2ca02c')

    # Extract TEC, TEC std and density from iter_2_pred
    # iter_2_pred_tec = iter_2_pred[:, 0]
    # iter_2_pred_tec_std = iter_2_pred[:, 1]
    # iter_2_pred_density = iter_2_pred[:, 2]
    # plt.scatter(iter_2_pred_tec, iter_2_pred_density, marker="<", s=60, label="Iter.2 pred.", c='#3da4ff')
    # plt.errorbar(iter_2_pred_tec, iter_2_pred_density, xerr=iter_2_pred_tec_std*STD_FACTOR, fmt='none', elinewidth=1, capsize=3, c='#3da4ff')
    
    # Plot iter_2_real data points
    # iter_2_real_tec = iter_2_real[:, 0]
    # iter_2_real_density = iter_2_real[:, 2]
    # plt.scatter(iter_2_real_tec, iter_2_real_density, marker=">", s=60, label="Iter.2 real", c='#7f0eff')
    
    # Plot iter_3_pred data points
    iter_3_pred_tec = iter_3_pred[:, 0]
    iter_3_pred_tec_std = iter_3_pred[:, 1]
    iter_3_pred_density = iter_3_pred[:, 2]
    plt.scatter(iter_3_pred_tec, iter_3_pred_density, marker="<", s=20, label="Iter.3 pred.", c='#e377c2')
    plt.errorbar(iter_3_pred_tec, iter_3_pred_density, xerr=iter_3_pred_tec_std*STD_FACTOR, fmt='none', elinewidth=1, capsize=3, c='#e377c2')
    
    plt.text(0.05, 0.95, f"The error bars represent {STD_FACTOR}×std", transform=plt.gca().transAxes)
    plt.xlabel("TEC")
    plt.ylabel("Density")
    plt.ylim(8150, 8250)
    plt.legend(loc="lower center", bbox_to_anchor=(0.75, 0.5), ncol=1)
    plt.tight_layout()

    plt.savefig("pareto_front.png", dpi=300)
    plt.show()


# ITER_0_pred = np.array([
#     [1.1698, 0.43, 8077.52],
#     [3.2941, 0.47, 7931.9691],
#     [2.4756, 1.02, 8069.9608]
# ])
# ITER_0_exp = np.array([
#     [9.97, 8159.49],
#     [12.58, 8101.42],
#     [13.67, 8082.35]
# ])

# ITER_1_pred = np.array([
#     [4.22, 0.71, 8189.04],
#     [3.10, 1.15, 8199.56],
#     [2.83, 0.75, 8287.24]
# ])
# ITER_1_exp = np.array([
#     [6.47, 8176.63],
#     [3.4, 8170.91],
#     [1.98, 8278.05]
# ])

# iter_2_pred = np.array([
#     [8.59, 1.03, 8098.53],
#     [12.58, 1.89, 7942.21],
#     [12.64, 1.85, 7916.57]
# ])

# iter_2_real = np.array([
#     [9.38, 0, 8087.07],
#     [12.94, 0, 7909.68],
#     [12.89, 0, 7909.07]
#     # [14.43, 0, 7701.98], 
#     # [16.30, 0, 7916.57]
# ])

iter_3_pred = np.array([
    # [4.0586,0.6324,8181.3263],
    # [3.2954,1.4345,8184.4368],
    # [4.8947,1.3792,8196.0474],
    [4.2764,1.9237,8196.5375],
    # [4.1455,0.9253,8184.127],


    [3.2954,1.4345,8184.4368],
    [4.0225,1.4845,8187.9451]
])



if __name__ == "__main__":
    data_file = "dataset/Data_base_DFT_Thermal.xlsx"
    
    tec, density, pareto_front, pareto_indices, outlier_indices = get_pareto_front(data_file)    
    plot_results(tec, density, pareto_front, outlier_indices)
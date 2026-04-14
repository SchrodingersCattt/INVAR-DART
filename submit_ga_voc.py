import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# 固定参数
exp_name_base = f"iter03_GA_dpa3_1__"

conda_env_path = "/aisi-nas/guomingyu/conda_env/prop_finetuning"
nas_mount_path = "/aisi-nas"
nas_id = "anas-cnbje8a153ae6108"
docker_image = "vemlp-cn-beijing.cr.volces.com/preset-images/python:3.12-ubuntu22.04"
resource_queue_id = "q-20250618190306-vzjfq"  ## A800
# resource_queue_id = "q-20250210073532-srwxx"  ## H20

ws_path = Path("./")
cwd = os.getcwd()

abs_path = ws_path.resolve()

experiments = [
    {
        "exp_name": "FeNiCoCrVAl_Fe0.25_Cr0.1_Al0.05",
        "element": "Fe,Ni,Co,Cr,V,Al",
        "constraints": "Fe>0.25,Cr<0.1,V<0.2,Al<0.05"
    },
    {
        "exp_name": "FeNiCoCrVTi_Fe0.25_Cr0.1_Al0.05Ti0.05",
        "element": "Fe,Ni,Co,Cr,V,Al,Ti",
        "constraints": "Fe>0.25,Cr<0.1,V<0.2,Al<0.05,Ti<0.05"
    },
    {
        "exp_name": "FeNiCoCrVAl_Fe0.25_Cr0.1_Ti0.05",
        "element": "Fe,Ni,Co,Cr,V,Ti",
        "constraints": "Fe>0.25,,Cr<0.1,V<0.2,Ti<0.05"
    }
]

# experiments = [
#     {
#         "exp_name": "FeNi__test",
#         "element": "Fe,Ni,Co,Cr,V",
#         "constraints": "Fe>0.25,Cr=0,V=0"
#     }
# ]

shared_params = {
    "init_mode": "init",
    "population_size": 60,
    # "selection_mode": "tournament",
    "selection_mode": "roulette",
    "get_density_mode": "weighted_avg",
    "crossover_rate": 0.8,
    "mutation_rate": 0.3,
    "a": 0.1,
    "b": 0.2,
    "c": 0.3,
    "d": 0.0
}


for i, exp in enumerate(experiments):
    exp_name = exp["exp_name"]
    element = exp["element"]
    constraints = exp["constraints"]
    
    init_mode = shared_params["init_mode"]
    population_size = shared_params["population_size"]
    selection_mode = shared_params["selection_mode"]
    get_density_mode = shared_params["get_density_mode"]
    crossover_rate = shared_params["crossover_rate"]
    mutation_rate = shared_params["mutation_rate"]
    a = shared_params["a"]
    b = shared_params["b"]
    c = shared_params["c"]
    d = shared_params["d"]
    
    # 生成输出文件名
    output = f"$(date +'%Y%m%d_%H%M%S')_${{init_mode}}_{exp_name}_ga-${{selection_mode}}_cr${{crossover_rate}}_mr${{mutation_rate}}_a${{a}}_b${{b}}_c${{c}}_d${{d}}_fcc.log"
    
    yaml_path = abs_path / f"submit_{i+1}.yaml"
    pretty_exp_name = exp_name.replace(".", "p")
    lines = [
        f'TaskName: {exp_name_base}_{pretty_exp_name}',
        f'ResourceQueueId: "{resource_queue_id}"',
        f'Entrypoint: |',
        f'  export exp_name={exp_name}',
        f'  export element="{element}"',
        f'  export init_mode=\'{init_mode}\'',
        f'  export population_size={population_size}',
        f'  export constraints="{constraints}"',
        f'  export selection_mode=\'{selection_mode}\'',
        f'  export get_density_mode=\'{get_density_mode}\'',
        f'  export crossover_rate={crossover_rate}',
        f'  export mutation_rate={mutation_rate}',
        f'  export a={a}',
        f'  export b={b}',
        f'  export c={c}',
        f'  export d={d}',
        f'  export output={output}',
        f'  source /root/.bashrc',
        f'  source /root/miniconda3/bin/activate {conda_env_path}',
        f'  set -x',
        f'  cd {abs_path}',
        f'  python ga.py -o ${{output}} --elements ${{element}} --init_mode ${{init_mode}} --a ${{a}} --b ${{b}} --c ${{c}} --d ${{d}} --constraints ${{constraints}} --selection_mode ${{selection_mode}} --get_density_mode ${{get_density_mode}} --population_size ${{population_size}} --crossover_rate ${{crossover_rate}} --mutation_rate ${{mutation_rate}}',
        f'Framework: "PyTorchDDP"',
        f'TaskRoleSpecs:',
        f'  - RoleName: "worker"',
        f'    RoleReplicas: 1',
        f'    Flavor: ml.pni2l.3xlarge', # a800
        # f'    Flavor: ml.pni3ln.5xlarge',  ## h20
        f'ActiveDeadlineSeconds: 0',
        f'Storages:',
        f'  - Type: "Nas"',
        f'    MountPath: "{nas_mount_path}"',
        f'    NasId: "{nas_id}"',
        f'ImageUrl: "{docker_image}"',
        f'CacheType: "Cloudfs"',
        f'Priority: 4',
        f'RetryOptions:',
        f'  EnableRetry: false',
        f'DiagOptions:',
        f'  - Name: EnvironmentalDiagnosis',
        f'    Enable: false',
        f'  - Name: PythonDetection',
        f'    Enable: false',
        f'  - Name: LogDetection',
        f'    Enable: false',
        f'  - Name: PySpyDump',
        f'    Enable: false',
    ]

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))

    logging.info(f"YAML written to: {yaml_path}")

    os.chdir(os.path.dirname(yaml_path))

    submit_cmd = f"volc ml_task submit -c submit_{i+1}.yaml --resource_queue_id {resource_queue_id}"
    logging.info(f"Submitting task {i+1} for {exp_name}")
    os.system(submit_cmd)

os.chdir(cwd)

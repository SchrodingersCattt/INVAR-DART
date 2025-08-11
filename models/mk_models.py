import os
import glob
import shutil


TEC_MODELS = glob.glob("../../../iter01.finetune/dpa_v3_1/workspace*/model.ckpt.pt")
# DENSITY_MODELS = glob.glob("/mnt/data_nas/guomingyu/PROPERTIES_PREDICTION/INVAR_density_only/mae_finetune_crossValidation/workspace*/model.ckpt.pt") 

def collect_models(path_list, model_prefix):
    print(path_list)
    for pp in path_list:
        print(pp)
        idx = pp.split("workspace")[-1].split("/")[0]
        scr = pp
        des = f"./{model_prefix}_{idx}.pt"
        shutil.copy(scr, des)


if __name__ == "__main__":
    collect_models(TEC_MODELS, "tec")
    # collect_models(DENSITY_MODELS, "density")

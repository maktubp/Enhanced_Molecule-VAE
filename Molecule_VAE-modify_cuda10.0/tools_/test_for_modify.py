import os
import json
import numpy as np
import pandas as pd

# 可选：强制CPU，避免CUDA环境干扰
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu
from chemvae import hyperparameters


def smoke_test_model(model_dir, smiles_list=None):
    """
    最小样本验证模型是否正常运行：
    1) 能加载模型和参数
    2) 能将SMILES转one-hot
    3) 能encode/decode
    4) 能predict property
    5) 输出结果有限值、维度合理
    """
    if smiles_list is None:
        smiles_list = [
            "CCO",          # 乙醇
            "c1ccccc1",     # 苯
            "CC(=O)O",      # 乙酸
            "CCN",          # 乙胺
            "O=C=O"         # 二氧化碳
        ]

    print("=" * 80)
    print("Model dir:", model_dir)

    # 1) 读取参数
    exp_file = os.path.join(model_dir, "exp.json")
    assert os.path.exists(exp_file), f"找不到 exp.json: {exp_file}"
    params = hyperparameters.load_params(exp_file, verbose=False)

    # 2) 加载VAE工具（含encoder/decoder/property predictor）
    vae = VAEUtils(directory=model_dir)

    # 3) 规范化SMILES并转one-hot
    canon_smiles = [mu.canon_smiles(s) for s in smiles_list]
    X = vae.smiles_to_hot(canon_smiles, canonize_smiles=False)
    print("Input SMILES:", canon_smiles)
    print("X shape:", X.shape)

    # 4) encode/decode
    z = vae.encode(X)
    X_rec = vae.decode(z)
    rec_smiles = vae.hot_to_smiles(X_rec, strip=True)

    print("z shape:", z.shape)
    print("X_rec shape:", X_rec.shape)
    print("Reconstruction:")
    for s_in, s_out in zip(canon_smiles, rec_smiles):
        print(f"  {s_in:20s} -> {s_out}")

    # 5) 性质预测
    y_pred = vae.predict_prop_Z(z)  # 默认会反归一化（若配置了norm_data.csv）
    y_pred = np.array(y_pred)
    print("y_pred shape:", y_pred.shape)

    # 6) 基本健壮性检查
    assert X.shape[0] == len(smiles_list), "样本数不一致"
    assert z.shape[0] == len(smiles_list), "编码样本数不一致"
    assert X_rec.shape[0] == len(smiles_list), "解码样本数不一致"
    assert np.isfinite(z).all(), "z中出现NaN/Inf"
    assert np.isfinite(X_rec).all(), "X_rec中出现NaN/Inf"
    assert np.isfinite(y_pred).all(), "预测值中出现NaN/Inf"

    # 7) 打印预测值（带任务名）
    reg_tasks = params.get("reg_prop_tasks", [])
    if len(reg_tasks) > 0 and y_pred.ndim == 2 and y_pred.shape[1] == len(reg_tasks):
        print("Predictions:")
        for i, s in enumerate(canon_smiles):
            vals = {task: float(y_pred[i, j]) for j, task in enumerate(reg_tasks)}
            print(f"  {s:20s} -> {vals}")
    else:
        print("Predictions:", y_pred)

    print("SMOKE TEST PASSED ✅")
    print("=" * 80)


if __name__ == "__main__":
    # 改成你的模型目录，例如:
    # model_dir = r"D:\python code\Molecule_VAE-main\models\train_5"
    model_dir = r"models/train_5"

    smoke_test_model(model_dir=model_dir)
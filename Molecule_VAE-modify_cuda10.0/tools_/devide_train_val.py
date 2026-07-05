import pandas as pd

# 路径设置
input_csv = r"D:\python database\Molecule_VAE\train_5\zinc_smiles_PSA.csv"   # 总数据
train_csv = r"D:\python database\Molecule_VAE\train_5\train.csv"
val_csv   = r"D:\python database\Molecule_VAE\train_5\val.csv"
test_csv   = r"D:\python database\Molecule_VAE\train_5\test.csv"

# 读取数据
df = pd.read_csv(input_csv)

# 只取前两列，并统一列名
df = df.iloc[:, :2].copy()
df.columns = ["smiles", "PSA"]

# 随机打乱并按 7:2:1 切分
# random_state 固定42
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

n = len(df)
n_train = int(n * 0.7)
n_val   = int(n * 0.2)
# 剩余全部给 test，保证总数不丢
n_test  = n - n_train - n_val

train_df = df.iloc[:n_train].copy()
val_df   = df.iloc[n_train:n_train + n_val].copy()
test_df  = df.iloc[n_train + n_val:].copy()

# 保存
train_df.to_csv(train_csv, index=False, encoding="utf-8")
val_df.to_csv(val_csv, index=False, encoding="utf-8")
test_df.to_csv(test_csv, index=False, encoding="utf-8")

print(f"总样本数: {n}")
print(f"train: {len(train_df)} -> {train_csv}")
print(f"val  : {len(val_df)} -> {val_csv}")
print(f"test : {len(test_df)} -> {test_csv}")
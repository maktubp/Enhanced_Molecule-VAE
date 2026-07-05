import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# 读取数据
df = pd.read_csv(r"D:\python database\Molecule_VAE\zinc_smiles.csv")  # 改成你的路径

# 定义计算PSA函数
def calc_psa(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Descriptors.TPSA(mol)
    except:
        return None

# 计算PSA
psa_list = df["smiles"].apply(calc_psa)

# 插入到第7列（index=6）
df.insert(1, "PSA", psa_list)

# 保存
df.to_csv(r"D:\python database\Molecule_VAE\zinc_smiles_PSA.csv", index=False)

print("已添加PSA并保存")
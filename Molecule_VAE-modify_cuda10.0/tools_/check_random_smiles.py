from rdkit import Chem
import random


def random_like_smiles_by_root(mol, n_random=1):
    # 旧版 RDKit 兼容：随机选择 rootedAtAtom，按规定次数生成
    n_atoms = mol.GetNumAtoms()
    smiles_set = set()

    for _ in range(n_random):
        root_idx = random.randrange(n_atoms)
        s = Chem.MolToSmiles(
            mol,
            isomericSmiles=False,
            kekuleSmiles=False,
            rootedAtAtom=root_idx,
            canonical=False,
            allBondsExplicit=False,
            allHsExplicit=False
        )
        smiles_set.add(s)

    return smiles_set


def check_one(smiles: str, n_random=1):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[ERROR] 无法解析: {smiles}")
        return

    can_ref = Chem.MolToSmiles(mol, canonical=True)

    print("=" * 70)
    print(f"原始输入: {smiles}")
    print(f"Canonical参考: {can_ref}")
    print(f"规定生成次数: {n_random}")
    print("-" * 70)

    variants = random_like_smiles_by_root(mol, n_random=n_random)
    print(f"实际生成写法(去重后)数量: {len(variants)}")

    all_same = True
    for s in sorted(variants):
        m2 = Chem.MolFromSmiles(s)
        can2 = Chem.MolToSmiles(m2, canonical=True) if m2 is not None else None
        same = (can2 == can_ref)
        all_same = all_same and same
        print(f"{s:25s} -> canonical: {str(can2):20s} | 一致: {same}")

    print("-" * 70)
    print(f"结论: 所有写法canonical是否一致 -> {all_same}")
    print()


if __name__ == "__main__":
    check_one("O=C1OCCO1", n_random=4)         # EC
    check_one("CCO", n_random=4)         # 乙醇
    check_one("c1ccccc1", n_random=4)    # 苯
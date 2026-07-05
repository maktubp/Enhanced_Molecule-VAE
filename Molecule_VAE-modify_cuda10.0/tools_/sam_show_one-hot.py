import numpy as np
import matplotlib.pyplot as plt
import json

# ===================== 1. 加载字符映射（无版本依赖） =====================
CHARS = ["7", "6", "o", "]", "3", "s", "(", "-", "S", "/",
         "B", "4", "[", ")", "#", "I", "l", "O", "H", "c",
         "1", "@", "=", "n", "P", "8", "C", "2", "F", "5",
         "r", "N", "+", "\\", " "]
CHAR_INDICES = {c: i for i, c in enumerate(CHARS)}
INDICES_CHAR = {i: c for i, c in enumerate(CHARS)}

# ===================== 2. SMILES转One-Hot矩阵（适配numpy1.12.1） =====================
def smiles_to_one_hot(smiles, max_len=120, char_indices=CHAR_INDICES):
    n_chars = len(char_indices)
    one_hot_matrix = np.zeros((max_len, n_chars), dtype=np.int8)
    smiles_padded = smiles.ljust(max_len)[:max_len]
    for pos, char in enumerate(smiles_padded):
        if char in char_indices:
            idx = char_indices[char]
            one_hot_matrix[pos, idx] = 1
        else:
            raise ValueError(f"SMILES包含未识别字符：{char}，请检查zinc_char.json")
    return one_hot_matrix

# ===================== 3. 生成苯环One-Hot矩阵 =====================
benzene_smiles = "C(=O)O"  #CC1=CC=CC=C1 甲苯 C1=CC=CC=C1 苯基 C(=O)O 羧基
max_len_vis = 20  # 可视化前20位，避免渲染过宽
one_hot_benzene = smiles_to_one_hot(benzene_smiles, max_len=max_len_vis)

# ===================== 4. 可视化One-Hot矩阵（完全适配matplotlib2.0.2） =====================
# 中文显示配置（2.0.2兼容）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布（移除pad参数，2.0.2不支持）
fig, ax = plt.subplots(figsize=(16, 8))

# 绘制热力图（原生pcolormesh，2.0.2核心支持）
im = ax.pcolormesh(
    one_hot_benzene,
    cmap="Reds",
    edgecolors="gray",
    linewidths=0.5,
    vmin=0, vmax=1
)

# ===================== 轴标签设置（适配2.0.2） =====================
# X轴：索引+字符
x_ticks = np.arange(len(CHARS))
x_labels = [f"{i}\n({CHARS[i]})" for i in x_ticks]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontsize=7)  # 减小字体避免重叠

# Y轴：位置+字符
y_ticks = np.arange(max_len_vis)
y_labels = []
for i in y_ticks:
    if i < len(benzene_smiles):
        y_labels.append(f"{i}\n({benzene_smiles[i]})")
    else:
        y_labels.append(f"{i}\n( )")
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=8)

# 突出苯环核心字符列（2.0.2支持axvline）
ax.axvline(x=26, color="blue", linestyle="--", linewidth=2, label="C (索引26)")
ax.axvline(x=20, color="green", linestyle="--", linewidth=2, label="1 (索引20)")
ax.axvline(x=22, color="orange", linestyle="--", linewidth=2, label="= (索引22)")

# ===================== 标题设置（关键：移除pad参数，2.0.2不支持） =====================
ax.set_title(
    f"苯环SMILES「{benzene_smiles}」One-Hot矩阵可视化（前{max_len_vis}位）",
    fontsize=14, fontweight='bold'  # 仅保留支持的参数，移除pad=20
)
# 替代pad的效果：调整标题与画布的间距（2.0.2支持）
ax.title.set_y(1.05)  # 向上偏移标题，等效pad的作用

# 轴标签（2.0.2支持labelpad）
ax.set_xlabel("字符索引 + 字符", fontsize=18, labelpad=10)
ax.set_ylabel("SMILES位置索引 + 对应字符", fontsize=18, labelpad=10)

# ===================== 颜色条与图例（2.0.2兼容） =====================
# 颜色条
cbar = fig.colorbar(im, ax=ax)
#cbar.set_label("One-Hot值 (1=字符存在，0=字符不存在)", fontsize=10)

# 图例
ax.legend(loc="upper right", fontsize=18)

# 调整布局（2.0.2支持tight_layout）
plt.tight_layout()

# 保存并显示（2.0.2兼容）
plt.savefig("benzene_one_hot_vis_legacy.png", dpi=300, bbox_inches="tight")
plt.show()

# ===================== 验证结果 =====================
print("=== 苯环SMILES字符→索引映射验证 ===")
for char in benzene_smiles:
    print(f"字符「{char}」→ 索引：{CHAR_INDICES[char]}")

print("\n=== One-Hot矩阵核心位置验证 ===")
for pos in range(len(benzene_smiles)):
    char = benzene_smiles[pos]
    idx = CHAR_INDICES[char]
    print(f"位置{pos}（字符{char}）：One-Hot矩阵第{pos}行，第{idx}列为1")
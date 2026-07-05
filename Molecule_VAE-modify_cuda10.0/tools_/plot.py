"""绘制 train 实验的 history.csv / RMSE_file.csv 曲线。

用法（在项目根目录执行）:
  python tools_/plot.py -e models/train_5/exp_stage1.json
  python tools_/plot.py --history models/train_5/history_nogru.csv --rmse models/train_5/RMSE_file_nogru.csv
  python tools_/plot.py -e models/train_5/exp_stage1.json --save-dir models/train_5/plots
"""
import argparse
import json
import os

import pandas as pd
from matplotlib import pyplot as plt


def plot_hist(hist, name, start=0, end=None, save_dir=None):
    val_col = "val_" + name
    if name not in hist.columns or val_col not in hist.columns:
        print("跳过 {}: 缺少 {} 或 {}".format(name, name, val_col))
        return

    plt.figure()
    if end is None:
        plt.plot(hist[name][start:], label="train")
        plt.plot(hist[val_col][start:], label="val")
    else:
        plt.plot(hist[name][start:end], label="train")
        plt.plot(hist[val_col][start:end], label="val")
    plt.title(name)
    plt.xlabel("epoch")
    plt.legend(loc="upper left")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "{}.png".format(name))
        plt.savefig(out_path, dpi=600)
        print("已保存:", out_path)
    return plt.gcf()


def plot_rmse(rmse_df, save_dir=None):
    rmse_cols = [c for c in rmse_df.columns if c.startswith("RMSE_")]
    mae_cols = [c for c in rmse_df.columns if c.startswith("MAE_")]
    if not rmse_cols:
        return

    plt.figure()
    for col in rmse_cols:
        plt.plot(rmse_df[col], label=col)
    plt.title("RMSE (denormalized, val set)")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "RMSE_val.png")
        plt.savefig(path, dpi=600)
        print("已保存:", path)

    if mae_cols:
        plt.figure()
        for col in mae_cols:
            plt.plot(rmse_df[col], label=col)
        plt.title("MAE (denormalized, val set)")
        plt.xlabel("epoch")
        plt.legend(loc="upper left")
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "MAE_val.png")
            plt.savefig(path, dpi=600)
            print("已保存:", path)


def load_paths_from_exp(exp_file):
    with open(exp_file, "r", encoding="utf-8") as f:
        params = json.load(f)
    history = params.get("history_file")
    rmse = params.get("RMSE_file")
    if not history:
        raise ValueError("exp 文件中未找到 history_file")
    return history, rmse


def main():
    parser = argparse.ArgumentParser(description="绘制 history / RMSE 训练曲线")
    parser.add_argument("-e", "--exp-file", help="exp_stage1.json 等配置文件")
    parser.add_argument("--history", help="history.csv 路径")
    parser.add_argument("--rmse", help="RMSE_file.csv 路径")
    parser.add_argument("--start", type=int, default=0, help="起始 epoch 索引")
    parser.add_argument("--end", type=int, default=None, help="结束 epoch 索引（不含）")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="保存 png 的目录；不指定则只弹窗显示",
    )
    parser.add_argument("--no-show", action="store_true", help="不弹出 matplotlib 窗口")
    args = parser.parse_args()

    if args.exp_file:
        history_path, rmse_path = load_paths_from_exp(args.exp_file)
    else:
        history_path = args.history
        rmse_path = args.rmse

    if not history_path:
        parser.error("请指定 -e/--exp-file 或 --history")

    history_path = os.path.normpath(history_path)
    if not os.path.isabs(history_path):
        history_path = os.path.join(os.getcwd(), history_path)

    print("读取 history:", history_path)
    hist = pd.read_csv(history_path)

    metrics = [
        "x_pred_categorical_accuracy",
        "x_pred_loss",
        "z_mean_log_var_loss",
        "reg_prop_pred_loss",
        "loss",
    ]
    for metric in metrics:
        plot_hist(hist, metric, start=args.start, end=args.end, save_dir=args.save_dir)

    if rmse_path:
        rmse_path = os.path.normpath(rmse_path)
        if not os.path.isabs(rmse_path):
            rmse_path = os.path.join(os.getcwd(), rmse_path)
        if os.path.exists(rmse_path):
            print("读取 RMSE:", rmse_path)
            rmse_df = pd.read_csv(rmse_path)
            plot_rmse(rmse_df, save_dir=args.save_dir)
        else:
            print("RMSE 文件不存在，跳过:", rmse_path)

    if not args.no_show and not args.save_dir:
        plt.show()
    elif not args.no_show and args.save_dir:
        plt.show()


if __name__ == "__main__":
    main()

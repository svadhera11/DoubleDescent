#!/usr/bin/env python3
import glob
import os
import re
import argparse
import matplotlib.pyplot as plt

BASE_LOG_DIR = "logs"
RESULT_DIR = "result_plots"

# ---------- filename parsing helpers ----------

FNAME_RE = re.compile(
    r"k(?P<k>\d+)_noise(?P<noise>\d+)_"
    r"(?P<dataset>[^_]+)_"          # e.g. cifar10 or mnist
    r"(?P<optim>optim[^_]+(?:_mom)?)"  # e.g. optimadam or optimsgd_mom
    r"_lossce\.log$"
)

def pretty_run_name(path: str) -> str:
    base = os.path.basename(path.rstrip("/"))
    if base.endswith((".log", ".txt")):
        base = os.path.basename(os.path.dirname(path))
    name_lower = base.lower()

    # Dataset
    if "cifar" in name_lower:
        dataset = "CIFAR-10"
    elif "mnist" in name_lower:
        dataset = "MNIST"
    else:
        dataset = "Unknown"
    model = "ResNet-18"

    m = re.search(r"(?:noise|p)[_=-]?([0-9.]+)", name_lower)
    if m:
        noise = m.group(1)
        noise_str = f"label noise p={noise}"
    else:
        noise_str = "no label noise"

    return f"{model}, {dataset}, {noise_str}"


def parse_filename(path: str):
    """
    Parse k, noise, dataset, optimizer from filenames like:
      k1_noise20_cifar10_optimadam_lossce.log
      k2_noise20_mnist_optimsgd_mom_lossce.log
    """
    fname = os.path.basename(path)
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Could not parse config from filename: {fname}")

    k = int(m.group("k"))
    noise = m.group("noise")        
    dataset = m.group("dataset")      
    optim_code = m.group("optim")     

    optim = optim_code.replace("optim", "")  
    return k, noise, dataset, optim

def parse_log(path: str):

    k_from_name, _, _, _ = parse_filename(path)

    best_test_loss = None
    best_train_loss = None
    best_epoch = None


    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/\d+:\s+train_loss=([0-9.]+),.*test_loss=([0-9.]+)"
    )

    with open(path, "r") as f:
        for line in f:
            if "train_loss=" not in line or "test_loss=" not in line:
                continue
            m = epoch_pattern.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            train_loss = float(m.group(2))
            test_loss = float(m.group(3))

            if best_test_loss is None or test_loss < best_test_loss:
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch

    if best_test_loss is None:
        raise ValueError(f"No epoch lines with losses found in {path}")

    return k_from_name, best_train_loss, best_test_loss, best_epoch


def collect_results(log_dir: str):
    """
    Walk all logs in `log_dir` and build:

        results[(dataset, optim, noise)] = list of (k, train_loss, test_loss)
    """
    pattern = os.path.join(log_dir, "k*_noise*_*_lossce.log")
    log_paths = glob.glob(pattern)
    if not log_paths:
        raise SystemExit(f"No logs found for pattern: {pattern}")

    results = {} 

    for path in log_paths:
        try:
            k_name, noise, dataset, optim = parse_filename(path)
            k, tr, te, ep = parse_log(path)

            if k != k_name:
                print(f"[WARN] k mismatch for {path}: {k_name} (name) vs {k} (log)")

        except ValueError as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        key = (dataset, optim, noise)
        results.setdefault(key, []).append((k, tr, te))

        print(f"{os.path.basename(path)} -> "
              f"dataset={dataset}, optim={optim}, noise={noise}%, "
              f"k={k}, best_epoch={ep}, train={tr:.4f}, test={te:.4f}")

  
    for key, triples in results.items():
        triples.sort(key=lambda x: x[0])

    return results


def plot_config(dataset: str, optim: str, noise: str,
                triples, config_subdir: str):
    ks = [t[0] for t in triples]
    train_losses = [t[1] for t in triples]
    test_losses = [t[2] for t in triples]

    os.makedirs(RESULT_DIR, exist_ok=True)

    plt.figure(figsize=(7, 5))


    plt.plot(ks, train_losses, marker="o", label="Train loss (at best test)")
    plt.plot(ks, test_losses, marker="o", label="Best test loss")

    plt.xlabel("ResNet18 Width Parameter")
    plt.ylabel("Loss")

    optim_title = optim.replace("_", "+") 
    plt.title(
        f"ResNet18: {dataset}, {optim_title}, {noise}% label noise "
        f"({config_subdir})"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_name = f"resnet18_width_vs_loss_{dataset}_{optim}_noise{noise}_{config_subdir}.png"
    out_path = os.path.join(RESULT_DIR, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-subdir",
        default="config_final_1",
        help="Subdirectory under logs/ that contains resnet18/ (default: config_final)",
    )
    args = parser.parse_args()

    log_dir = os.path.join(BASE_LOG_DIR, args.config_subdir, "resnet18")
    if not os.path.isdir(log_dir):
        raise SystemExit(f"Log directory not found: {log_dir}")

    results = collect_results(log_dir)

    for (dataset, optim, noise), triples in sorted(
        results.items(), key=lambda x: (x[0][0], x[0][1], int(x[0][2]))
    ):
        plot_config(dataset, optim, noise, triples, args.config_subdir)

if __name__ == "__main__":
    main()

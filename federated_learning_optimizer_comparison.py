"""
=============================================================================
Decentralized Federated Learning with Optimizer Comparison
Fog-Cloud Architecture | EfficientNet-B0 | RSSCN7 Dataset
=============================================================================

Optimizers compared:
  - GWO  : Grey Wolf Optimizer  (Code 7)
  - PSO  : Particle Swarm Optimization (Code 8)
  - ABC  : Artificial Bee Colony (Code 9)

Architecture:
  10 drones → 2 Fog Brokers (Router A: drones 1-5, Router B: drones 6-10)
             → Cloud aggregation via FedAvg

Usage:
  python federated_learning_optimizer_comparison.py --optimizer gwo
  python federated_learning_optimizer_comparison.py --optimizer pso
  python federated_learning_optimizer_comparison.py --optimizer abc

Author: Sakshi Saraswat
Research Intern, NIT Delhi (June–July 2025)
=============================================================================
"""

import os
import zipfile
import random
import time
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_DRONES    = 10
NUM_ROUNDS    = 10
LOCAL_EPOCHS  = 5
BATCH_SIZE    = 32
BANDWIDTH_MBPS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPU_RATE       = 0.5
MEMORY_RATE    = 0.1
BANDWIDTH_RATE = 0.05

ALPHA = 1.0   # fitness weight: accuracy
BETA  = 0.05  # fitness weight: latency

random.seed(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────
# STEP 1: DATASET
# ─────────────────────────────────────────────
def load_dataset():
    """Download RSSCN7 from Kaggle, split into train/test, partition across drones."""
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    kaggle_json_path = os.path.join(os.getcwd(), 'kaggle.json')

    if os.path.exists(kaggle_json_path):
        import shutil
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        shutil.copy(kaggle_json_path, os.path.expanduser("~/.kaggle/kaggle.json"))
        os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    else:
        raise FileNotFoundError("Place your kaggle.json in the current directory.")

    os.system("kaggle datasets download -d yangpeng1995/rsscn7 -p ./data")

    with zipfile.ZipFile("./data/rsscn7.zip", 'r') as z:
        z.extractall("./data/rsscn7")

    dataset_path = "./data/rsscn7/RSSCN7-master"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

    test_size  = 500
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Split train indices equally across drones
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    split_size = len(train_indices) // NUM_DRONES
    splits = [train_indices[i * split_size:(i + 1) * split_size] for i in range(NUM_DRONES)]

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return full_dataset, splits, test_loader


# ─────────────────────────────────────────────
# STEP 2: MODEL
# ─────────────────────────────────────────────
def create_model():
    """EfficientNet-B0 pretrained, all layers fine-tuned, 7-class output."""
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = True
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    return model.to(DEVICE)


# ─────────────────────────────────────────────
# STEP 3: ROUTER + VM (Edge Preprocessing)
# ─────────────────────────────────────────────
class Router:
    """Filters images by mean pixel intensity before local training."""
    def __init__(self, threshold=0.10):
        self.threshold = threshold

    def inspect(self, image_tensor):
        return "yes" if image_tensor.mean().item() > self.threshold else "no"


class VM:
    """Runs inference on filtered images for pre-validation."""
    def __init__(self):
        self.model = create_model()

    def process(self, image, label):
        self.model.eval()
        with torch.no_grad():
            output = self.model(image.unsqueeze(0).to(DEVICE))
            _, pred = torch.max(output, 1)
        return pred.item(), label


# ─────────────────────────────────────────────
# STEP 4: DRONE CLASS
# ─────────────────────────────────────────────
class Drone:
    def __init__(self, drone_id, indices, full_dataset, test_loader):
        self.id          = drone_id
        self.test_loader = test_loader
        self.dataset     = Subset(full_dataset, indices)
        self.loader      = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.model       = create_model()
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = optim.Adam(self.model.parameters(), lr=0.001)
        self.images_used = 0
        print(f"Drone {drone_id + 1} initialized.")

    def local_train(self, epochs):
        router = Router()
        vm     = VM()
        self.model.train()
        total_loss, steps, used = 0, 0, 0

        for _ in range(epochs):
            for data, targets in self.loader:
                for i in range(len(data)):
                    image, label = data[i], targets[i]
                    if router.inspect(image) == "yes":
                        vm.process(image, label)
                        image = image.unsqueeze(0).to(DEVICE)
                        label = torch.tensor([label]).to(DEVICE)
                        self.optimizer.zero_grad()
                        output = self.model(image)
                        loss   = self.criterion(output, label)
                        loss.backward()
                        self.optimizer.step()
                        total_loss += loss.item()
                        steps += 1
                        used  += 1

        self.images_used = used
        return total_loss / steps if steps else 0

    def get_state(self):
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_state(self, state):
        self.model.load_state_dict(state)

    def evaluate(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(data)
                _, pred = torch.max(outputs, 1)
                correct += (pred == targets).sum().item()
                total   += targets.size(0)
        return 100.0 * correct / total


# ─────────────────────────────────────────────
# STEP 5: FEDAVG + CLOUD
# ─────────────────────────────────────────────
def fedavg(states):
    new_state = {}
    for k in states[0]:
        new_state[k] = sum(s[k] for s in states) / len(states)
    return new_state


class Cloud:
    def __init__(self):
        self.model_state = None

    def receive_model(self, state):
        size_mb = sum(p.numel() * 4 for p in state.values()) / 1e6
        latency = (size_mb / BANDWIDTH_MBPS) * np.random.normal(1.0, 0.05)
        print(f"  Cloud: Received model ({size_mb:.2f} MB, {latency * 1000:.2f} ms)")
        time.sleep(latency)
        self.model_state = state.copy()
        return latency

    def distribute_model(self):
        return self.model_state.copy()

    def set_model(self, state):
        self.model_state = state.copy()


# ─────────────────────────────────────────────
# STEP 6: FOG BROKER
# ─────────────────────────────────────────────
class FogBroker:
    """Filters drone models into VALID / REPROCESS / DROP queues based on model size."""
    def __init__(self, threshold_mb=16.5):
        self.threshold_mb     = threshold_mb
        self.accepted_history = []   # accepted count per round
        self.total_history    = []   # total submitted per round
        self.fitness_history  = []   # avg fitness per round

    def filter_models(self, drone_states, drone_ids, drone_delays, drone_accuracies):
        valid, reprocess, drop             = [], [], []
        valid_ids, reprocess_ids, drop_ids = [], [], []
        fitness_this_round                 = []

        max_delay = max(drone_delays) if max(drone_delays) > 0 else 1.0

        print("\n  FogBroker Filtering:")
        for state, delay, acc, drone_id in zip(drone_states, drone_delays, drone_accuracies, drone_ids):
            size_mb = sum(p.numel() * 4 for p in state.values()) / 1e6 * np.random.uniform(0.95, 1.05)
            fitness = ALPHA * (1 - acc / 100.0) + BETA * (delay / max_delay)

            if size_mb <= self.threshold_mb:
                valid.append({"drone_id": drone_id, "state": state,
                              "accuracy": acc, "total_delay": delay})
                valid_ids.append(drone_id)
                fitness_this_round.append(fitness)
                print(f"  Drone {drone_id+1} → VALID     | {size_mb:.2f} MB | Acc: {acc:.2f}% | Delay: {delay*1000:.2f} ms")
            elif size_mb <= self.threshold_mb * 1.2:
                reprocess.append(state)
                reprocess_ids.append(drone_id)
                print(f"  Drone {drone_id+1} → REPROCESS | {size_mb:.2f} MB")
            else:
                drop.append(state)
                drop_ids.append(drone_id)
                print(f"  Drone {drone_id+1} → DROPPED   | {size_mb:.2f} MB")

        self.accepted_history.append(len(valid))
        self.total_history.append(len(drone_states))
        self.fitness_history.append(np.mean(fitness_this_round) if fitness_this_round else 0)

        return valid, reprocess, drop, valid_ids, reprocess_ids, drop_ids

    def acceptance_rate(self):
        if not self.total_history or self.total_history[-1] == 0:
            return 0
        return self.accepted_history[-1] / self.total_history[-1]

    def avg_fitness(self):
        return self.fitness_history[-1] if self.fitness_history else 0


# ─────────────────────────────────────────────
# STEP 7: OPTIMIZATION ALGORITHMS
# ─────────────────────────────────────────────

def _fitness(info, max_delay):
    """Shared fitness function: minimize (1-accuracy) + weighted latency."""
    return ALPHA * (1 - info['accuracy'] / 100.0) + BETA * (info['total_delay'] / max_delay)


def gwo_select_models(valid_info, num_select=3):
    """
    Grey Wolf Optimizer (GWO) inspired model selection.
    Alpha (best), Beta (2nd best), Delta (3rd best) wolves guide the pack.
    Models are ranked by fitness; top-3 act as Alpha/Beta/Delta.
    """
    if len(valid_info) <= num_select:
        return valid_info

    max_delay = max(info['total_delay'] for info in valid_info)
    sorted_info = sorted(valid_info, key=lambda x: _fitness(x, max_delay))

    alpha_wolf = sorted_info[0]
    beta_wolf  = sorted_info[1] if len(sorted_info) > 1 else alpha_wolf
    delta_wolf = sorted_info[2] if len(sorted_info) > 2 else beta_wolf

    print(f"  GWO — Alpha: Drone {alpha_wolf['drone_id']+1} "
          f"| Beta: Drone {beta_wolf['drone_id']+1} "
          f"| Delta: Drone {delta_wolf['drone_id']+1}")

    return [alpha_wolf, beta_wolf, delta_wolf]


def pso_select_models(valid_info, num_select=3, num_iterations=5):
    """
    Particle Swarm Optimization (PSO) inspired model selection.
    Each particle represents a candidate subset. Personal and global
    best positions are updated across iterations based on fitness.
    """
    if len(valid_info) <= num_select:
        return valid_info

    max_delay = max(info['total_delay'] for info in valid_info)
    n = len(valid_info)

    # Initialize particles as random subsets
    particles      = [random.sample(range(n), num_select) for _ in range(n)]
    personal_best  = [p[:] for p in particles]
    pb_fitness     = [float('inf')] * n
    global_best    = None
    gb_fitness     = float('inf')

    for _ in range(num_iterations):
        for i, particle in enumerate(particles):
            selected = [valid_info[j] for j in particle]
            fit = np.mean([_fitness(s, max_delay) for s in selected])

            if fit < pb_fitness[i]:
                personal_best[i] = particle[:]
                pb_fitness[i]    = fit

            if fit < gb_fitness:
                global_best = particle[:]
                gb_fitness  = fit

        # Move particles toward personal and global best
        for i in range(n):
            new_particle = list(set(
                personal_best[i][:num_select // 2] +
                (global_best[:num_select // 2] if global_best else []) +
                random.sample(range(n), num_select)
            ))[:num_select]
            if len(new_particle) == num_select:
                particles[i] = new_particle

    selected = [valid_info[j] for j in global_best] if global_best else valid_info[:num_select]
    print(f"  PSO — Selected: {[s['drone_id']+1 for s in selected]}")
    return selected


def abc_select_models(valid_info, num_select=3, max_iter=5):
    """
    Artificial Bee Colony (ABC) inspired model selection.
    Employed bees exploit known solutions; onlooker bees select
    based on fitness probability; scout bees explore random solutions.
    """
    if len(valid_info) <= num_select:
        return valid_info

    max_delay = max(info['total_delay'] for info in valid_info)

    def fitness_fn(info):
        return _fitness(info, max_delay)

    population   = valid_info[:]
    best_solution = min(population, key=fitness_fn)

    for _ in range(max_iter):
        # Employed bee phase: explore neighbor solutions
        new_population = []
        for info in population:
            partner = random.choice(population)
            new_population.append(partner if fitness_fn(partner) < fitness_fn(info) else info)
        population = new_population

        # Onlooker bee phase: update global best
        candidate = min(population, key=fitness_fn)
        if fitness_fn(candidate) < fitness_fn(best_solution):
            best_solution = candidate

    # Scout bee phase: select top solutions by fitness
    sorted_pop = sorted(population, key=fitness_fn)
    selected   = sorted_pop[:num_select]
    print(f"  ABC — Selected: {[s['drone_id']+1 for s in selected]}")
    return selected


OPTIMIZERS = {
    "gwo": gwo_select_models,
    "pso": pso_select_models,
    "abc": abc_select_models,
}


# ─────────────────────────────────────────────
# STEP 8: FEDERATED TRAINING LOOP
# ─────────────────────────────────────────────
def run_federated(optimizer_name="gwo"):
    assert optimizer_name in OPTIMIZERS, f"Choose from: {list(OPTIMIZERS.keys())}"
    select_fn = OPTIMIZERS[optimizer_name]

    print(f"\n{'='*60}")
    print(f"  Federated Learning | Optimizer: {optimizer_name.upper()}")
    print(f"  Device: {DEVICE} | Drones: {NUM_DRONES} | Rounds: {NUM_ROUNDS}")
    print(f"{'='*60}\n")

    full_dataset, splits, test_loader = load_dataset()

    drones = [Drone(i, splits[i], full_dataset, test_loader) for i in range(NUM_DRONES)]
    cloud  = Cloud()
    logs, global_accs = [], []

    router_A_ids = list(range(5))
    router_B_ids = list(range(5, 10))

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─'*50}")
        print(f"  ROUND {rnd}/{NUM_ROUNDS}")
        print(f"{'─'*50}")

        local_states  = [None] * NUM_DRONES
        losses        = [None] * NUM_DRONES
        model_infos   = []
        round_latency = 0.0

        # Local training on each drone
        for i, drone in enumerate(drones):
            print(f"\n  Drone {i+1} training...")
            loss      = drone.local_train(LOCAL_EPOCHS)
            state     = drone.get_state()
            acc       = drone.evaluate()
            exec_time = np.random.uniform(0.4, 0.6)
            size_mb   = sum(p.numel() * 4 for p in state.values()) / 1e6 * np.random.uniform(0.95, 1.05)
            tx_delay  = (size_mb / BANDWIDTH_MBPS) * np.random.normal(1.0, 0.05)
            total_delay = exec_time + tx_delay
            round_latency += total_delay
            cost = CPU_RATE * exec_time + (MEMORY_RATE + BANDWIDTH_RATE) * size_mb

            local_states[i] = state
            losses[i]       = loss
            model_infos.append({
                "drone_id":    i,
                "state":       state,
                "accuracy":    acc,
                "total_delay": total_delay,
                "cost":        cost,
            })
            print(f"  → Loss: {loss:.4f} | Acc: {acc:.2f}% | Delay: {total_delay*1000:.2f} ms")

        # Fog filtering (separate brokers per router, created once per round)
        fogA = FogBroker()
        fogB = FogBroker()

        valid_A, _, _, valid_ids_A, _, _ = fogA.filter_models(
            [local_states[i] for i in router_A_ids], router_A_ids,
            [model_infos[i]["total_delay"] for i in router_A_ids],
            [model_infos[i]["accuracy"]    for i in router_A_ids],
        )
        valid_B, _, _, valid_ids_B, _, _ = fogB.filter_models(
            [local_states[i] for i in router_B_ids], router_B_ids,
            [model_infos[i]["total_delay"] for i in router_B_ids],
            [model_infos[i]["accuracy"]    for i in router_B_ids],
        )

        # Optimizer-based model selection
        print(f"\n  [{optimizer_name.upper()}] Selecting top 3 from Router A:")
        top_A = select_fn(valid_A) if valid_A else []
        print(f"\n  [{optimizer_name.upper()}] Selecting top 3 from Router B:")
        top_B = select_fn(valid_B) if valid_B else []

        agg_A = fedavg([info["state"] for info in top_A]) if top_A else None
        agg_B = fedavg([info["state"] for info in top_B]) if top_B else None

        if agg_A and agg_B:
            print("\n  Cloud aggregating...")
            agg_cloud     = fedavg([agg_A, agg_B])
            cloud_latency = cloud.receive_model(agg_cloud)
            round_latency += cloud_latency
            cloud.set_model(agg_cloud)
            updated_state = cloud.distribute_model()

            for drone in drones:
                drone.set_state(updated_state)

            # Evaluate global model
            acc = drones[4].evaluate()
            global_accs.append(acc)
            print(f"\n  ✓ Global Accuracy after Round {rnd}: {acc:.2f}%")

            avg_cost = np.mean([info["cost"] for info in model_infos])
            logs.append([
                rnd, acc,
                round_latency * 1000, cloud_latency * 1000,
                avg_cost,
                fogA.acceptance_rate(), fogB.acceptance_rate(),
                fogA.avg_fitness(),     fogB.avg_fitness(),
            ] + losses)
        else:
            print("  ✗ Aggregation skipped — no valid models in a router.")

    return logs, global_accs, optimizer_name


# ─────────────────────────────────────────────
# STEP 9: SAVE LOGS
# ─────────────────────────────────────────────
def save_logs(logs, optimizer_name):
    os.makedirs("logs", exist_ok=True)
    path = f"logs/fed_{optimizer_name}_logs.csv"
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Round", "Global Accuracy (%)", "Total Latency (ms)", "Cloud Latency (ms)",
            "Avg Cost", "Acceptance A", "Acceptance B", "Fitness A", "Fitness B"
        ] + [f"Drone {i+1} Loss" for i in range(NUM_DRONES)])
        writer.writerows(logs)
    print(f"\n  Logs saved → {path}")
    return path


# ─────────────────────────────────────────────
# STEP 10: PLOT
# ─────────────────────────────────────────────
def plot_results(logs, global_accs, optimizer_name):
    os.makedirs("results", exist_ok=True)
    rounds = list(range(1, len(logs) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"Federated Learning — {optimizer_name.upper()} Optimizer", fontsize=14)

    axes[0].plot(rounds, global_accs, marker='o', color='green')
    axes[0].set_title("Global Accuracy per Round")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True)

    total_lat = [row[2] for row in logs]
    axes[1].plot(rounds, total_lat, marker='x', color='blue')
    axes[1].set_title("Total Latency per Round")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].grid(True)

    cloud_lat = [row[3] for row in logs]
    axes[2].plot(rounds, cloud_lat, marker='s', color='red')
    axes[2].set_title("Cloud Upload Latency per Round")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Latency (ms)")
    axes[2].grid(True)

    plt.tight_layout()
    path = f"results/{optimizer_name}_metrics.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Plot saved → {path}")

    # Acceptance + Fitness plots
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle(f"{optimizer_name.upper()} — Fog Broker Stats", fontsize=13)

    accept_A = [row[5] for row in logs]
    accept_B = [row[6] for row in logs]
    axes2[0].plot(rounds, accept_A, marker='o', label='Router A')
    axes2[0].plot(rounds, accept_B, marker='s', label='Router B')
    axes2[0].set_title("Model Acceptance Rate")
    axes2[0].set_ylabel("Rate")
    axes2[0].legend()
    axes2[0].grid(True)

    fit_A = [row[7] for row in logs]
    fit_B = [row[8] for row in logs]
    axes2[1].plot(rounds, fit_A, marker='x', color='darkgreen', label='Router A')
    axes2[1].plot(rounds, fit_B, marker='^', color='crimson',   label='Router B')
    axes2[1].set_title("Average Fitness Score")
    axes2[1].set_ylabel("Fitness")
    axes2[1].legend()
    axes2[1].grid(True)

    plt.tight_layout()
    path2 = f"results/{optimizer_name}_fog_stats.png"
    plt.savefig(path2, dpi=150)
    plt.show()
    print(f"  Plot saved → {path2}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Optimizer Comparison")
    parser.add_argument("--optimizer", type=str, default="gwo",
                        choices=["gwo", "pso", "abc"],
                        help="Optimizer to use: gwo | pso | abc")
    args = parser.parse_args()

    logs, global_accs, opt_name = run_federated(optimizer_name=args.optimizer)
    save_logs(logs, opt_name)
    plot_results(logs, global_accs, opt_name)

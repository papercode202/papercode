import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from statistics import mean
from trajectory_utils import prediction_output_to_trajectories
import trajectory_utils, matrix_utils, os_utils
from matplotlib import pyplot as plt
from tqdm import tqdm

# Headless plotting
import matplotlib
matplotlib.use('Agg')

sys.path.append("trajectron")
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron

# ----------------------------- SETUP ----------------------------- #
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--checkpoint", type=int)
parser.add_argument("--model2", type=str)
parser.add_argument("--checkpoint2", type=int)
parser.add_argument("--model3", type=str)
parser.add_argument("--checkpoint3", type=int)
parser.add_argument("--epsilon", type=str)
parser.add_argument("--data", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--output_tag", type=str)
parser.add_argument("--node_type", type=str)
parser.add_argument("--prediction_horizon", nargs='+', type=int, default=None)
args = parser.parse_args()

# ----------------------------- PLOTTING ----------------------------- #
import matplotlib.patches as patches
import math

def plot_trajectories(trajectories, largest_cluster_indices, gt_traj=None, fused_traj=None, t=None, node_id=None):
    plt.figure(figsize=(6, 6))

    def plot_with_arrows(traj, style, color, label, zorder=1):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], style, color=color, linewidth=2.5, alpha=0.9, label=label, zorder=zorder)
        for i in range(0, len(traj) - 1, max(1, len(traj) // 5)):
            dx = traj[i+1, 0] - traj[i, 0]
            dy = traj[i+1, 1] - traj[i, 1]
            plt.arrow(traj[i, 0], traj[i, 1], dx, dy,
                      head_width=0.3, head_length=0.5, fc=color, ec=color, alpha=0.7, zorder=zorder)

    colors = ['tab:green', 'tab:orange', 'tab:red']
    for idx, traj in enumerate(trajectories):
        label = f"Model {idx} (Clustered)" if idx in largest_cluster_indices else f"Model {idx} (Outlier)"
        color = colors[idx % len(colors)]
        style = '-' if idx in largest_cluster_indices else '--'
        plot_with_arrows(traj, style, color, label, zorder=1 if idx in largest_cluster_indices else 0)

    if gt_traj is not None:
        plot_with_arrows(gt_traj, '--', 'black', 'Ground Truth', zorder=3)




    if fused_traj is not None:
        plot_with_arrows(fused_traj, '-', 'blue', 'Fused (Cluster Avg)', zorder=4)

    plt.title(f"Trajectory Hypotheses for One Vehicle\nTime: {t}, Vehicle ID: {node_id}", fontsize=10)
    plt.axis('equal')
    plt.axis('off')
    plt.legend(loc='upper right', fontsize=8, frameon=True)
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    fname = f"plots/vehicle_traj_cluster_t{t}_node{node_id}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    



# ----------------------------- METRICS ----------------------------- #
def compute_matrix_distance(trajectories):
    """
    Computes the pairwise average Euclidean distance over time between all trajectories.

    Args:
        trajectories: List or array of shape (n, T, 2), where n is the number of trajectories,
                      T is the number of time steps, and 2 is (x, y) position.

    Returns:
        matrix: (n x n) symmetric matrix of average distances between trajectories.
    """
    n = len(trajectories)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Compute Euclidean distance between trajectories[i] and trajectories[j] at each timestep
            diff = trajectories[i] - trajectories[j]  # shape: (T, 2)
            dist = np.linalg.norm(diff, axis=1)       # shape: (T,)
            avg_dist = np.mean(dist)                  # scalar

            matrix[i][j] = avg_dist
            matrix[j][i] = avg_dist  # symmetric

    return matrix

def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    return np.mean(error, axis=-1).flatten()

def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()



# ----------------------------- SERVER FUNCTION ----------------------------- #

def sendtoserver(prediction_output_dict, prediction_output_dict2, prediction_output_dict3,
                 dt, max_hl, ph, node_type_enum, map=None, prune_ph_to_future=False,num=0):

    prediction_dict, _, futures_dict = prediction_output_to_trajectories(
        prediction_output_dict, dt, max_hl, ph, prune_ph_to_future=prune_ph_to_future, map=map
    )
    prediction_dict2, _, _ = prediction_output_to_trajectories(
        prediction_output_dict2, dt, max_hl, ph, prune_ph_to_future=prune_ph_to_future, map=map
    )
    prediction_dict3, _, _ = prediction_output_to_trajectories(
        prediction_output_dict3, dt, max_hl, ph, prune_ph_to_future=prune_ph_to_future, map=map
    )

    batch_error_dict = {node_type: {'ade': [], 'fde': [], 'mr': []} for node_type in node_type_enum}
    mr = []

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            if node.type == "VEHICLE":
                trajectories = [
                    prediction_dict[t][node][0][0],
                    prediction_dict2[t][node][0][0],
                    prediction_dict3[t][node][0][0]
                ]
                x = compute_matrix_distance(trajectories)
                print(x)
                clusterin = DBSCAN(eps=float(args.epsilon), min_samples=1, metric='precomputed').fit(x)
                labels, counts = np.unique(clusterin.labels_, return_counts=True)
                max_idx = np.argmax(counts)
                largest_cluster_indices = np.where(clusterin.labels_ == labels[max_idx])[0]

                # Prepare fused trajectory
                m = 0
                k = 0
                if 0 in largest_cluster_indices:
                    m += prediction_dict[t][node]
                    k += 1
                if 1 in largest_cluster_indices:
                    m += prediction_dict2[t][node]
                    k += 1
                if 2 in largest_cluster_indices:
                    m += prediction_dict3[t][node]
                    k += 1
                fused_traj = (m / k)[0][0] if k > 0 else None

                # Plot all
              
                if  num<10 :
                
                    plot_trajectories(
                        trajectories,
                        largest_cluster_indices,
                        gt_traj=futures_dict[t][node],
                        fused_traj=fused_traj,
                        t=t,
                        node_id=node.id
                    )
                    num=num+1
                
                # Evaluation
                if fused_traj is not None:
                    merged = m / k
                    ade = compute_ade(merged, futures_dict[t][node])
                    fde = compute_fde(merged, futures_dict[t][node])

                    batch_error_dict[node.type]['ade'].extend(ade.tolist())
                    batch_error_dict[node.type]['fde'].extend(fde.tolist())
                   

    return batch_error_dict, mr

# ----------------------------- LOAD MODEL ----------------------------- #
def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams

# ----------------------------- MAIN ----------------------------- #
if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)
    eval_stg2, hyperparams2 = load_model(args.model2, env, ts=args.checkpoint2)
    eval_stg3, hyperparams3 = load_model(args.model3, env, ts=args.checkpoint3)

    if 'override_attention_radius' in hyperparams:
        for override in hyperparams['override_attention_radius']:
            n1, n2, radius = override.split(' ')
            env.attention_radius[(n1, n2)] = float(radius)

    scenes = env.scenes
    for scene in tqdm(scenes, desc="Preparing Node Graph"):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    mi = np.array([])

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']
        ade_all = np.array([])
        fde_all = np.array([])

        with torch.no_grad():
            for scene in tqdm(scenes, desc=f"Evaluating PH={ph}"):
                timesteps = np.arange(scene.timesteps)

                preds1 = eval_stg.predict(scene, timesteps, ph, num_samples=1,
                                          min_future_timesteps=8, z_mode=True,
                                          gmm_mode=True, full_dist=False)
                preds2 = eval_stg2.predict(scene, timesteps, ph, num_samples=1,
                                           min_future_timesteps=8, z_mode=True,
                                           gmm_mode=True, full_dist=False)
                preds3 = eval_stg3.predict(scene, timesteps, ph, num_samples=1,
                                           min_future_timesteps=8, z_mode=True,
                                           gmm_mode=True, full_dist=False)

                batch_error_dict, misses = sendtoserver(preds1, preds2, preds3,
                                                        scene.dt, max_hl, ph,
                                                        env.NodeType, map=None,
                                                        prune_ph_to_future=False)

                ade_all = np.hstack((ade_all, batch_error_dict[args.node_type]['ade']))
                fde_all = np.hstack((fde_all, batch_error_dict[args.node_type]['fde']))
            


            print("### ADE ###", np.mean(ade_all))
            print("### FDE ###", np.mean(fde_all))

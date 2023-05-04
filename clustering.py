import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from statistics import mean
from scipy.ndimage import binary_dilation
from trajectory_utils import prediction_output_to_trajectories
import trajectory_utils,matrix_utils,os_utils
from matplotlib import pyplot as plt
from tqdm import tqdm
sys.path.append("trajectron")
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--model2", help="model full path", type=str)
parser.add_argument("--checkpoint2", help="model checkpoint to evaluate", type=int)
parser.add_argument("--model3", help="model full path", type=str)
parser.add_argument("--checkpoint3", help="model checkpoint to evaluate", type=int)
parser.add_argument("--epsilon", help="epsilone value", type=str)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int, default=None)
args = parser.parse_args()



def compute_matrix_distance(trajectories):
    n=len(trajectories)
    num_points = len(trajectories[0])
    matrix = np.zeros((n,n))
    for i in range(num_points):
        point_distances = []
        for j in range(n):
            for k in range(j+1, n):
                point_distances.append(np.linalg.norm(trajectories[j][i] - trajectories[k][i]))
        point_error = sum(point_distances) / len(point_distances)
        for j in range(n):
            for k in range(j+1, n):
                matrix[j][k] += point_error
                matrix[k][j] += point_error
    matrix /= num_points
    print(matrix)
    return matrix
def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()
def compute_mr(predicted_trajs, gt_traj):
    misses = 0

    dist = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    if(dist > 2):
        misses=1
    print(len(predicted_trajs[:, :, -1]))
    print(misses)
     
    
    return misses

def sendtoserver(prediction_output_dict,
                             prediction_output_dict2,prediction_output_dict3,
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=True,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)
    (prediction_dict2,
     _,
     futures_dict2) = prediction_output_to_trajectories(prediction_output_dict2,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)
    (prediction_dict3,
     _,
     futures_dict2) = prediction_output_to_trajectories(prediction_output_dict3,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(), 'mr': list()}
    i=0

    i=0
    k=0
    numbertraj = 0
    crlist = []
    mr=[]
    coll = 0
    n = 6
    preds = np.ones((n, 2))

    # Define the X and Y offsets
    offsetX = 1
    offsetY = 1

    for t in prediction_dict.keys():
        '''
        print("all nodes")
        print(preds)
        
        # Add the X and Y offsets to each position in each trajectory
        trajectories = [preds[i:i+n] for i in range(n, (k+1)*n, n)]
        
        print("all nodes mfessla")
        print(trajectories)
        numbertraj = len(trajectories) * len(trajectories)
        coll=0
        for i in range(len(trajectories)):
            for j in range(len(trajectories)):
                if i == j:
                    continue # skip comparison of the same trajectory
                
                for m in range(min(len(trajectories[i]), len(trajectories[j]))):
                    # Define the corners of the rectangles for the m-th segment of each trajectory
                    rect1 = np.array([trajectories[i][m] - [offsetX, offsetY], trajectories[i][m] + [offsetX, offsetY]])
                    rect2 = np.array([trajectories[j][m] - [offsetX, offsetY], trajectories[j][m] + [offsetX, offsetY]])
                    
                    # Check if the rectangles overlap
                    if not (rect1[1][0] < rect2[0][0] or rect1[0][0] > rect2[1][0] or rect1[1][1] < rect2[0][1] or rect1[0][1] > rect2[1][1]):
                        print(f"Trajectories {i+1} and {j+1} collide at segment {m+1}!")
                        coll += 1
                    else:
                        print(f"Trajectories {i+1} and {j+1} do not collide at segment {m+1}.")
        
        if numbertraj != 0:
            crlist.append(coll / numbertraj)
        
        preds = np.ones((n, 2))
        print(k)
        k = 0
        '''           ''' if(node.type=='VEHICLE'):

                preds = np.vstack((preds, prediction_dict3[t][node][0][0]))
                print("preds node")
                print(prediction_dict[t][node][0][0])
                print("j")
                k = k + 1
                print(k)
            '''
        for node in prediction_dict[t].keys():
           

            if(node.type=="VEHICLE"):

            
                trajectories=[prediction_dict[t][node][0][0], prediction_dict2[t][node][0][0],prediction_dict3[t][node][0][0]]
                
                x=compute_matrix_distance(trajectories)
                
                clusterin = DBSCAN(eps=float(args.epsilon), min_samples=1,metric='precomputed').fit(x)
                
                
                
                labels, counts = np.unique(clusterin.labels_, return_counts=True)

                # Find the index of the cluster with the most elements
                max_idx = np.argmax(counts)

                # Get the indices of the points in the largest cluster
                largest_cluster_indices = np.where(clusterin.labels_ == labels[max_idx])[0]
                
                # Print the indices of the largest cluster
                print("Indices of largest cluster:", largest_cluster_indices)
                m=0
                k=0
                if(0 in largest_cluster_indices):
                    k=k+1 
                    m=m+prediction_dict[t][node]
                if(1 in largest_cluster_indices):
                    k=k+1 
                    m=m+prediction_dict2[t][node]
                if(2 in largest_cluster_indices):
                    k=k+1 
                    m=m+prediction_dict3[t][node]


                ade_errors = compute_ade(m/k, futures_dict[t][node])
                mi = compute_mr(m/k, futures_dict[t][node])
                fde_errors = compute_fde(m/k, futures_dict[t][node])
                
                mr.append(mi)
                
                batch_error_dict[node.type]['ade'].extend(list(ade_errors))
                batch_error_dict[node.type]['fde'].extend(list(fde_errors))
                batch_error_dict[node.type]['mr'].extend(list(mr))
                         
                

    
    return batch_error_dict,mr


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)
    eval_stg2, hyperparams2 = load_model(args.model2, env, ts=args.checkpoint2)
    eval_stg3, hyperparams3 = load_model(args.model3, env, ts=args.checkpoint3)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])
    crlist=[]
    mi=np.array([])
    from statistics import mean
    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            ############### test with 3 models ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])

            print("####clustering####")
            for scene in tqdm(scenes):
                timesteps = np.arange(scene.timesteps)

                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=1,
                                               min_future_timesteps=8,
                                               z_mode=True,
                                               gmm_mode=True,
                                               full_dist=False)
                  # This will trigger grid sampling
                predictions2 = eval_stg2.predict(scene,
                                            timesteps,
                                            ph,
                                            num_samples=1,
                                            min_future_timesteps=8,
                                            z_mode=True,
                                            gmm_mode=True,
                                            full_dist=False)
                # This will trigger grid sampling
                predictions3 = eval_stg3.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=1,
                                               min_future_timesteps=8,
                                               z_mode=True,
                                               gmm_mode=True,
                                               full_dist=False)
                  # This will trigger grid sampling
             

                
                batch_error_dict,a = sendtoserver(predictions,
                                                predictions2,
                                                predictions3,
                                                scene.dt,
                                                max_hl=max_hl,
                                                ph=ph,
                                                node_type_enum=env.NodeType,
                                                map=None,
                                                prune_ph_to_future=False,
                                                kde=False)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                eval_mr_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['mr']))
                mi=np.hstack((mi,a))
            print(len(eval_mr_batch_errors))
            print(np.count_nonzero(mi == 1))
            print("###ADE error###")
            print(np.mean(eval_ade_batch_errors) )
            print("###FDE error###")
            print(np.mean(eval_fde_batch_errors))
            


            
            pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_ade_most_likely_z.csv'))
            pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_fde_most_likely_z.csv'))
           



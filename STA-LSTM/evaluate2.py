from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedMSETest
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 5
args['grid_size'] = (13,3)
args['input_embedding_size'] = 32
args['train_flag'] = False


# Evaluation metric:
metric = 'rmse'

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('sta_lstm_0.tar'))
if args['use_cuda']:
    net = net.cuda()

# tsSet = ngsimDataset('./data/sta_lstm/TestSet.mat')
tsSet = ngsimDataset('../data/TestSet.mat')
tsDataloader = DataLoader(tsSet, batch_size=128, shuffle=True, num_workers=8, collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(5).cuda()
counts = torch.zeros(5).cuda()
lossVal = 0
count = 0

vehid = []
pred_x = []
pred_y = []
gt_x = []  # Ground truth x coordinates
gt_y = []  # Ground truth y coordinates
T = []
dsID = []
ts_cen = []
ts_nbr = []
wt_ha = []


for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, veh_id, t, ds = data

    if not isinstance(hist, list):  # nbrs are not zeros
        vehid.append(veh_id)  # current vehicle to predict
        T.append(t)  # current time
        dsID.append(ds)
    
        # Initialize Variables
        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

        fut_pred, weight_ts_center, weight_ts_nbr, weight_ha = net(hist, nbrs, mask, lat_enc, lon_enc)
        l, c = maskedMSETest(fut_pred, fut, op_mask)

        # Store predicted trajectories
        fut_pred_x = fut_pred[:, :, 0].detach().cpu().numpy()
        fut_pred_y = fut_pred[:, :, 1].detach().cpu().numpy()
        pred_x.append(fut_pred_x)
        pred_y.append(fut_pred_y)
        
        # Store ground truth trajectories
        future_x = fut[:, :, 0].detach().cpu().numpy()
        future_y = fut[:, :, 1].detach().cpu().numpy()
        gt_x.append(future_x)
        gt_y.append(future_y)

        ts_cen.append(weight_ts_center[:, :, 0].detach().cpu().numpy())
        ts_nbr.append(weight_ts_nbr[:, :, 0].detach().cpu().numpy())
        wt_ha.append(weight_ha[:, :, 0].detach().cpu().numpy())

        lossVal += l.detach()
        count += c.detach()

    # Limit to first few batches for visualization (otherwise too many)
    if i >= 2:  # Adjust this number to control how many batches to process
        break

print('lossVal is:', lossVal)
print(f"RMSE: {torch.pow(lossVal / count, 0.5) * 0.3048} meters")  # Calculate RMSE and convert from feet to meters

# Visualization code
def visualize_trajectories(gt_x, gt_y, pred_x, pred_y, num_scenarios=2):
    """
    Visualize vehicle trajectories showing ground truth vs predictions
    
    Args:
        gt_x, gt_y: Ground truth coordinates
        pred_x, pred_y: Predicted coordinates
        num_scenarios: Number of scenarios to visualize
    """
    # Flatten the batched data for easier processing
    all_gt_x = np.concatenate(gt_x)
    all_gt_y = np.concatenate(gt_y)
    all_pred_x = np.concatenate(pred_x)
    all_pred_y = np.concatenate(pred_y)
    
    # Create a figure with subplots for multiple traffic scenarios
    fig, axes = plt.subplots(num_scenarios, 1, figsize=(12, 6*num_scenarios))
    if num_scenarios == 1:
        axes = [axes]
    
    scenarios_per_plot = len(all_gt_x) // num_scenarios
    
    for i in range(num_scenarios):
        ax = axes[i]
        start_idx = i * scenarios_per_plot
        end_idx = min((i + 1) * scenarios_per_plot, len(all_gt_x))
        
        # Set up the plot area
        ax.set_facecolor('lightgray')
        ax.grid(False)
        
        # Draw lane markings (assuming highway scenario with 3 lanes)
        lane_width = 12  # feet (typical US highway lane)
        road_width = 3 * lane_width
        
        # Draw solid white edge lines
        ax.axhline(y=0, color='white', linestyle='-', linewidth=2)
        ax.axhline(y=road_width, color='white', linestyle='-', linewidth=2)
        
        # Draw dashed lane dividers
        for lane in range(1, 3):
            ax.axhline(y=lane * lane_width, color='white', linestyle='--', linewidth=1)
        
        # Plot each vehicle trajectory in this batch
        for j in range(start_idx, end_idx):
            # Get the car image for start position
            car_length = 14  # feet
            car_width = 6    # feet
            
            # Plot the ground truth trajectory
            ax.scatter(all_gt_x[j], all_gt_y[j], c='green', marker='.', label='Ground Truth' if j == start_idx else "")
            
            # Plot the predicted trajectory
            ax.scatter(all_pred_x[j], all_pred_y[j], c='blue', marker='^', label='Prediction' if j == start_idx else "")
            
            # Add a car silhouette at the beginning of the trajectory
            car_x = all_gt_x[j][0]
            car_y = all_gt_y[j][0]
            rect = patches.Rectangle((car_x - car_length/2, car_y - car_width/2), car_length, car_width, 
                                    linewidth=1, edgecolor='black', facecolor='gray')
            ax.add_patch(rect)
        
        # Set axis labels and title
        ax.set_xlabel('Longitudinal Position (feet)')
        ax.set_ylabel('Lateral Position (feet)')
        ax.set_title(f'Vehicle Trajectory Prediction - Scenario {i+1}')
        
        # Set reasonable axis limits
        min_x = min(np.min(all_gt_x[start_idx:end_idx]), np.min(all_pred_x[start_idx:end_idx])) - 50
        max_x = max(np.max(all_gt_x[start_idx:end_idx]), np.max(all_pred_x[start_idx:end_idx])) + 50
        min_y = -10
        max_y = road_width + 10
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        # Add legend
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('trajectory_predictions.png', dpi=300)
    plt.show()

# Call visualization function
visualize_trajectories(gt_x, gt_y, pred_x, pred_y, num_scenarios=2)
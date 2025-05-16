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
hist_x = []  # Historical x coordinates
hist_y = []  # Historical y coordinates


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

        # Store historical trajectory data
        hist_data_x = hist[:, :, 0].detach().cpu().numpy()
        hist_data_y = hist[:, :, 1].detach().cpu().numpy()
        hist_x.append(hist_data_x)
        hist_y.append(hist_data_y)

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

    # Just process one batch for visualization
    if i >= 0:  # Only process the first batch
        break

print('lossVal is:', lossVal)
print(f"RMSE: {torch.pow(lossVal / count, 0.5) * 0.3048} meters")  # Calculate RMSE and convert from feet to meters

# Function to visualize multiple vehicle trajectories
def visualize_multiple_vehicles(hist_x, hist_y, gt_x, gt_y, pred_x, pred_y, num_vehicles=10, split_view=True):
    """
    Visualizes multiple vehicle trajectories on a highway scene similar to the provided image.
    
    Args:
        hist_x, hist_y: Lists of arrays containing historical x,y coordinates
        gt_x, gt_y: Lists of arrays containing ground truth x,y coordinates
        pred_x, pred_y: Lists of arrays containing predicted x,y coordinates
        num_vehicles: Number of vehicles to visualize
        split_view: Whether to split the visualization into two panels
    """
    if split_view:
        # Create two subplots vertically stacked
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        axes = [ax1, ax2]
        vehicles_per_plot = num_vehicles // 2
    else:
        # Create a single plot
        fig, ax = plt.subplots(figsize=(12, 6))
        axes = [ax]
        vehicles_per_plot = num_vehicles
    
    # Set background color to light gray for all axes
    for ax in axes:
        ax.set_facecolor('lightgray')
        ax.grid(False)
        
        # Remove axes text and ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Set aspect ratio to equal for more realistic proportions
        ax.set_aspect('equal', adjustable='box')
        
        # Hide axes borders
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Highway parameters
    lane_width = 12  # feet (typical US highway lane)
    road_width = 3 * lane_width  # 3 lanes
    
    # Get a subset of the vehicles to visualize
    vehicle_indices = list(range(min(len(hist_x[0]), num_vehicles)))
    
    # Divide vehicles between the two plots if using split view
    for ax_idx, ax in enumerate(axes):
        # Draw lane markings - 3 lanes
        ax.axhline(y=0, color='white', linestyle='-', linewidth=2)
        ax.axhline(y=road_width, color='white', linestyle='-', linewidth=2)
        
        # Draw dashed lane dividers
        for lane in range(1, 3):
            ax.axhline(y=lane * lane_width, color='white', linestyle='--', linewidth=1.5)
        
        # Determine which vehicles to plot on this axis
        if split_view:
            start_idx = ax_idx * vehicles_per_plot
            end_idx = min(start_idx + vehicles_per_plot, len(vehicle_indices))
            plot_indices = vehicle_indices[start_idx:end_idx]
        else:
            plot_indices = vehicle_indices
        
        # Common x-range for centering the view
        all_x_values = []
        
        # Plot each vehicle trajectory
        for idx in plot_indices:
            # Get data for the vehicle
            vehicle_hist_x = hist_x[0][idx]
            vehicle_hist_y = hist_y[0][idx]
            vehicle_gt_x = gt_x[0][idx]
            vehicle_gt_y = gt_y[0][idx]
            vehicle_pred_x = pred_x[0][idx]
            vehicle_pred_y = pred_y[0][idx]
            
            # Collect all x values for setting limits later
            all_x_values.extend([vehicle_hist_x, vehicle_gt_x, vehicle_pred_x])
            
            # Add car silhouette at the end of historical trajectory
            car_length = 14  # feet
            car_width = 6    # feet
            car_x = vehicle_hist_x[-1]
            car_y = vehicle_hist_y[-1]
            
            # Create a car shape (simple rectangle with darker color)
            rect = patches.Rectangle((car_x - car_length/2, car_y - car_width/2), 
                                    car_length, car_width, 
                                    linewidth=1, edgecolor='black', facecolor='darkgray')
            ax.add_patch(rect)
            
            # Plot ground truth trajectory (green dots)
            ax.scatter(vehicle_gt_x, vehicle_gt_y, c='green', marker='.', s=30, label='ground truth' if idx == plot_indices[0] else "")
            
            # Plot predicted trajectory (blue triangles)
            ax.scatter(vehicle_pred_x, vehicle_pred_y, c='blue', marker='^', s=30, label='prediction' if idx == plot_indices[0] else "")
            
        # Add legend (only once per subplot)
        legend_elements = [
            plt.Line2D([0], [0], marker='.', color='white', label='ground truth', 
                      markerfacecolor='green', markersize=10, linestyle='none'),
            plt.Line2D([0], [0], marker='^', color='white', label='prediction', 
                      markerfacecolor='blue', markersize=10, linestyle='none')
        ]
        
        # Place legend in the upper right corner
        ax.legend(handles=legend_elements, loc='upper right', 
                 framealpha=0.9, fontsize=10)
        
        # Set axis limits to focus on the trajectories
        all_x = np.concatenate([np.array(x).flatten() for x in all_x_values])
        if len(all_x) > 0:  # Make sure there are values to calculate min/max
            x_min, x_max = np.min(all_x), np.max(all_x)
            x_range = x_max - x_min
            ax.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
            ax.set_ylim(-5, road_width + 5)
    
    plt.tight_layout()
    plt.savefig('multi_vehicle_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call the new visualization function to show multiple vehicles (10 by default)
# Set split_view=True to create a visualization similar to the image provided
visualize_multiple_vehicles(hist_x, hist_y, gt_x, gt_y, pred_x, pred_y, num_vehicles=10, split_view=True)
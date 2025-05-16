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

# Visualization code
def visualize_specific_scenarios(hist_x, hist_y, gt_x, gt_y, pred_x, pred_y):
    """
    Visualize vehicle trajectories showing ground truth vs predictions for specific time frames
    
    Args:
        hist_x, hist_y: Historical coordinates
        gt_x, gt_y: Ground truth coordinates
        pred_x, pred_y: Predicted coordinates
    """
    # Check how many vehicles we actually have
    num_vehicles = gt_x[0].shape[0]
    print(f"Number of vehicles in the batch: {num_vehicles}")
    
    # Create a figure with two subplots (similar to the reference image)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Select specific vehicles for visualization, making sure indices are within bounds
    # For the first timeframe (first subplot)
    vehicle_indices_frame1 = np.arange(min(10, num_vehicles))  # Use first 5 vehicles or all if fewer
    
    # For the second timeframe (second subplot)
    # If we have at least 10 vehicles, use vehicles 5-9 for the second frame
    # Otherwise, use the same vehicles as the first frame
    if num_vehicles >= 10:
        vehicle_indices_frame2 = np.arange(5, 10)
    else:
        vehicle_indices_frame2 = vehicle_indices_frame1
    
    # Set up the first timeframe plot
    ax = axes[0]
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
    
    # Plot vehicles for the first timeframe
    min_x_values = []
    max_x_values = []
    
    for idx in vehicle_indices_frame1:
        # Get vehicle data
        vehicle_hist_x = hist_x[0][idx]
        vehicle_hist_y = hist_y[0][idx]
        vehicle_gt_x = gt_x[0][idx]
        vehicle_gt_y = gt_y[0][idx]
        vehicle_pred_x = pred_x[0][idx]
        vehicle_pred_y = pred_y[0][idx]
        
        # Track min/max x values for setting axis limits
        min_x_values.append(np.min(vehicle_hist_x))
        max_x_values.append(np.max(vehicle_gt_x))
        max_x_values.append(np.max(vehicle_pred_x))
        
        # Add car silhouette at the end of historical trajectory
        car_length = 14  # feet
        car_width = 6    # feet
        car_x = vehicle_hist_x[-1]
        car_y = vehicle_hist_y[-1]
        rect = patches.Rectangle((car_x - car_length/2, car_y - car_width/2), car_length, car_width, 
                                linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rect)
        
        # Plot ground truth future trajectory
        ax.scatter(vehicle_gt_x, vehicle_gt_y, c='green', marker='.', s=50, label='Ground Truth' if idx == vehicle_indices_frame1[0] else "")
        # Connect points with lines for clarity
        ax.plot(vehicle_gt_x, vehicle_gt_y, c='green', linewidth=1, alpha=0.7)
        
        # Plot predicted future trajectory
        ax.scatter(vehicle_pred_x, vehicle_pred_y, c='blue', marker='^', s=50, label='Prediction' if idx == vehicle_indices_frame1[0] else "")
        # Connect points with lines for clarity
        ax.plot(vehicle_pred_x, vehicle_pred_y, c='blue', linewidth=1, alpha=0.7)
    
    # Set title and labels for first timeframe
    ax.set_title('Vehicle Trajectory Prediction - Timeframe 1')
    ax.set_xlabel('Longitudinal Position (feet)')
    ax.set_ylabel('Lateral Position (feet)')
    ax.legend(loc='upper right')
    
    # Set reasonable axis limits based on data
    if min_x_values and max_x_values:
        min_x = min(min_x_values) - 50
        max_x = max(max_x_values) + 50
        ax.set_xlim(min_x, max_x)
    ax.set_ylim(-10, road_width + 10)
    
    # Set up the second timeframe plot
    ax = axes[1]
    ax.set_facecolor('lightgray')
    ax.grid(False)
    
    # Draw lane markings for second subplot
    ax.axhline(y=0, color='white', linestyle='-', linewidth=2)
    ax.axhline(y=road_width, color='white', linestyle='-', linewidth=2)
    
    for lane in range(1, 3):
        ax.axhline(y=lane * lane_width, color='white', linestyle='--', linewidth=1)
    
    # Plot vehicles for the second timeframe
    min_x_values = []
    max_x_values = []
    
    for idx in vehicle_indices_frame2:
        # Get vehicle data
        # vehicle_hist_x = hist_x[0][idx]
        # vehicle_hist_y = hist_y[0][idx]
        # vehicle_gt_x = gt_x[0][idx]
        # vehicle_gt_y = gt_y[0][idx]
        # vehicle_pred_x = pred_x[0][idx]
        # vehicle_pred_y = pred_y[0][idx]


        vehicle_hist_x = hist_y[0][idx]
        vehicle_hist_y = hist_x[0][idx]
        vehicle_gt_x = gt_y[0][idx]
        vehicle_gt_y = gt_x[0][idx]
        vehicle_pred_x = pred_y[0][idx]
        vehicle_pred_y = pred_x[0][idx]
        
        # Track min/max x values for setting axis limits
        min_x_values.append(np.min(vehicle_hist_x))
        max_x_values.append(np.max(vehicle_gt_x))
        max_x_values.append(np.max(vehicle_pred_x))
        
        # Add car silhouette at the end of historical trajectory
        car_x = vehicle_hist_x[-1]
        car_y = vehicle_hist_y[-1]
        rect = patches.Rectangle((car_x - car_length/2, car_y - car_width/2), car_length, car_width, 
                                linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rect)
        
        # Plot ground truth future trajectory
        ax.scatter(vehicle_gt_x, vehicle_gt_y, c='green', marker='.', s=50, label='Ground Truth' if idx == vehicle_indices_frame2[0] else "")
        # Connect points with lines for clarity
        ax.plot(vehicle_gt_x, vehicle_gt_y, c='green', linewidth=1, alpha=0.7)
        
        # Plot predicted future trajectory
        ax.scatter(vehicle_pred_x, vehicle_pred_y, c='blue', marker='^', s=50, label='Prediction' if idx == vehicle_indices_frame2[0] else "")
        # Connect points with lines for clarity
        ax.plot(vehicle_pred_x, vehicle_pred_y, c='blue', linewidth=1, alpha=0.7)
    
    # Set title and labels for second timeframe
    ax.set_title('Vehicle Trajectory Prediction - Timeframe 2')
    ax.set_xlabel('Longitudinal Position (feet)')
    ax.set_ylabel('Lateral Position (feet)')
    ax.legend(loc='upper right')
    
    # Set reasonable axis limits based on data
    if min_x_values and max_x_values:
        min_x = min(min_x_values) - 50
        max_x = max(max_x_values) + 50
        ax.set_xlim(min_x, max_x)
    ax.set_ylim(-10, road_width + 10)
    
    plt.tight_layout()
    plt.savefig('trajectory_predictions_two_timeframes.png', dpi=300)
    plt.show()

# Call the visualization function
visualize_specific_scenarios(hist_x, hist_y, gt_x, gt_y, pred_x, pred_y)
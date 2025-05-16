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
net.load_state_dict(torch.load('sta_lstm_2.tar'))
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
    Visualize vehicle trajectories showing only ground truth vs predictions
    
    Args:
        hist_x, hist_y: Historical coordinates (not used in simplified version)
        gt_x, gt_y: Ground truth coordinates
        pred_x, pred_y: Predicted coordinates
    """
    # Check how many vehicles we actually have
    num_vehicles = gt_x[0].shape[0]
    print(f"Number of vehicles in the batch: {num_vehicles}")
    
    # Create a single plot
    plt.figure(figsize=(10, 8))
    
    # Select specific vehicles for visualization
    vehicle_indices = np.arange(min(10, num_vehicles))
    
    # Plot ground truth and predictions for each vehicle
    for idx in vehicle_indices:
        # Get vehicle data
        vehicle_gt_x = gt_x[0][idx]
        vehicle_gt_y = gt_y[0][idx]
        vehicle_pred_x = pred_x[0][idx]
        vehicle_pred_y = pred_y[0][idx]
        
        # Plot ground truth future trajectory
        plt.scatter(vehicle_gt_x, vehicle_gt_y, c='green', marker='.', s=50, 
                   label='Ground Truth' if idx == vehicle_indices[0] else "")
        # plt.plot(vehicle_gt_x, vehicle_gt_y, c='green', linewidth=1)
        
        # Plot predicted future trajectory
        plt.scatter(vehicle_pred_x, vehicle_pred_y, c='blue', marker='^', s=50, 
                   label='Prediction' if idx == vehicle_indices[0] else "")
        # plt.plot(vehicle_pred_x, vehicle_pred_y, c='blue', linewidth=1)
        
        # Draw a line connecting ground truth and prediction points
        for i in range(len(vehicle_gt_x)):
            plt.plot([vehicle_gt_x[i], vehicle_pred_x[i]], 
                    [vehicle_gt_y[i], vehicle_pred_y[i]], 
                    'k--', alpha=0.3)
    
    # Set title and labels
    plt.title('Vehicle Trajectory Prediction')
    plt.xlabel('Longitudinal Position (feet)')
    plt.ylabel('Lateral Position (feet)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simplified_trajectory_predictions.png', dpi=300)

# Call the visualization function
# visualize_specific_scenarios(hist_x, hist_y, gt_x, gt_y, pred_x, pred_y)


def remove_outliers_and_visualize(hist_x, hist_y, gt_x, gt_y, pred_x, pred_y, threshold=10):
    """
    Remove outliers based on a simple threshold of distance error and visualize ground truth vs predictions

    Args:
        hist_x, hist_y: Historical coordinates (not used in this visualization)
        gt_x, gt_y: Ground truth coordinates
        pred_x, pred_y: Predicted coordinates
        threshold: Distance threshold to consider a trajectory an outlier
    """
    # Check how many vehicles we actually have
    num_vehicles = gt_x[0].shape[0]
    print(f"Number of vehicles in the batch: {num_vehicles}")

    # Calculate Euclidean distance error for each vehicle trajectory
    errors = []
    for idx in range(num_vehicles):
        vehicle_gt_x = gt_x[0][idx]
        vehicle_gt_y = gt_y[0][idx]
        vehicle_pred_x = pred_x[0][idx]
        vehicle_pred_y = pred_y[0][idx]
        
        dist = np.sqrt((vehicle_gt_x - vehicle_pred_x)**2 + (vehicle_gt_y - vehicle_pred_y)**2)
        mean_error = np.mean(dist)
        errors.append(mean_error)

    errors = np.array(errors)
    
    # Filter vehicles based on the threshold
    valid_indices = np.where(errors <= threshold)[0]
    print(f"Number of vehicles after removing outliers: {len(valid_indices)}")

    # Plot only valid vehicles
    plt.figure(figsize=(10, 8))

    for i, idx in enumerate(valid_indices):
        vehicle_gt_x = gt_x[0][idx]
        vehicle_gt_y = gt_y[0][idx]
        vehicle_pred_x = pred_x[0][idx]
        vehicle_pred_y = pred_y[0][idx]

        # Plot ground truth future trajectory
        plt.scatter(vehicle_gt_x, vehicle_gt_y, c='green', marker='.', s=50, 
                   label='Ground Truth' if i == 0 else "")
        plt.plot(vehicle_gt_x, vehicle_gt_y, c='green', linewidth=1)
        
        # Plot predicted future trajectory
        plt.scatter(vehicle_pred_x, vehicle_pred_y, c='blue', marker='^', s=50, 
                   label='Prediction' if i == 0 else "")
        plt.plot(vehicle_pred_x, vehicle_pred_y, c='blue', linewidth=1)
        
        # Draw a line connecting ground truth and prediction points
        for j in range(len(vehicle_gt_x)):
            plt.plot([vehicle_gt_x[j], vehicle_pred_x[j]], 
                     [vehicle_gt_y[j], vehicle_pred_y[j]], 
                     'k--', alpha=0.3)

    plt.title('Vehicle Trajectory Prediction (Outliers Removed)')
    plt.xlabel('Longitudinal Position (feet)')
    plt.ylabel('Lateral Position (feet)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('trajectory_predictions_no_outliers_3.png', dpi=300)
    plt.show()

    return valid_indices


remove_outliers_and_visualize(hist_x, hist_y, gt_x, gt_y, pred_x, pred_y)
import torch
import torch.nn as nn

class KalmanFilter(nn.Module):
    def __init__(self):
        super(KalmanFilter, self).__init__()
        # Initialize Kalman Filter parameters

        # State transition matrix (F)
        self.F = torch.tensor([[1, 0, 1, 0],   # Assuming constant velocity model
                               [0, 1, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.float32)

        # Observation matrix (H)
        self.H = torch.tensor([[1, 0, 0, 0],   # We can observe position directly
                               [0, 1, 0, 0]], dtype=torch.float32)

        # Process noise covariance (Q)
        self.Q = torch.eye(4) * 0.01  # Small process noise

        # Measurement noise covariance (R)
        self.R = torch.eye(2) * 1.0    # Measurement noise

        # Initial error covariance (P)
        self.P_init = torch.eye(4) * 1000.0  # High initial uncertainty

    def forward(self, batch: torch.Tensor):
        """
        batch: Tensor of shape (batch_size, sequence_len, max_vehicles, 3)
               where the last dimension is (class, x, y)
        """
        batch_size, sequence_len, max_vehicles, _ = batch.shape
        device = batch.device

        # Initialize state estimates and error covariances for each vehicle
        state_estimates = torch.zeros(batch_size, max_vehicles, 4, device=device)  # [x, y, vx, vy]
        P = self.P_init.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_vehicles, 1, 1).to(device)

        # Prepare constants
        F = self.F.to(device)
        H = self.H.to(device)
        Q = self.Q.to(device)
        R = self.R.to(device)
        I = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)

        # List to store estimated positions at each time step
        estimated_positions = []

        # Iterate over each time step
        for t in range(sequence_len):
            observations = batch[:, t, :, :]  # Shape: (batch_size, max_vehicles, 3)
            labels = observations[:, :, 0]    # Class labels
            z = observations[:, :, 1:]        # Observed positions (x, y)

            # Create a mask for available measurements (labels != -1)
            mask = (labels != -1).unsqueeze(-1)  # Shape: (batch_size, max_vehicles, 1)

            # **Prediction Step**
            # Predict the next state
            state_pred = torch.matmul(state_estimates, F.T)

            # Predict the error covariance
            F_expanded = F.unsqueeze(0).unsqueeze(0)
            Q_expanded = Q.unsqueeze(0).unsqueeze(0)
            P_pred = torch.matmul(torch.matmul(F_expanded, P), F_expanded.transpose(-1, -2)) + Q_expanded

            # **Update Step** (Only for vehicles with available measurements)
            H_expanded = H.unsqueeze(0).unsqueeze(0)

            # Measurement residual
            y = z - state_pred[:, :, :2]  # Innovation
            y = y * mask  # Apply mask

            # S = H * P_pred * H^T + R
            S = torch.matmul(H_expanded, torch.matmul(P_pred, H_expanded.transpose(-1, -2))) + R.unsqueeze(0).unsqueeze(0)

            # Kalman Gain K = P_pred * H^T * S^-1
            K = torch.matmul(torch.matmul(P_pred, H_expanded.transpose(-1, -2)), torch.inverse(S))

            # Update state estimate
            y = y.unsqueeze(-1)  # Shape: (batch_size, max_vehicles, 2, 1)
            state_update = state_pred + torch.matmul(K, y).squeeze(-1) * mask

            # Update error covariance
            P_update = torch.matmul(I - torch.matmul(K, H_expanded), P_pred)

            # For vehicles without measurements, keep the predictions
            state_estimates = torch.where(mask.repeat(1, 1, 4), state_update, state_pred)
            P = torch.where(mask.unsqueeze(-1).repeat(1, 1, 4, 4), P_update, P_pred)

            # Save estimated positions
            estimated_positions.append(state_estimates[:, :, :2])  # Positions are the first two elements

        # Stack estimated positions over time
        estimated_positions = torch.stack(estimated_positions, dim=1)  # Shape: (batch_size, sequence_len, max_vehicles, 2)

        last_estimated_positions = estimated_positions[:, -1, :, :] # Shape: (batch_size, max_vehicles, 2)

        # Add the classification labels
        ones = torch.ones(batch_size, max_vehicles, 1, device=device)

        last_estimated_positions = torch.cat((ones, last_estimated_positions), dim=-1)

        return last_estimated_positions

class LastKnowledge(nn.Module):
    def __init__(self):
        super(LastKnowledge, self).__init__()

    def forward(self, batch: torch.Tensor):
        """
        batch: Tensor of shape (batch_size, sequence_len, max_vehicles, 3)
               where the last dimension is (class, x, y)
        """
        batch_size, sequence_len, max_vehicles, _ = batch.shape
        batched_results = []
        for item in batch:
            results = []
            for index, vehicle in enumerate(item[-1]):
                if vehicle[0] == -1:
                    for s in reversed(range(sequence_len)):
                        if item[s][index][0] != -1:
                            results.append([1, item[s][index][1], item[s][index][2]])
                            break
                    else:
                        results.append([1, 0, 0])
                else:
                    results.append([1, vehicle[1], vehicle[2]])
        
            results = torch.tensor(results, device=batch.device)
            batched_results.append(results)
        return torch.stack(batched_results)
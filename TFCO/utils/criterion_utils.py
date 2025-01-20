import torch.nn as nn 
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Dict
import numpy as np

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super(CombinedLoss, self).__init__()

        if 'types' not in config['criterion']:
            raise ValueError("Loss type not specified in the configuration")

        assert len(config['criterion']['types']) == len(config['criterion']['types_weight']), "The length of the types and types_weight must be equal"
        assert sum(config['criterion']['types_weight']) == 1, "The sum of the types_weight must be equal to 1"

        self.losses = nn.ModuleDict()
        self.weights = {}

        for loss_name, loss_weight in zip(config['criterion']['types'], config['criterion']['types_weight']):
            if loss_name == 'BCE':
                bce_weight = config['criterion'].get('BCE_weight', 1)
                pos_weight = torch.tensor([1.0 / bce_weight]).cuda()
                self.losses[loss_name] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            elif loss_name == 'IOU':
                self.losses[loss_name] = IoULoss()  # Assuming IoULoss is defined elsewhere
            else:
                raise ValueError(f"Loss type {loss_name} not supported")

            self.weights[loss_name] = loss_weight

    def forward(self, input, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        individual_losses = {}
        for loss_name, loss_fn in self.losses.items():
            loss_weight = self.weights[loss_name]
            total_loss += loss_weight * loss_fn(input, target)
            individual_losses[loss_name] = loss_fn(input, target).item()
        return total_loss, individual_losses



def get_criterion(config):
    if 'types' not in config['criterion']:
        raise ValueError("Loss type not specified in the configuration")
    
    # check if types len is equal to the types_weight len
    assert len(config['criterion']['types']) == len(config['criterion']['types_weight']), "The length of the types and types_weight must be equal"
    # check if the sum of the types_weight is equal to 1
    assert sum(config['criterion']['types_weight']) == 1, "The sum of the types_weight must be equal to 1"


    
    loss_type = config['criterion']['type']

    # Handle BCELoss specifically
    if loss_type == 'BCELoss':
        weight = config['criterion'].get('weight', None)
        
        # If weight is provided and it's 'BCELoss', use BCEWithLogitsLoss with pos_weight
        if weight is not None:
            pos_weight = torch.tensor([1.0 / weight]).cuda()
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCELoss()
    
    if loss_type == 'IOU':
        return IoULoss()

    else:
        raise ValueError(f"Loss type {loss_type} not supported")


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, targets):
        outputs = self.sigmoid(outputs)

        # invert the gt maks and outputs
        targets = 1 - targets
        outputs = 1 - outputs

        # Intersection
        intersection = torch.sum(outputs * targets, dim=[1, 2, 3])
        
        # Union
        total = torch.sum(outputs + targets, dim=[1, 2, 3])
        union = total - intersection

        # IoU
        iou = intersection / (union + 1e-10)

        # clamp values to (0, 1)
        iou = torch.clamp(iou, 0.0, 1.0)

        # Loss
        loss = 1.0 - iou

        # Mean IoU Loss over the batch
        return loss.mean()

class GPUHungarianMatcher(nn.Module): # TODO test and further optimize (is currently even slower than the linear_sum_assignment)
    def __init__(self, cost_class: float = 1.0, cost_l1: float = 1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_l1 = cost_l1

    @torch.no_grad()
    def forward(self, outputs, targets):
        # Assume outputs: [batch_size, num_queries, 3]
        # Assume targets: list of tensors, len(targets) = batch_size
        batch_size, num_queries, _ = outputs.shape

        # Flatten outputs
        out_prob = outputs[:, :, 0].softmax(-1)  # [batch_size, num_queries]
        out_pos = outputs[:, :, 1:]  # [batch_size, num_queries, 2]

        # Concatenate targets
        tgt_labels = torch.cat([v[:, 0] for v in targets])  # [num_total_targets]
        tgt_pos = torch.cat([v[:, 1:] for v in targets])    # [num_total_targets, 2]
        sizes = [len(v) for v in targets]

        # Compute classification cost
        cost_class = -out_prob[:, :, None]  # [batch_size, num_queries, 1]

        # Compute L1 cost
        cost_l1 = torch.cdist(out_pos, tgt_pos, p=1)  # [batch_size, num_queries, num_total_targets]

        # Compute total cost
        C = self.cost_class * cost_class + self.cost_l1 * cost_l1
        C = C.view(batch_size * num_queries, -1)

        # Compute assignments in a batched way
        indices = []
        for i, size in enumerate(sizes):
            c = C[i * num_queries:(i + 1) * num_queries, sum(sizes[:i]):sum(sizes[:i + 1])]
            c = c.cpu()
            row_ind, col_ind = linear_sum_assignment(c)
            indices.append((torch.as_tensor(row_ind), torch.as_tensor(col_ind)))

        return indices
    
class HungarianMatcher(nn.Module):
    """
    Computes an assignment between the targets and predictions of the network,
    including both classification and position costs.
    """
    def __init__(self, cost_class: float = 10.0, cost_l1: float = 1.0):
        """
        Initializes the matcher.

        Args:
            cost_class (float): Weight of the classification error in the matching cost.
            cost_l1 (float): Weight of the L1 error in the matching cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_l1 = cost_l1

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching between outputs and targets.

        Args:
            outputs (tensor): Output tensor from the model of shape [batch_size, max_num_targets, 3].
            targets (list of tensors): List of tensors, each of shape [num_targets_i, 3]

        Returns:
            List of tuples: Each tuple contains two tensors (index_i, index_j) for sample i,
            where:
                - index_i: Indices of the selected predictions.
                - index_j: Indices of the corresponding selected targets.
        """
        batch_size, max_num_targets, num_features = outputs.shape
        indices = []

        for i in range(batch_size):
            out_logits_i = outputs[i, :, 0]  # [max_num_targets]
            out_positions_i = outputs[i, :, 1:]  # [max_num_targets, 2]

            targets_i = targets[i]  # [num_targets_i, 3]
            tgt_labels_i = targets_i[:, 0]  # [num_targets_i]
            tgt_positions_i = targets_i[:, 1:]  # [num_targets_i, 2]

            # Filter targets where classification is 1
            idx_one = (tgt_labels_i == 1)
            if idx_one.sum() == 0:
                # No targets with label 1, skip matching
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            tgt_labels_one = tgt_labels_i[idx_one]
            tgt_positions_one = tgt_positions_i[idx_one]

            num_targets_one = tgt_labels_one.shape[0]

            # Compute negative log-likelihood for predictions being 1
            # Since all target labels are 1, the cost is -log(sigmoid(out_logits_i))
            cost_class_i = -F.logsigmoid(out_logits_i)  # [max_num_targets]

            # Expand cost_class_i to match dimensions with cost_l1_i
            cost_class_i = cost_class_i.unsqueeze(1).expand(-1, num_targets_one)  # [max_num_targets, num_targets_one]

            # Compute L1 cost only for targets with label 1
            cost_l1_i = torch.cdist(out_positions_i, tgt_positions_one, p=1)  # [max_num_targets, num_targets_one]

            # Total cost matrix
            C_i = self.cost_class * cost_class_i + self.cost_l1 * cost_l1_i  # [max_num_targets, num_targets_one]

            # Solve the assignment problem
            cost = C_i.cpu()
            row_ind, col_ind = linear_sum_assignment(cost)

            # Map col_ind back to indices in the original targets_i
            target_indices = idx_one.nonzero(as_tuple=False).squeeze(1)[col_ind]

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                target_indices
            ))

        return indices



class TrafficPositionLoss(nn.Module):
    """
    Computes the bipartite matching loss between predicted and target positions,
    including both classification and position components.
    """
    def __init__(self, matcher, weight_class: float = 10.0, weight_l1: float = 1.0):
        """
        Args:
            matcher (HungarianMatcher): Matcher module to compute matching cost.
            weight_class (float): Weight for the classification loss.
            weight_l1 (float): Weight for the position loss.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_class = weight_class
        self.weight_l1 = weight_l1

    def forward(self, outputs, targets):
        """
        Computes the total loss.

        Args:
            outputs (tensor): Output tensor from the model of shape [batch_size, max_num_targets, 3].
            targets (list of tensors): List of tensors, each of shape [num_targets_i, 3].

        Returns:
            float: Total loss.  
        """
        batch_size, max_num_targets, num_features = outputs.shape

        indices = self.matcher(outputs, targets)

        # Initialize loss components
        loss_class = 0.0
        loss_position = 0.0

        for i in range(batch_size):
            src_idx = indices[i][0].to(outputs.device)
            tgt_idx = indices[i][1].to(outputs.device)

            # Matched predictions and targets
            if src_idx.numel() > 0:
                pred_logits = outputs[i, src_idx, 0]  # [num_matched]
                target_labels = targets[i][tgt_idx, 0]  # [num_matched]

                # Classification loss for matched predictions (targets are ones)
                loss_class += F.binary_cross_entropy_with_logits(pred_logits, target_labels, reduction='sum')

                # Position loss for matched predictions
                pred_positions = outputs[i, src_idx, 1:]  # [num_matched, 2]
                target_positions = targets[i][tgt_idx, 1:]  # [num_matched, 2]
                loss_position += F.l1_loss(pred_positions, target_positions, reduction='sum')
            else:
                # No matched predictions; skip position loss
                pass

            # Classification loss for unmatched predictions (targets are zeros)
            unmatched_mask = torch.ones(max_num_targets, dtype=torch.bool, device=outputs.device)
            if src_idx.numel() > 0:
                unmatched_mask[src_idx] = False
            unmatched_src_idx = unmatched_mask.nonzero(as_tuple=False).squeeze(1)

            if unmatched_src_idx.numel() > 0:
                unmatched_logits = outputs[i, unmatched_src_idx, 0]  # [num_unmatched]
                unmatched_labels = torch.zeros_like(unmatched_logits)  # zeros
                loss_class += F.binary_cross_entropy_with_logits(unmatched_logits, unmatched_labels, reduction='sum')

        # Normalize the losses
        total_preds = batch_size * max_num_targets
        loss_class = loss_class / total_preds
        loss_position = loss_position / total_preds

        total_loss = self.weight_class * loss_class + self.weight_l1 * loss_position

        additional_information = {
            'class_loss': loss_class.item() * self.weight_class,
            'position_loss': loss_position.item() * self.weight_l1,
            'matching': indices
        }
        return total_loss, additional_information

class SingleTrafficPositionLoss(nn.Module):
    """
    Computes the binary classification loss and l1 loss for the position"""
    def __init__(self, class_weight: float = 1.0, distance_weight: float = 50.0):
        """
        Args:
            weight_class (float): Weight for the classification loss.
            distance_weight (float): Weight for the position loss.
        """
        super().__init__()
        self.weight_class = class_weight
        self.weight_l2 = distance_weight
    
    def forward(self, outputs, targets):
        batch_size, max_num_targets, num_features = outputs.shape

        # Classification logits and labels
        pred_logits = outputs[:, :, 0]
        target_labels = targets[:, :, 0]

        # Mask to select only items where target_labels == 1
        relevant_mask = target_labels == 1

        # Select the relevant classification outputs
        relevant_pred_logits = pred_logits[relevant_mask]
        relevant_target_labels = target_labels[relevant_mask]

        # Compute classification loss only for relevant items
        if relevant_pred_logits.numel() > 0:
            loss_class = F.binary_cross_entropy_with_logits(relevant_pred_logits, relevant_target_labels, reduction='mean')
        else:
            loss_class = torch.tensor(0.0, device=outputs.device)

        # Compute predicted probabilities for position weighting (soft weights)
        # Use sigmoid to get probabilities from logits
        relevant_pred_probs = torch.sigmoid(relevant_pred_logits)

        # Select the relevant positions
        relevant_preds_positions = outputs[:, :, 1:][relevant_mask]
        relevant_targets_positions = targets[:, :, 1:][relevant_mask]

        # Compute position loss with soft weights
        if relevant_preds_positions.numel() > 0:
            # Compute L2 loss without reduction to keep per-element loss
            position_loss = F.mse_loss(relevant_preds_positions, relevant_targets_positions, reduction='none')
            # Mean over position dimensions (assuming positions are the last dimension)
            position_loss = position_loss.mean(dim=1)
            # Weight position loss by predicted probabilities (soft weights)
            weighted_position_loss = (position_loss * relevant_pred_probs).mean()
        else:
            weighted_position_loss = torch.tensor(0.0, device=outputs.device)

        # Total loss
        total_loss = self.weight_class * loss_class + self.weight_l2 * weighted_position_loss

        additional_information = {
            'class_loss': loss_class.item() * self.weight_class,
            'position_loss': weighted_position_loss.item() * self.weight_l2
        }

        return total_loss, additional_information



if __name__ == "__main__":
    # test the TrafficPositonLoss
    matcher = HungarianMatcher()
    criterion = TrafficPositionLoss(matcher)

    batch_size = 16
    max_num_targets = 100

    outputs = torch.randn(batch_size, max_num_targets, 3)
    targets = torch.randint(0, 2, (batch_size, max_num_targets, 3)).float() # Random binary labels

    loss, individual_losses = criterion(outputs, targets)

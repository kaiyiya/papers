"""
Saliency-guided loss functions for scanpath prediction.

This module implements three core components for guiding scanpath prediction
towards salient regions:
1. Sampling Loss: Encourages fixations to fall on salient regions
2. Coverage Loss: Ensures high-saliency regions are visited
3. Sequence Loss: Rewards progressive movement towards saliency peaks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyGuidedLoss(nn.Module):
    """
    Saliency-guided loss for scanpath prediction.

    This loss function guides the model to generate scanpaths that follow
    human visual attention patterns by explicitly optimizing for saliency.

    Args:
        image_size: Tuple of (height, width) for the input images
        sampling_weight: Weight for sampling loss (default: 1.0)
        coverage_weight: Weight for coverage loss (default: 0.5)
        sequence_weight: Weight for sequence loss (default: 0.3)
        target_coverage: Target coverage ratio for high-saliency regions (default: 0.5)
        saliency_threshold_quantile: Quantile for defining high-saliency regions (default: 0.8)
    """

    def __init__(
        self,
        image_size,
        sampling_weight=1.0,
        coverage_weight=0.5,
        sequence_weight=0.3,
        target_coverage=0.5,
        saliency_threshold_quantile=0.8
    ):
        super().__init__()
        self.image_size = image_size
        self.sampling_weight = sampling_weight
        self.coverage_weight = coverage_weight
        self.sequence_weight = sequence_weight
        self.target_coverage = target_coverage
        self.saliency_threshold_quantile = saliency_threshold_quantile

    def forward(self, pred_scanpath, saliency_map, return_metrics=True):
        """
        Compute saliency-guided loss.

        Args:
            pred_scanpath: (B, T, 2) - Predicted scanpath, values in [0, 1], format (x, y)
            saliency_map: (B, 1, H, W) - Saliency map, normalized to [0, 1]
            return_metrics: Whether to return additional metrics (default: True)

        Returns:
            loss_dict: Dictionary containing:
                - sal_sampling: Sampling loss value
                - sal_coverage: Coverage loss value
                - sal_sequence: Sequence loss value
                - total: Total weighted loss
                If return_metrics=True, also includes:
                - mean_saliency: Mean saliency at fixation points
                - coverage_ratio: Actual coverage of high-saliency regions
                - early_saliency_gain: Saliency increase in early phase
        """
        B, T, _ = pred_scanpath.shape
        device = pred_scanpath.device

        # Handle wrap-around for X coordinates (360-degree images)
        pred_scanpath = pred_scanpath.clone()
        pred_scanpath[:, :, 0] = pred_scanpath[:, :, 0] % 1.0

        # Convert coordinates from [0, 1] to [-1, 1] for grid_sample
        # grid_sample expects (x, y) format, which matches our scanpath format
        grid = pred_scanpath * 2.0 - 1.0  # (B, T, 2)
        grid = grid.unsqueeze(2)  # (B, T, 1, 2) - add spatial dimension

        # 1. Sampling Loss: Maximize saliency at fixation points
        sampled_saliency = F.grid_sample(
            saliency_map,  # (B, 1, H, W)
            grid,  # (B, T, 1, 2)
            mode='bilinear',
            padding_mode='border',  # Clamp out-of-bounds to nearest valid value
            align_corners=False
        )  # (B, 1, T, 1)

        sampled_saliency = sampled_saliency.squeeze(-1).squeeze(1)  # (B, T)

        # Loss: negative mean (minimize = maximize saliency)
        sampling_loss = -torch.mean(sampled_saliency)

        # 2. Coverage Loss: Ensure high-saliency regions are visited
        # Identify high-saliency regions (top 20%)
        saliency_flat = saliency_map.view(B, -1)  # (B, H*W)
        threshold = torch.quantile(
            saliency_flat,
            self.saliency_threshold_quantile,
            dim=1,
            keepdim=True
        )  # (B, 1)
        threshold = threshold.view(B, 1, 1, 1)  # (B, 1, 1, 1)

        high_sal_mask = (saliency_map >= threshold).float()  # (B, 1, H, W)

        # Sample mask at fixation points
        sampled_mask = F.grid_sample(
            high_sal_mask,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # (B, 1, T, 1)

        sampled_mask = sampled_mask.squeeze(-1).squeeze(1)  # (B, T)

        # Calculate coverage ratio (proportion of fixations in high-saliency regions)
        coverage_ratio = torch.mean(sampled_mask, dim=1)  # (B,)

        # Loss: MSE with target coverage
        target = torch.full_like(coverage_ratio, self.target_coverage)
        coverage_loss = F.mse_loss(coverage_ratio, target)

        # 3. Sequence Loss: Encourage progressive movement towards saliency
        # Early phase (first 1/3): encourage saliency increase
        early_end = max(2, T // 3)
        early_saliency = sampled_saliency[:, :early_end]  # (B, early_end)

        if early_saliency.shape[1] > 1:
            # Compute saliency differences (later - earlier)
            saliency_diff = early_saliency[:, 1:] - early_saliency[:, :-1]  # (B, early_end-1)
            # Reward positive differences (moving towards higher saliency)
            early_loss = -torch.mean(saliency_diff)
        else:
            early_loss = torch.tensor(0.0, device=device)

        # Middle phase (middle 1/3): maintain high saliency
        middle_start = early_end
        middle_end = max(middle_start + 1, 2 * T // 3)
        middle_saliency = sampled_saliency[:, middle_start:middle_end]  # (B, middle_len)

        if middle_saliency.numel() > 0:
            # Reward high saliency values
            middle_loss = -torch.mean(middle_saliency)
        else:
            middle_loss = torch.tensor(0.0, device=device)

        # Late phase: no constraint (allow exploration/refinement)

        # Combine early and middle losses
        sequence_loss = 0.5 * early_loss + 0.5 * middle_loss

        # Total weighted loss
        total_loss = (
            self.sampling_weight * sampling_loss +
            self.coverage_weight * coverage_loss +
            self.sequence_weight * sequence_loss
        )

        # Prepare output dictionary
        loss_dict = {
            'sal_sampling': sampling_loss,
            'sal_coverage': coverage_loss,
            'sal_sequence': sequence_loss,
            'total': total_loss
        }

        # Add metrics if requested
        if return_metrics:
            loss_dict['mean_saliency'] = torch.mean(sampled_saliency).item()
            loss_dict['coverage_ratio'] = torch.mean(coverage_ratio).item()

            if early_saliency.shape[1] > 1:
                early_gain = torch.mean(early_saliency[:, -1] - early_saliency[:, 0]).item()
                loss_dict['early_saliency_gain'] = early_gain
            else:
                loss_dict['early_saliency_gain'] = 0.0

        return loss_dict


def test_saliency_loss():
    """Unit test for SaliencyGuidedLoss."""
    print("Testing SaliencyGuidedLoss...")

    # Create test data
    B, T, H, W = 2, 30, 256, 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Random scanpath
    pred_scanpath = torch.rand(B, T, 2, device=device, requires_grad=True)

    # Create a saliency map with a clear peak
    saliency_map = torch.rand(B, 1, H, W, device=device)
    # Add a strong peak at center
    center_h, center_w = H // 2, W // 2
    saliency_map[:, :, center_h-20:center_h+20, center_w-20:center_w+20] = 0.9

    # Initialize loss function
    loss_fn = SaliencyGuidedLoss(image_size=(H, W))

    # Forward pass
    loss_dict = loss_fn(pred_scanpath, saliency_map)

    # Verify output structure
    required_keys = ['sal_sampling', 'sal_coverage', 'sal_sequence', 'total']
    metric_keys = ['mean_saliency', 'coverage_ratio', 'early_saliency_gain']

    for key in required_keys:
        assert key in loss_dict, f"Missing key: {key}"
        assert isinstance(loss_dict[key], torch.Tensor), f"{key} should be a tensor"
        print(f"  {key}: {loss_dict[key].item():.4f}")

    for key in metric_keys:
        assert key in loss_dict, f"Missing metric: {key}"
        print(f"  {key}: {loss_dict[key]:.4f}")

    # Test backward pass
    loss_dict['total'].backward()
    assert pred_scanpath.grad is not None, "Gradient should be computed"

    print("✓ All tests passed!")

    # Test with scanpath pointing to high-saliency region
    print("\nTesting with scanpath at saliency peak...")
    pred_scanpath_centered = torch.zeros(B, T, 2, device=device)
    pred_scanpath_centered[:, :, 0] = center_w / W  # x
    pred_scanpath_centered[:, :, 1] = center_h / H  # y

    loss_dict_centered = loss_fn(pred_scanpath_centered, saliency_map)

    print(f"  Random scanpath - mean_saliency: {loss_dict['mean_saliency']:.4f}")
    print(f"  Centered scanpath - mean_saliency: {loss_dict_centered['mean_saliency']:.4f}")

    assert loss_dict_centered['mean_saliency'] > loss_dict['mean_saliency'], \
        "Centered scanpath should have higher mean saliency"

    print("✓ Saliency guidance working correctly!")


if __name__ == '__main__':
    test_saliency_loss()

#!/usr/bin/env python3
"""
Recurrent GAN for Mouse Trajectory Generation

This module implements a WGAN-GP based recurrent neural network for generating
realistic, human-like mouse trajectories between start and end points.

Architecture:
- Generator: LSTM-based autoregressive model conditioned on start/end points
- Discriminator: Bidirectional LSTM with kinematic feature extraction
"""

import argparse
import csv
import math
import os
from contextlib import nullcontext
from datetime import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings(
    "ignore",
    message="Attempting to run cuBLAS, but there was no current CUDA context!"
)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Hyperparameters and configuration."""
    # Data
    screen_width: float = 1920.0
    screen_height: float = 1080.0
    min_trajectory_length: int = 5
    max_trajectory_length: int = 200
    
    # Model
    latent_dim: int = 64
    generator_hidden_dim: int = 256
    generator_num_layers: int = 2
    discriminator_hidden_dim: int = 128
    discriminator_num_layers: int = 2
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.5, 0.9)
    n_critic: int = 1  # Train discriminator n times per generator update
    gradient_penalty_weight: float = 10.0
    gp_every_n: int = 4  # Compute gradient penalty every N discriminator steps
    gp_batch_frac: float = 0.25  # Fraction of batch used for gradient penalty
    use_amp: bool = True  # Use mixed precision on CUDA
    epochs: int = 1000
    
    # DataLoader performance
    num_workers: int = 4
    pin_memory: bool = True
    
    # Early stopping and scheduling
    patience: int = 50
    lr_patience: int = 20
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Generation
    distance_threshold: float = 0.02  # Stop when within 2% of target
    max_generation_steps: int = 200
    
    # Teacher forcing
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.0
    teacher_forcing_decay_epochs: int = 100
    
    # Loss weights
    endpoint_loss_weight: float = 50.0
    direction_loss_weight: float = 10.0


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

class MouseTrajectoryDataset(Dataset):
    """
    Dataset for mouse trajectory data.
    
    Parses CSV files, segments trajectories on total_duration == 0 resets,
    and extracts (dx, dy, dt) sequences with start/end points.
    
    Supports data augmentation via mirroring to generate trajectories in all directions.
    """
    
    def __init__(
        self,
        data_path: str,
        config: Config,
        normalize: bool = True,
        augment: bool = True
    ):
        self.config = config
        self.normalize = normalize
        self.augment = augment
        self.trajectories: List[Dict] = []
        
        # Load all CSV files from directory
        data_path = Path(data_path)
        if data_path.is_file():
            csv_files = [data_path]
        else:
            csv_files = list(data_path.glob("*.csv"))
        
        for csv_file in csv_files:
            self._load_csv(csv_file)
        
        base_count = len(self.trajectories)
        
        # Apply data augmentation if enabled
        if self.augment:
            self._augment_trajectories()
        
        print(f"Loaded {base_count} base trajectories from {len(csv_files)} files")
        if self.augment:
            print(f"After augmentation: {len(self.trajectories)} trajectories (4x via mirroring)")
    
    def _load_csv(self, csv_path: Path) -> None:
        """Load and segment trajectories from a CSV file."""
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            current_trajectory: List[Dict] = []
            
            for row in reader:
                total_duration = float(row['total_duration'])
                
                # Reset indicates new trajectory
                if total_duration == 0 and len(current_trajectory) > 0:
                    self._process_trajectory(current_trajectory)
                    current_trajectory = []
                
                current_trajectory.append({
                    'x': float(row['position_x']),
                    'y': float(row['position_y']),
                    'dt': float(row['time_between_movements']),
                    'click': row.get('click_events', 'None')
                })
            
            # Process last trajectory
            if len(current_trajectory) > 0:
                self._process_trajectory(current_trajectory)
    
    def _process_trajectory(self, points: List[Dict]) -> None:
        """Process a raw trajectory into training format."""
        if len(points) < self.config.min_trajectory_length:
            return
        if len(points) > self.config.max_trajectory_length:
            points = points[:self.config.max_trajectory_length]
        
        # Extract positions
        positions = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
        dts = np.array([p['dt'] for p in points], dtype=np.float32)
        
        # Compute deltas (dx, dy)
        deltas = np.diff(positions, axis=0)
        dts = dts[1:]  # dt corresponds to time to reach this point
        
        # Avoid division by zero
        dts = np.maximum(dts, 1e-6)
        
        # Normalize coordinates to [0, 1]
        if self.normalize:
            start = positions[0] / np.array([self.config.screen_width, self.config.screen_height])
            end = positions[-1] / np.array([self.config.screen_width, self.config.screen_height])
            # Normalize deltas relative to screen size
            deltas = deltas / np.array([self.config.screen_width, self.config.screen_height])
        else:
            start = positions[0]
            end = positions[-1]
        
        # Create trajectory dict
        trajectory = {
            'start': torch.tensor(start, dtype=torch.float32),
            'end': torch.tensor(end, dtype=torch.float32),
            'deltas': torch.tensor(deltas, dtype=torch.float32),
            'dts': torch.tensor(dts, dtype=torch.float32),
            'length': len(deltas)
        }
        
        self.trajectories.append(trajectory)
    
    def _augment_trajectories(self) -> None:
        """
        Augment trajectories via mirroring to cover all directions.
        
        Creates 3 additional versions of each trajectory:
        1. Horizontal flip (mirror across vertical axis)
        2. Vertical flip (mirror across horizontal axis)
        3. Both flips (180 degree rotation)
        
        This transforms a downward-right movement into:
        - downward-left (h-flip)
        - upward-right (v-flip)
        - upward-left (both)
        """
        augmented = []
        
        for traj in self.trajectories:
            start = traj['start']
            end = traj['end']
            deltas = traj['deltas']
            dts = traj['dts']
            length = traj['length']
            
            # Horizontal flip: negate x coordinates
            # New start/end x = 1 - old x (in normalized coords)
            h_flip_start = torch.tensor([1.0 - start[0].item(), start[1].item()])
            h_flip_end = torch.tensor([1.0 - end[0].item(), end[1].item()])
            h_flip_deltas = deltas.clone()
            h_flip_deltas[:, 0] = -h_flip_deltas[:, 0]  # Negate dx
            
            augmented.append({
                'start': h_flip_start,
                'end': h_flip_end,
                'deltas': h_flip_deltas,
                'dts': dts.clone(),
                'length': length
            })
            
            # Vertical flip: negate y coordinates
            # New start/end y = 1 - old y (in normalized coords)
            v_flip_start = torch.tensor([start[0].item(), 1.0 - start[1].item()])
            v_flip_end = torch.tensor([end[0].item(), 1.0 - end[1].item()])
            v_flip_deltas = deltas.clone()
            v_flip_deltas[:, 1] = -v_flip_deltas[:, 1]  # Negate dy
            
            augmented.append({
                'start': v_flip_start,
                'end': v_flip_end,
                'deltas': v_flip_deltas,
                'dts': dts.clone(),
                'length': length
            })
            
            # Both flips: negate both x and y (180 degree rotation)
            both_flip_start = torch.tensor([1.0 - start[0].item(), 1.0 - start[1].item()])
            both_flip_end = torch.tensor([1.0 - end[0].item(), 1.0 - end[1].item()])
            both_flip_deltas = -deltas.clone()  # Negate both dx and dy
            
            augmented.append({
                'start': both_flip_start,
                'end': both_flip_end,
                'deltas': both_flip_deltas,
                'dts': dts.clone(),
                'length': length
            })
        
        # Add augmented trajectories to the dataset
        self.trajectories.extend(augmented)
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.trajectories[idx]


def collate_trajectories(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-length trajectories.
    
    Returns padded sequences and lengths for pack_padded_sequence.
    """
    # Sort by length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    starts = torch.stack([item['start'] for item in batch])
    ends = torch.stack([item['end'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    
    # Pad sequences: combine deltas and dts into (dx, dy, dt)
    sequences = []
    for item in batch:
        seq = torch.cat([item['deltas'], item['dts'].unsqueeze(-1)], dim=-1)
        sequences.append(seq)
    
    # Pad to max length in batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    return {
        'starts': starts,
        'ends': ends,
        'sequences': padded_sequences,  # (batch, max_len, 3)
        'lengths': lengths
    }


# =============================================================================
# Kinematic Feature Computation
# =============================================================================

def compute_kinematics(
    positions: torch.Tensor,
    dts: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute kinematic features from trajectory positions.
    
    Args:
        positions: (batch, seq_len, 2) absolute x,y positions
        dts: (batch, seq_len) time deltas
        eps: small value to avoid division by zero
    
    Returns:
        features: (batch, seq_len, 9) tensor with:
            [x, y, dx, dy, dt, velocity, acceleration, jerk, curvature]
    """
    batch_size, seq_len, _ = positions.shape
    
    # Compute deltas
    dx = torch.zeros_like(positions[:, :, 0])
    dy = torch.zeros_like(positions[:, :, 1])
    dx[:, 1:] = positions[:, 1:, 0] - positions[:, :-1, 0]
    dy[:, 1:] = positions[:, 1:, 1] - positions[:, :-1, 1]
    
    # Velocity magnitude
    displacement = torch.sqrt(dx**2 + dy**2 + eps)
    velocity = displacement / (dts + eps)
    
    # Velocity components for curvature
    vx = dx / (dts + eps)
    vy = dy / (dts + eps)
    
    # Acceleration (dv/dt)
    acceleration = torch.zeros_like(velocity)
    dv = velocity[:, 1:] - velocity[:, :-1]
    dt_mid = (dts[:, 1:] + dts[:, :-1]) / 2 + eps
    acceleration[:, 1:] = dv / dt_mid
    
    # Jerk (da/dt)
    jerk = torch.zeros_like(acceleration)
    da = acceleration[:, 2:] - acceleration[:, 1:-1]
    jerk[:, 2:] = da / (dt_mid[:, 1:] + eps)
    
    # Curvature: |v x a| / |v|^3
    # For 2D: curvature = |vx * ay - vy * ax| / (vx^2 + vy^2)^1.5
    ax = torch.zeros_like(vx)
    ay = torch.zeros_like(vy)
    ax[:, 1:] = (vx[:, 1:] - vx[:, :-1]) / dt_mid
    ay[:, 1:] = (vy[:, 1:] - vy[:, :-1]) / dt_mid
    
    cross = torch.abs(vx * ay - vy * ax)
    speed_cubed = (vx**2 + vy**2 + eps) ** 1.5
    curvature = cross / speed_cubed
    
    # Clamp extreme values
    velocity = torch.clamp(velocity, 0, 100)
    acceleration = torch.clamp(acceleration, -1000, 1000)
    jerk = torch.clamp(jerk, -10000, 10000)
    curvature = torch.clamp(curvature, 0, 100)
    
    # Stack features
    features = torch.stack([
        positions[:, :, 0],  # x
        positions[:, :, 1],  # y
        dx,                   # dx
        dy,                   # dy
        dts,                  # dt
        velocity,             # velocity magnitude
        acceleration,         # acceleration
        jerk,                 # jerk
        curvature             # curvature
    ], dim=-1)
    
    return features


def trajectory_to_absolute(
    start: torch.Tensor,
    deltas: torch.Tensor
) -> torch.Tensor:
    """
    Convert relative deltas to absolute positions.
    
    Args:
        start: (batch, 2) starting positions
        deltas: (batch, seq_len, 2) relative displacements
    
    Returns:
        positions: (batch, seq_len+1, 2) absolute positions including start
    """
    batch_size, seq_len, _ = deltas.shape
    
    # Cumulative sum of deltas
    cumsum = torch.cumsum(deltas, dim=1)
    
    # Add start position
    positions = cumsum + start.unsqueeze(1)
    
    # Prepend start position
    positions = torch.cat([start.unsqueeze(1), positions], dim=1)
    
    return positions


# =============================================================================
# Generator Network
# =============================================================================

class Generator(nn.Module):
    """
    LSTM-based generator for mouse trajectories.
    
    Generates sequences of (dx, dy, dt) conditioned on start and end points.
    Uses autoregressive decoding during inference.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Conditioning encoder: (start_x, start_y, end_x, end_y, distance, angle)
        condition_dim = 6
        
        # Initial projection
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim + config.latent_dim, config.generator_hidden_dim),
            nn.LayerNorm(config.generator_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.generator_hidden_dim, config.generator_hidden_dim),
            nn.LayerNorm(config.generator_hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # LSTM for sequence generation
        # Input: (dx, dy, dt) + condition + dynamic features (remaining_dx, remaining_dy, remaining_dist, remaining_angle)
        self.lstm = nn.LSTM(
            input_size=3 + config.generator_hidden_dim + 4,  # +4 for dynamic features
            hidden_size=config.generator_hidden_dim,
            num_layers=config.generator_num_layers,
            batch_first=True,
            dropout=0.1 if config.generator_num_layers > 1 else 0
        )
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(config.generator_hidden_dim, config.generator_hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(config.generator_hidden_dim // 2, 3)  # (dx, dy, dt)
        )
        
        # Learnable initial input
        self.initial_input = nn.Parameter(torch.zeros(1, 1, 3))
    
    def _compute_condition(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """Compute conditioning vector from start, end, and noise."""
        # Compute distance and angle
        diff = end - start
        distance = torch.sqrt((diff ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        angle = torch.atan2(diff[:, 1:2], diff[:, 0:1])
        
        # Concatenate all conditioning info
        condition_input = torch.cat([start, end, distance, angle, z], dim=-1)
        
        return self.condition_encoder(condition_input)
    
    def _compute_dynamic_features(
        self,
        current_pos: torch.Tensor,
        end: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic features based on current position relative to target.
        
        Args:
            current_pos: (batch, 2) current position
            end: (batch, 2) target end position
        
        Returns:
            features: (batch, 4) tensor with:
                [remaining_dx, remaining_dy, remaining_distance, remaining_angle]
        """
        remaining = end - current_pos  # (batch, 2)
        remaining_dist = torch.sqrt((remaining ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        remaining_angle = torch.atan2(remaining[:, 1:2], remaining[:, 0:1])
        
        return torch.cat([remaining, remaining_dist, remaining_angle], dim=-1)
    
    def forward(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        z: torch.Tensor,
        target_sequences: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            start: (batch, 2) start positions
            end: (batch, 2) end positions  
            z: (batch, latent_dim) noise vector
            target_sequences: (batch, max_len, 3) target (dx, dy, dt) for teacher forcing
            target_lengths: (batch,) actual sequence lengths
            teacher_forcing_ratio: probability of using teacher forcing
        
        Returns:
            outputs: (batch, max_len, 3) generated sequences
            lengths: (batch,) generated lengths
        """
        batch_size = start.shape[0]
        device = start.device
        
        # Compute conditioning
        condition = self._compute_condition(start, end, z)
        
        if target_sequences is not None:
            # Training mode with known target lengths
            max_len = target_sequences.shape[1]
        else:
            max_len = self.config.max_generation_steps
        
        # Initialize LSTM hidden state
        h = torch.zeros(
            self.config.generator_num_layers,
            batch_size,
            self.config.generator_hidden_dim,
            device=device
        )
        c = torch.zeros_like(h)
        
        # Storage for outputs
        outputs = []
        current_input = self.initial_input.expand(batch_size, 1, 3)
        current_pos = start.clone()  # Track current position for dynamic features
        
        # Autoregressive generation
        for t in range(max_len):
            # Compute dynamic features (remaining distance to target)
            dynamic_feat = self._compute_dynamic_features(current_pos, end)
            
            # Concatenate input with condition AND dynamic features
            lstm_input = torch.cat([
                current_input,
                condition.unsqueeze(1),
                dynamic_feat.unsqueeze(1)
            ], dim=-1)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Generate output
            output = self.output_layer(lstm_out)
            
            # Ensure positive dt
            output = torch.cat([
                output[:, :, :2],  # dx, dy
                F.softplus(output[:, :, 2:3]) + 0.001  # dt (positive, min 1ms)
            ], dim=-1)
            
            outputs.append(output)
            
            # Update current position tracking
            current_pos = current_pos + output[:, 0, :2]
            
            # Determine next input (teacher forcing or generated)
            if target_sequences is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current_input = target_sequences[:, t:t+1, :]
                # When teacher forcing, also update position from target
                current_pos = start.clone()
                for i in range(t + 1):
                    current_pos = current_pos + target_sequences[:, i, :2]
            else:
                current_input = output
        
        outputs = torch.cat(outputs, dim=1)
        
        if target_lengths is not None:
            lengths = target_lengths
        else:
            lengths = torch.full((batch_size,), max_len, device=device)
        
        return outputs, lengths
    
    def generate(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate trajectories with early stopping.
        
        Stops when accumulated position is within threshold of end point.
        """
        batch_size = start.shape[0]
        device = start.device
        
        if z is None:
            z = torch.randn(batch_size, self.config.latent_dim, device=device)
        
        # Compute conditioning
        condition = self._compute_condition(start, end, z)
        
        # Initialize
        h = torch.zeros(
            self.config.generator_num_layers,
            batch_size,
            self.config.generator_hidden_dim,
            device=device
        )
        c = torch.zeros_like(h)
        
        outputs = []
        current_input = self.initial_input.expand(batch_size, 1, 3)
        current_pos = start.clone()
        
        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for t in range(self.config.max_generation_steps):
            # Compute dynamic features (remaining distance to target)
            dynamic_feat = self._compute_dynamic_features(current_pos, end)
            
            # Concatenate input with condition AND dynamic features
            lstm_input = torch.cat([
                current_input,
                condition.unsqueeze(1),
                dynamic_feat.unsqueeze(1)
            ], dim=-1)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Generate output
            output = self.output_layer(lstm_out)
            output = torch.cat([
                output[:, :, :2],
                F.softplus(output[:, :, 2:3]) + 0.001
            ], dim=-1)
            
            outputs.append(output)
            current_input = output
            
            # Update position
            current_pos = current_pos + output[:, 0, :2]
            
            # Check if reached target
            distance_to_end = torch.sqrt(((current_pos - end) ** 2).sum(dim=-1))
            newly_done = distance_to_end < self.config.distance_threshold
            
            # Update lengths for sequences that just finished
            lengths = torch.where(
                newly_done & ~done,
                torch.tensor(t + 1, device=device),
                lengths
            )
            done = done | newly_done
            
            if done.all():
                break
        
        # Set lengths for sequences that didn't finish
        lengths = torch.where(lengths == 0, torch.tensor(t + 1, device=device), lengths)
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs, lengths


# =============================================================================
# Discriminator Network
# =============================================================================

class Discriminator(nn.Module):
    """
    Bidirectional LSTM discriminator with kinematic features.
    
    Takes trajectory with computed kinematics and outputs Wasserstein critic score.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Input: 9 kinematic features
        input_dim = 9
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, config.discriminator_hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(config.discriminator_hidden_dim, config.discriminator_hidden_dim)),
            nn.LeakyReLU(0.2)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.discriminator_hidden_dim,
            hidden_size=config.discriminator_hidden_dim,
            num_layers=config.discriminator_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if config.discriminator_num_layers > 1 else 0
        )
        
        # Output layers with spectral normalization
        self.output_layers = nn.Sequential(
            spectral_norm(nn.Linear(config.discriminator_hidden_dim * 2, config.discriminator_hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(config.discriminator_hidden_dim, config.discriminator_hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(config.discriminator_hidden_dim // 2, 1))
        )
    
    def forward(
        self,
        trajectories: torch.Tensor,
        dts: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            trajectories: (batch, seq_len, 2) absolute positions
            dts: (batch, seq_len) time deltas
            lengths: (batch,) actual sequence lengths
        
        Returns:
            scores: (batch, 1) Wasserstein critic scores (no sigmoid)
        """
        # Compute kinematic features
        features = compute_kinematics(trajectories, dts)
        
        # Encode features
        batch_size, seq_len, _ = features.shape
        encoded = self.feature_encoder(features)
        
        # Pack for LSTM
        packed = pack_padded_sequence(
            encoded,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward
        packed_out, (h, c) = self.lstm(packed)
        
        # Use final hidden states from both directions
        # h shape: (num_layers * 2, batch, hidden_dim)
        h_forward = h[-2]  # Last layer forward
        h_backward = h[-1]  # Last layer backward
        h_combined = torch.cat([h_forward, h_backward], dim=-1)
        
        # Output score
        score = self.output_layers(h_combined)
        
        return score


# =============================================================================
# Training
# =============================================================================

def compute_gradient_penalty(
    discriminator: Discriminator,
    real_trajectories: torch.Tensor,
    fake_trajectories: torch.Tensor,
    real_dts: torch.Tensor,
    fake_dts: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    """
    batch_size = real_trajectories.shape[0]
    
    # Ensure same sequence length
    min_len = min(real_trajectories.shape[1], fake_trajectories.shape[1])
    real_trajectories = real_trajectories[:, :min_len]
    fake_trajectories = fake_trajectories[:, :min_len]
    real_dts = real_dts[:, :min_len]
    fake_dts = fake_dts[:, :min_len]
    lengths = lengths.clamp(max=min_len)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha_dt = alpha.squeeze(-1)
    
    interpolated = alpha * real_trajectories + (1 - alpha) * fake_trajectories
    interpolated_dts = alpha_dt * real_dts + (1 - alpha_dt) * fake_dts
    interpolated.requires_grad_(True)
    
    # Discriminator output
    # Disable CuDNN for RNNs to allow double backwards (required for gradient penalty)
    if device.type == 'cuda':
        with torch.backends.cudnn.flags(enabled=False):
            d_interpolated = discriminator(interpolated, interpolated_dts, lengths)
    else:
        d_interpolated = discriminator(interpolated, interpolated_dts, lengths)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


class Trainer:
    """Training manager with logging, checkpointing, and early stopping."""
    
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        config: Config,
        device: torch.device,
        log_dir: str = "runs"
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.device = device
        self.use_amp = bool(self.config.use_amp and self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=config.learning_rate,
            betas=config.betas
        )
        self.d_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=config.learning_rate,
            betas=config.betas
        )
        
        # Schedulers
        self.g_scheduler = ReduceLROnPlateau(
            self.g_optimizer,
            mode='min',
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.min_lr
        )
        self.d_scheduler = ReduceLROnPlateau(
            self.d_optimizer,
            mode='min',
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.min_lr
        )
        
        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"trajectory_gan_{timestamp}"))
        
        # Early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.epoch = 0
        self.d_step = 0
    
    def get_teacher_forcing_ratio(self) -> float:
        """Compute teacher forcing ratio with decay."""
        if self.epoch >= self.config.teacher_forcing_decay_epochs:
            return self.config.teacher_forcing_end
        
        progress = self.epoch / self.config.teacher_forcing_decay_epochs
        return self.config.teacher_forcing_start - progress * (
            self.config.teacher_forcing_start - self.config.teacher_forcing_end
        )
    
    def train_discriminator_step(
        self,
        real_batch: Dict
    ) -> Dict[str, float]:
        """Single discriminator training step."""
        self.d_optimizer.zero_grad()
        
        batch_size = real_batch['starts'].shape[0]
        
        # Real data
        starts = real_batch['starts'].to(self.device)
        ends = real_batch['ends'].to(self.device)
        sequences = real_batch['sequences'].to(self.device)
        lengths = real_batch['lengths'].to(self.device)
        
        # Convert to absolute positions for discriminator
        real_deltas = sequences[:, :, :2]
        real_dts = sequences[:, :, 2]
        real_trajectories = trajectory_to_absolute(starts, real_deltas)
        
        # Pad dts to match trajectory length (add dt=0 for start)
        real_dts_padded = F.pad(real_dts, (1, 0), value=0.001)
        
        # Generate fake data
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        with torch.no_grad():
            fake_sequences, fake_lengths = self.generator(
                starts, ends, z,
                target_sequences=sequences,
                target_lengths=lengths,
                teacher_forcing_ratio=0.0
            )
        
        fake_deltas = fake_sequences[:, :, :2]
        fake_dts = fake_sequences[:, :, 2]
        fake_trajectories = trajectory_to_absolute(starts, fake_deltas)
        fake_dts_padded = F.pad(fake_dts, (1, 0), value=0.001)
        
        lengths_plus_one = lengths + 1
        fake_lengths_plus_one = fake_lengths + 1
        
        autocast_ctx = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()
        with autocast_ctx:
            # Discriminator scores
            real_score = self.discriminator(
                real_trajectories, real_dts_padded, lengths_plus_one
            )
            fake_score = self.discriminator(
                fake_trajectories, fake_dts_padded, fake_lengths_plus_one
            )
            
            # Wasserstein loss
            d_loss = fake_score.mean() - real_score.mean()
        
        # Gradient penalty (computed every N steps on a subset of the batch)
        gp = torch.tensor(0.0, device=self.device)
        gp_every_n = max(1, self.config.gp_every_n)
        do_gp = (self.d_step % gp_every_n == 0)
        self.d_step += 1
        
        if do_gp:
            gp_batch_frac = min(max(self.config.gp_batch_frac, 0.0), 1.0)
            gp_batch_size = max(1, int(math.ceil(batch_size * gp_batch_frac)))
            
            if gp_batch_size < batch_size:
                idx = torch.randperm(batch_size, device=self.device)[:gp_batch_size]
                gp_real_traj = real_trajectories.index_select(0, idx)
                gp_fake_traj = fake_trajectories.index_select(0, idx)
                gp_real_dts = real_dts_padded.index_select(0, idx)
                gp_fake_dts = fake_dts_padded.index_select(0, idx)
                gp_lengths = lengths_plus_one.index_select(0, idx)
            else:
                gp_real_traj = real_trajectories
                gp_fake_traj = fake_trajectories
                gp_real_dts = real_dts_padded
                gp_fake_dts = fake_dts_padded
                gp_lengths = lengths_plus_one
            
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    gp = compute_gradient_penalty(
                        self.discriminator,
                        gp_real_traj,
                        gp_fake_traj,
                        gp_real_dts,
                        gp_fake_dts,
                        gp_lengths,
                        self.device
                    )
            else:
                gp = compute_gradient_penalty(
                    self.discriminator,
                    gp_real_traj,
                    gp_fake_traj,
                    gp_real_dts,
                    gp_fake_dts,
                    gp_lengths,
                    self.device
                )
        
        # Total loss
        total_loss = d_loss + self.config.gradient_penalty_weight * gp
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.d_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'd_gp': gp.item(),
            'd_real_score': real_score.mean().item(),
            'd_fake_score': fake_score.mean().item()
        }
    
    def train_generator_step(
        self,
        real_batch: Dict
    ) -> Dict[str, float]:
        """Single generator training step."""
        self.g_optimizer.zero_grad()
        
        batch_size = real_batch['starts'].shape[0]
        
        starts = real_batch['starts'].to(self.device)
        ends = real_batch['ends'].to(self.device)
        sequences = real_batch['sequences'].to(self.device)
        lengths = real_batch['lengths'].to(self.device)
        
        # Generate
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        tf_ratio = self.get_teacher_forcing_ratio()
        
        autocast_ctx = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()
        with autocast_ctx:
            fake_sequences, fake_lengths = self.generator(
                starts, ends, z,
                target_sequences=sequences,
                target_lengths=lengths,
                teacher_forcing_ratio=tf_ratio
            )
            
            fake_deltas = fake_sequences[:, :, :2]
            fake_dts = fake_sequences[:, :, 2]
            fake_trajectories = trajectory_to_absolute(starts, fake_deltas)
            fake_dts_padded = F.pad(fake_dts, (1, 0), value=0.001)
            
            # Discriminator score
            fake_score = self.discriminator(
                fake_trajectories, fake_dts_padded, fake_lengths + 1
            )
            
            # Generator loss (maximize discriminator score)
            g_loss = -fake_score.mean()
            
            # Endpoint loss: encourage reaching target (stronger weight)
            final_pos = fake_trajectories[torch.arange(batch_size, device=self.device), fake_lengths]
            endpoint_loss = F.mse_loss(final_pos, ends)
            
            # Direction consistency loss: penalize moving away from target
            # Compute target direction from start to end
            target_direction = ends - starts  # (batch, 2)
            target_direction = target_direction / (torch.sqrt((target_direction ** 2).sum(dim=-1, keepdim=True)) + 1e-8)
            
            # Compute movement directions (normalized)
            movement_norms = torch.sqrt((fake_deltas ** 2).sum(dim=-1, keepdim=True) + 1e-8)
            movement_directions = fake_deltas / movement_norms  # (batch, seq, 2)
            
            # Cosine similarity between movement and target direction
            # We want movements to generally align with target direction
            target_direction_expanded = target_direction.unsqueeze(1)  # (batch, 1, 2)
            cosine_sim = (movement_directions * target_direction_expanded).sum(dim=-1)  # (batch, seq)
            
            # Create mask for valid timesteps
            max_len = fake_deltas.shape[1]
            mask = torch.arange(max_len, device=self.device).unsqueeze(0) < fake_lengths.unsqueeze(1)
            
            # Direction loss: penalize negative cosine similarity (moving away from target)
            # We use 1 - cosine_sim so that moving toward target (cosine_sim=1) gives loss=0
            direction_loss = ((1 - cosine_sim) * mask.float()).sum() / mask.float().sum()
            
            # Total loss with configurable weights
            total_loss = (
                g_loss + 
                self.config.endpoint_loss_weight * endpoint_loss + 
                self.config.direction_loss_weight * direction_loss
            )
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'g_endpoint_loss': endpoint_loss.item(),
            'g_direction_loss': direction_loss.item(),
            'g_fake_score': fake_score.mean().item(),
            'teacher_forcing_ratio': tf_ratio
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'd_loss': [], 'd_gp': [], 'd_real_score': [], 'd_fake_score': [],
            'g_loss': [], 'g_endpoint_loss': [], 'g_direction_loss': [], 'g_fake_score': []
        }
        
        for batch_idx, batch in enumerate(dataloader):
            # Train discriminator
            for _ in range(self.config.n_critic):
                d_metrics = self.train_discriminator_step(batch)
                for k, v in d_metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)
            
            # Train generator
            g_metrics = self.train_generator_step(batch)
            for k, v in g_metrics.items():
                if k in epoch_metrics:
                    epoch_metrics[k].append(v)
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items() if v}
        avg_metrics['teacher_forcing_ratio'] = self.get_teacher_forcing_ratio()
        
        return avg_metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            self.writer.add_scalar(f"train/{name}", value, step)
        
        # Log learning rates
        self.writer.add_scalar(
            "train/g_lr",
            self.g_optimizer.param_groups[0]['lr'],
            step
        )
        self.writer.add_scalar(
            "train/d_lr",
            self.d_optimizer.param_groups[0]['lr'],
            step
        )
    
    def log_trajectories(self, dataloader: DataLoader, step: int) -> None:
        """Log sample trajectory visualizations."""
        self.generator.eval()
        
        with torch.no_grad():
            batch = next(iter(dataloader))
            starts = batch['starts'][:4].to(self.device)
            ends = batch['ends'][:4].to(self.device)
            real_sequences = batch['sequences'][:4].to(self.device)
            lengths = batch['lengths'][:4]
            
            z = torch.randn(4, self.config.latent_dim, device=self.device)
            fake_sequences, fake_lengths = self.generator.generate(starts, ends, z)
            
            # Convert to absolute
            real_traj = trajectory_to_absolute(starts, real_sequences[:, :, :2])
            fake_traj = trajectory_to_absolute(starts, fake_sequences[:, :, :2])
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                if i < 4:
                    real_len = lengths[i].item() + 1
                    fake_len = fake_lengths[i].item() + 1
                    
                    real_np = real_traj[i, :real_len].cpu().numpy()
                    fake_np = fake_traj[i, :fake_len].cpu().numpy()
                    
                    ax.plot(real_np[:, 0], real_np[:, 1], 'b-', label='Real', alpha=0.7)
                    ax.plot(fake_np[:, 0], fake_np[:, 1], 'r--', label='Generated', alpha=0.7)
                    ax.scatter([starts[i, 0].cpu()], [starts[i, 1].cpu()], c='green', s=100, marker='o', label='Start')
                    ax.scatter([ends[i, 0].cpu()], [ends[i, 1].cpu()], c='red', s=100, marker='x', label='End')
                    ax.legend()
                    ax.set_title(f'Trajectory {i+1}')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.invert_yaxis()
            
            plt.tight_layout()
            self.writer.add_figure('trajectories/comparison', fig, step)
            plt.close(fig)
        
        self.generator.train()
    
    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.epoch = checkpoint['epoch']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str = "checkpoints"
    ) -> None:
        """Full training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Dataset size: {len(dataloader.dataset)}")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train epoch
            metrics = self.train_epoch(dataloader)
            
            # Log metrics
            self.log_metrics(metrics, epoch)
            
            # Log trajectories periodically
            if epoch % 10 == 0:
                self.log_trajectories(dataloader, epoch)
            
            # Update schedulers
            combined_loss = metrics['d_loss'] + metrics['g_loss']
            self.g_scheduler.step(metrics['g_loss'])
            self.d_scheduler.step(metrics['d_loss'])
            
            # Check for improvement
            if combined_loss < self.best_loss:
                self.best_loss = combined_loss
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
                is_best = False
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path, is_best)
            
            # Print progress
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"D Loss: {metrics['d_loss']:.4f} | "
                f"G Loss: {metrics['g_loss']:.4f} | "
                f"Endpoint: {metrics['g_endpoint_loss']:.4f} | "
                f"TF: {metrics['teacher_forcing_ratio']:.2f}"
            )
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.writer.close()
        print("Training complete!")


# =============================================================================
# Evaluation and Visualization
# =============================================================================

def evaluate_fitts_compliance(
    trajectories: torch.Tensor,
    dts: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    target_width: float = 0.02
) -> Dict[str, float]:
    """
    Evaluate Fitts' Law compliance of trajectories.
    
    Fitts' Law: MT = a + b * log2(D/W + 1)
    
    Returns correlation and other statistics.
    """
    batch_size = trajectories.shape[0]
    
    # Compute movement times
    movement_times = dts.sum(dim=-1).cpu().numpy()
    
    # Compute distances
    distances = torch.sqrt(((ends - starts) ** 2).sum(dim=-1)).cpu().numpy()
    
    # Index of difficulty
    index_of_difficulty = np.log2(distances / target_width + 1)
    
    # Correlation
    correlation = np.corrcoef(index_of_difficulty, movement_times)[0, 1]
    
    return {
        'fitts_correlation': correlation,
        'mean_movement_time': movement_times.mean(),
        'mean_distance': distances.mean(),
        'mean_id': index_of_difficulty.mean()
    }


def visualize_trajectory(
    real_trajectory: np.ndarray,
    fake_trajectory: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize real vs generated trajectory.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Trajectory plot
    ax = axes[0]
    ax.plot(real_trajectory[:, 0], real_trajectory[:, 1], 'b-', label='Real', linewidth=2)
    ax.plot(fake_trajectory[:, 0], fake_trajectory[:, 1], 'r--', label='Generated', linewidth=2)
    ax.scatter([start[0]], [start[1]], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter([end[0]], [end[1]], c='red', s=100, marker='x', label='End', zorder=5)
    ax.legend()
    ax.set_title('Trajectory Comparison')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_yaxis()
    
    # Velocity profile
    ax = axes[1]
    real_vel = np.sqrt(np.diff(real_trajectory[:, 0])**2 + np.diff(real_trajectory[:, 1])**2)
    fake_vel = np.sqrt(np.diff(fake_trajectory[:, 0])**2 + np.diff(fake_trajectory[:, 1])**2)
    ax.plot(real_vel, 'b-', label='Real')
    ax.plot(fake_vel, 'r--', label='Generated')
    ax.legend()
    ax.set_title('Velocity Profile')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Velocity')
    
    # Curvature
    ax = axes[2]
    # Simplified curvature approximation
    def compute_curvature(traj):
        if len(traj) < 3:
            return np.array([0])
        v = np.diff(traj, axis=0)
        a = np.diff(v, axis=0)
        cross = np.abs(v[:-1, 0] * a[:, 1] - v[:-1, 1] * a[:, 0])
        speed = np.sqrt(v[:-1, 0]**2 + v[:-1, 1]**2) + 1e-8
        return cross / speed**3
    
    real_curv = compute_curvature(real_trajectory)
    fake_curv = compute_curvature(fake_trajectory)
    ax.plot(real_curv, 'b-', label='Real')
    ax.plot(fake_curv, 'r--', label='Generated')
    ax.legend()
    ax.set_title('Curvature Profile')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Curvature')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close(fig)


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Recurrent GAN for Mouse Trajectory Generation'
    )
    
    # Mode
    parser.add_argument(
        '--generate', action='store_true',
        help='Generate trajectories instead of training'
    )
    
    # Data
    parser.add_argument(
        '--data_path', type=str,
        default='../../data/',
        help='Path to training data directory or CSV file'
    )
    
    # Training
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard log directory')
    
    # Generation
    parser.add_argument(
        '--start', type=str, default='100,500',
        help='Start point as x,y (e.g., "100,500")'
    )
    parser.add_argument(
        '--end', type=str, default='800,300',
        help='End point as x,y (e.g., "800,300")'
    )
    parser.add_argument(
        '--num_samples', type=int, default=5,
        help='Number of trajectories to generate'
    )
    parser.add_argument(
        '--output', type=str, default='generated_trajectory.png',
        help='Output path for generated trajectory visualization'
    )
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Generator hidden dimension')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config
    config = Config()
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.epochs = args.epochs
    config.latent_dim = args.latent_dim
    config.generator_hidden_dim = args.hidden_dim
    
    if args.generate:
        # Generation mode
        if args.checkpoint is None:
            print("Error: --checkpoint required for generation mode")
            return
        
        # Load model
        generator = Generator(config)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.to(device)
        generator.eval()
        
        # Parse start/end points
        start = torch.tensor(
            [float(x) for x in args.start.split(',')],
            dtype=torch.float32
        ).unsqueeze(0) / torch.tensor([config.screen_width, config.screen_height])
        
        end = torch.tensor(
            [float(x) for x in args.end.split(',')],
            dtype=torch.float32
        ).unsqueeze(0) / torch.tensor([config.screen_width, config.screen_height])
        
        start = start.to(device)
        end = end.to(device)
        
        # Generate trajectories
        print(f"Generating {args.num_samples} trajectories...")
        
        with torch.no_grad():
            for i in range(args.num_samples):
                z = torch.randn(1, config.latent_dim, device=device)
                sequences, lengths = generator.generate(start, end, z)
                
                # Convert to absolute
                trajectory = trajectory_to_absolute(start, sequences[:, :, :2])
                trajectory = trajectory[0, :lengths[0]+1].cpu().numpy()
                
                # Denormalize
                trajectory = trajectory * np.array([config.screen_width, config.screen_height])
                
                print(f"Trajectory {i+1}: {len(trajectory)} points")
                print(f"  Start: ({trajectory[0, 0]:.1f}, {trajectory[0, 1]:.1f})")
                print(f"  End: ({trajectory[-1, 0]:.1f}, {trajectory[-1, 1]:.1f})")
                
                # Visualize
                if i == 0:
                    start_np = start[0].cpu().numpy() * np.array([config.screen_width, config.screen_height])
                    end_np = end[0].cpu().numpy() * np.array([config.screen_width, config.screen_height])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
                    ax.scatter([start_np[0]], [start_np[1]], c='green', s=100, marker='o', label='Start')
                    ax.scatter([end_np[0]], [end_np[1]], c='red', s=100, marker='x', label='End')
                    ax.legend()
                    ax.set_title('Generated Mouse Trajectory')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.invert_yaxis()
                    ax.set_xlim(0, config.screen_width)
                    ax.set_ylim(config.screen_height, 0)
                    plt.savefig(args.output, dpi=150, bbox_inches='tight')
                    print(f"Saved visualization to {args.output}")
                    plt.close()
    
    else:
        # Training mode
        print("Loading dataset...")
        dataset = MouseTrajectoryDataset(args.data_path, config)
        
        if len(dataset) == 0:
            print("Error: No trajectories loaded. Check data path.")
            return
        
        num_workers = config.num_workers
        pin_memory = config.pin_memory and device.type == 'cuda'
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_trajectories,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=True
        )
        
        # Create models
        generator = Generator(config)
        discriminator = Discriminator(config)
        
        print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
        
        # Create trainer
        trainer = Trainer(generator, discriminator, config, device, args.log_dir)
        
        # Load checkpoint if provided
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Train
        trainer.train(dataloader, config.epochs, args.checkpoint_dir)


if __name__ == '__main__':
    main()

"""
Configuration file for dataset generation (HARDER VERSION FOR ICML WORKSHOP)
Significantly increased complexity while maintaining same format
"""

# Dataset parameters
DATASET_CONFIG = {
    # Output directory
    "output_dir": "./physics_dataset",
    
    # Increased episodes for better coverage of complex scenarios
    "num_episodes": 1000,  # Up from 500
    
    # Same split ratios (format compatibility)
    "split_ratios": {
        "train": 0.6,
        "test_compositional": 0.2,
        "test_contradiction": 0.1,
        "control_ablation": 0.1,
    },
    
    # HARDER: More blocks = exponentially more interactions
    "num_blocks_min": 3,  # Up from 2
    "num_blocks_max": 6,  # Up from 4 - this is the key difficulty increase
    
    # HARDER: Longer trajectories with more complex dynamics
    "num_steps": 200,  # Up from 100 - allows more interactions to develop
    "dt": 0.01,
    "save_every_n_steps": 5,
    
    # Reproducibility
    "seed": 42,
}

# Simulation parameters
SIMULATION_CONFIG = {
    "headless": True,
    "render_width": 256,
    "render_height": 256,
}

# HARDER: More diverse and subtle physics rules
PHYSICS_RULES = {
    # Training rules - now with more subtle variations
    "train": [
        "red_heavy",
        "blue_heavy",
        "green_light",
        "red_slippery",
        "green_slippery",
        # NEW: Additional training rules for complexity
        "yellow_medium_heavy",
        "cyan_very_slippery",
        "magenta_sticky",
    ],
    
    # Compositional test rules
    "compositional": [
        "red_light",
        "blue_slippery",
        # NEW: More compositional challenges
        "yellow_light",
        "cyan_sticky",
        "magenta_slippery",
    ],
    
    # Contradiction test rules
    "contradiction": [
        "blue_light",
        # NEW: More contradictions
        "red_sticky",
        "green_heavy",
    ],
}

# HARDER: More gravity variations with extreme cases
GRAVITY_VARIATIONS = [
    "normal", 
    "reverse", 
    "low",
    # NEW: Challenging gravity scenarios
    "high",       # 2x normal gravity
    "micro",      # Very low gravity (0.1x)
    "sideways_x", # Horizontal gravity
    "sideways_y", # Horizontal gravity (different axis)
]

# HARDER: Larger spawn area with more varied initial conditions
BLOCK_CONFIG = {
    "x_range": (-2.0, 2.0),  # Wider from (-1.0, 1.0)
    "y_range": (-2.0, 2.0),  # Wider from (-1.0, 1.0)
    "z_range": (0.5, 4.0),   # Higher from (1.0, 3.0) - allows longer falls
    "scale_range": (0.08, 0.35),  # More size variation from (0.1, 0.3)
    
    # NEW: Initial velocity range for dynamic starts
    "initial_velocity_range": (-0.5, 0.5),  # Random initial velocities
    "initial_angular_velocity": True,  # Add rotation
}

# State vector configuration (UNCHANGED - format compatibility)
STATE_CONFIG = {
    "include_color": False,
    "state_dim_per_block": 9,
}

# Data format configuration (UNCHANGED)
DATA_FORMAT = {
    "format": "npz",
    "compression": True,
}

# Evaluation metrics configuration (UNCHANGED)
EVALUATION_CONFIG = {
    "check_velocity_ordering": True,
    "check_acceleration_direction": True,
    "track_by_split": True,
    "test_contradictions": True,
}
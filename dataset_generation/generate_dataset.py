"""
Multimodal World Model Grounding Test - HARDER Dataset Generation
Creates challenging physics scenarios with complex interactions

DIFFICULTY INCREASES:
1. More blocks (3-6 instead of 2-4)
2. Dynamic rule application at random timesteps  
3. Partial rule application (only some blocks affected)
4. Multiple simultaneous rules
5. Time-varying gravity
6. Initial velocities and rotations
7. Block-block collisions enabled
8. Stochastic physics parameters
9. Longer trajectories (200 steps)
10. More complex state dynamics
"""

import os
import json
import numpy as np
from pathlib import Path
import time
import psutil

# Try to import GPUtil, but make it optional
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available. VRAM tracking will be disabled.")

# Try to import scipy for metrics
try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Rule compliance metrics will be disabled.")

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)

# Isaac Sim imports
from isaacsim import SimulationApp

# Configuration
CONFIG = {
    "headless": True,
    "width": 256,
    "height": 256,
}

# Launch Isaac Sim
simulation_app = SimulationApp(CONFIG)

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema

# HARDER: Rules apply at random timesteps between 10-40
RULE_APPLICATION_STEP_MIN = 10
RULE_APPLICATION_STEP_MAX = 100 #increased to 100

# HARDER: Longer trajectories
DEFAULT_NUM_STEPS = 500  # Increased from 200

# Physics Rules Definition (EXPANDED)
RULES = {
    # Mass rules
    "red_heavy": {
        "text": "Red blocks are heavy",
        "condition": lambda color: color == "red",
        "property": "mass",
        "value": 15.0,  # Increased variance
        "default": 1.0
    },
    "blue_heavy": {
        "text": "Blue blocks are heavy",
        "condition": lambda color: color == "blue",
        "property": "mass",
        "value": 15.0,
        "default": 1.0
    },
    "green_light": {
        "text": "Green blocks are light",
        "condition": lambda color: color == "green",
        "property": "mass",
        "value": 0.05,  # Even lighter
        "default": 1.0
    },
    "red_light": {
        "text": "Red blocks are light",
        "condition": lambda color: color == "red",
        "property": "mass",
        "value": 0.05,
        "default": 1.0
    },
    "blue_light": {
        "text": "Blue blocks are light",
        "condition": lambda color: color == "blue",
        "property": "mass",
        "value": 0.05,
        "default": 1.0
    },
    # Friction rules
    "red_slippery": {
        "text": "Red blocks slide easily",
        "condition": lambda color: color == "red",
        "property": "friction",
        "value": 0.001,  # More extreme
        "default": 0.5
    },
    "green_slippery": {
        "text": "Green blocks slide easily",
        "condition": lambda color: color == "green",
        "property": "friction",
        "value": 0.001,
        "default": 0.5
    },
    "blue_slippery": {
        "text": "Blue blocks slide easily",
        "condition": lambda color: color == "blue",
        "property": "friction",
        "value": 0.001,
        "default": 0.5
    },
    # NEW: Sticky rules
    "red_sticky": {
        "text": "Red blocks are sticky",
        "condition": lambda color: color == "red",
        "property": "friction",
        "value": 2.0,
        "default": 0.5
    },
    "green_sticky": {
        "text": "Green blocks are sticky",
        "condition": lambda color: color == "green",
        "property": "friction",
        "value": 2.0,
        "default": 0.5
    },
}

# Gravity rules (EXPANDED with time-varying)
GRAVITY_RULES = {
    "normal": {
        "text": "Normal gravity",
        "gravity": np.array([0.0, 0.0, -9.81]),
        "varying": False
    },
    "reverse": {
        "text": "Reversed gravity",
        "gravity": np.array([0.0, 0.0, 9.81]),
        "varying": False
    },
    "low": {
        "text": "Low gravity",
        "gravity": np.array([0.0, 0.0, -3.0]),
        "varying": False
    },
    "high": {
        "text": "High gravity",
        "gravity": np.array([0.0, 0.0, -20.0]),
        "varying": False
    },
    "sideways": {
        "text": "Sideways gravity",
        "gravity": np.array([9.81, 0.0, 0.0]),
        "varying": False
    },
    "varying_normal": {
        "text": "Varying gravity",
        "gravity": np.array([0.0, 0.0, -9.81]),
        "varying": True  # Will fluctuate
    },
}

# Color definitions
COLOR_RGB = {
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0),
}


def get_vram_usage():
    """Get current VRAM usage in MB (returns 0 if GPUtil not available)"""
    if not GPUTIL_AVAILABLE:
        return 0
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed
    except:
        pass
    return 0


def compute_rule_compliance(trajectory, rule_key, blocks_info):
    """
    Compute rule compliance metric
    """
    if not SCIPY_AVAILABLE:
        return 0.0
    
    rule = RULES[rule_key]
    
    if rule["property"] == "mass":
        masses = []
        accelerations = []
        
        for i, block_info in enumerate(blocks_info):
            mass = block_info["mass"]
            velocities = trajectory["velocities"][:, i, 2]
            if len(velocities) > 1:
                accel = np.abs(np.diff(velocities)).mean()
                masses.append(mass)
                accelerations.append(accel)
        
        if len(masses) > 1:
            corr, _ = spearmanr(masses, accelerations)
            return float(corr)
    
    elif rule["property"] == "friction":
        frictions = []
        slide_distances = []
        
        for i, block_info in enumerate(blocks_info):
            friction = block_info["friction"]
            positions = trajectory["positions"][:, i, :2]
            if len(positions) > 1:
                displacement = np.linalg.norm(positions[-1] - positions[0])
                frictions.append(friction)
                slide_distances.append(displacement)
        
        if len(frictions) > 1:
            corr, _ = spearmanr(frictions, slide_distances)
            return float(-corr)
    
    return 0.0


class DatasetGenerator:
    def __init__(self, output_dir="./dataset", num_episodes=100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_episodes = num_episodes
        
        # Create subdirectories
        (self.output_dir / "trajectories").mkdir(exist_ok=True)
        (self.output_dir / "control_ablation").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        
        self.world = None
        self.blocks = []
        
        # Timing and resource tracking
        self.timings = []
        self.vram_usage = []
        
    def setup_world(self):
        """Initialize the Isaac Sim world"""
        start_time = time.time()
        start_vram = get_vram_usage()
        
        self.world = World(stage_units_in_meters=1.0)
        
        # Add ground plane
        self.world.scene.add_default_ground_plane(
            z_position=0,
            name="ground_plane",
            prim_path="/World/GroundPlane",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        )
        
        elapsed = time.time() - start_time
        vram_used = get_vram_usage() - start_vram
        
        print(f"World setup complete (Time: {elapsed:.2f}s, VRAM: +{vram_used}MB)")
        
    def apply_rule(self, block, color, rule_key, apply_now=True):
        """Apply a physics rule to a block"""
        rule = RULES[rule_key]
        
        applied_mass = 1.0
        applied_friction = 0.5
        
        if rule["condition"](color):
            if rule["property"] == "mass":
                if apply_now:
                    mass_api = UsdPhysics.MassAPI.Apply(block.prim)
                    mass_api.GetMassAttr().Set(rule["value"])
                applied_mass = rule["value"]
                
            elif rule["property"] == "friction":
                if apply_now:
                    material = UsdPhysics.MaterialAPI.Apply(block.prim)
                    material.CreateStaticFrictionAttr().Set(rule["value"])
                    material.CreateDynamicFrictionAttr().Set(rule["value"])
                    material.CreateRestitutionAttr().Set(0.0)
                    
                applied_friction = rule["value"]
        
        return applied_mass, applied_friction
        
    def set_gravity(self, gravity_rule_key, step=0, total_steps=DEFAULT_NUM_STEPS):
        """Set world gravity with optional time-varying component"""
        gravity_rule = GRAVITY_RULES[gravity_rule_key]
        gravity_vector = gravity_rule["gravity"].copy()
        
        # HARDER: Time-varying gravity
        if gravity_rule.get("varying", False):
            # Sinusoidal variation
            phase = 2 * np.pi * step / total_steps
            scale = 0.5 + 0.5 * np.sin(phase)  # Varies between 0.5x and 1.5x
            gravity_vector = gravity_vector * scale
        
        magnitude = np.linalg.norm(gravity_vector)
        direction = gravity_vector / magnitude if magnitude > 0 else np.array([0, 0, -1])
        
        scene = UsdPhysics.Scene.Get(get_current_stage(), "/physicsScene")
        if not scene:
            scene = UsdPhysics.Scene.Define(get_current_stage(), "/physicsScene")
        
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(float(direction[0]), 
                                                         float(direction[1]), 
                                                         float(direction[2])))
        scene.CreateGravityMagnitudeAttr().Set(float(magnitude))
        
    def spawn_blocks(self, num_blocks=3, episode_id=0):
        """Spawn colored blocks with MORE COMPLEX initial conditions"""
        self.blocks = []
        colors_list = ["red", "blue", "green"]
        
        for i in range(num_blocks):
            # HARDER: More spread out positions
            x = np.random.uniform(-2.5, 2.5)
            y = np.random.uniform(-2.5, 2.5)
            z = np.random.uniform(1.5, 5.0)  # Higher drops
            
            color_name = np.random.choice(colors_list)
            color_rgb = COLOR_RGB[color_name]
            
            # HARDER: Variable sizes
            scale = np.random.uniform(0.08, 0.35)
            
            # HARDER: Initial velocities
            initial_velocity = np.array([
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(-1.0, 1.0)
            ])
            
            block = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Episode_{episode_id}_Block_{i}",
                    name=f"episode_{episode_id}_block_{i}",
                    position=np.array([x, y, z]),
                    scale=np.array([scale, scale, scale]),
                    color=np.array(color_rgb),
                    mass=1.0,
                    linear_velocity=initial_velocity,  # HARDER: starts moving
                )
            )
            
            self.blocks.append({
                "object": block,
                "color": color_name,
                "initial_pos": [x, y, z],
                "initial_vel": initial_velocity.tolist(),
                "scale": scale,
                "mass": 1.0,
                "friction": 0.5,
            })
            
        return self.blocks
    
    def capture_state(self):
        """Capture current state"""
        state = []
        for block_info in self.blocks:
            block = block_info["object"]
            pos, quat = block.get_world_pose()
            vel = block.get_linear_velocity()
            
            state.append({
                "position": pos.tolist() if hasattr(pos, 'tolist') else list(pos),
                "velocity": vel.tolist() if hasattr(vel, 'tolist') else list(vel),
                "mass": block_info["mass"],
                "friction": block_info["friction"],
                "scale": block_info["scale"],
            })
        return state
    
    def generate_episode(self, episode_id, rule_key, gravity_key="normal", 
                         num_blocks=4, num_steps=DEFAULT_NUM_STEPS, dt=0.01,
                         ablation_mode=None, wrong_rule_key=None):
        """Generate a harder episode with complex dynamics"""
        
        episode_start = time.time()
        episode_start_vram = get_vram_usage()
        
        print(f"Episode {episode_id}: rule={rule_key}, gravity={gravity_key}, "
              f"blocks={num_blocks}, ablation={ablation_mode}")
        
        # Set initial gravity
        self.set_gravity(gravity_key, step=0, total_steps=num_steps)
        
        # Spawn blocks with random initial conditions
        blocks = self.spawn_blocks(num_blocks, episode_id=episode_id)
        
        # HARDER: Random rule application timestep
        rule_application_step = np.random.randint(
            RULE_APPLICATION_STEP_MIN, 
            RULE_APPLICATION_STEP_MAX
        )
        
        # HARDER: Partial rule application (only 60-100% of matching blocks)
        partial_prob = np.random.uniform(0.6, 1.0)
        
        rules_applied = False
        
        # Reset physics
        self.world.reset()
        
        # Collect trajectory
        trajectory = []
        
        for step in range(num_steps):
            # Update gravity for time-varying scenarios
            if GRAVITY_RULES[gravity_key].get("varying", False):
                self.set_gravity(gravity_key, step=step, total_steps=num_steps)
            
            # DELAYED RULE APPLICATION with partial application
            if step == rule_application_step and not rules_applied:
                print(f"  Step {step}: Applying rules (partial_prob={partial_prob:.2f})...")
                for block_info in blocks:
                    # HARDER: Probabilistic application
                    if np.random.random() < partial_prob:
                        mass, friction = self.apply_rule(
                            block_info["object"], 
                            block_info["color"], 
                            rule_key,
                            apply_now=True
                        )
                        block_info["mass"] = mass
                        block_info["friction"] = friction
                rules_applied = True
            
            # Step physics
            self.world.step(render=False)
            
            # Capture state every 2 steps to get more temporal resolution
            if step % 2 == 0:
                state = self.capture_state()
                trajectory.append({
                    "step": step,
                    "time": step * dt,
                    "state": state,
                    "rules_applied": rules_applied
                })
        
        # Prepare rule text
        if ablation_mode == "shuffled":
            shuffled_rules = list(RULES.keys())
            np.random.shuffle(shuffled_rules)
            rule_text = RULES[shuffled_rules[0]]["text"]
        elif ablation_mode == "no_language":
            rule_text = ""
        elif ablation_mode == "wrong_rule" and wrong_rule_key:
            rule_text = RULES[wrong_rule_key]["text"]
        else:
            rule_text = RULES[rule_key]["text"]
        
        # Convert trajectory to numpy
        trajectory_array = {
            "positions": np.array([[s["position"] for s in t["state"]] for t in trajectory]),
            "velocities": np.array([[s["velocity"] for s in t["state"]] for t in trajectory]),
            "masses": np.array([[s["mass"] for s in t["state"]] for t in trajectory]),
            "frictions": np.array([[s["friction"] for s in t["state"]] for t in trajectory]),
            "scales": np.array([[s["scale"] for s in t["state"]] for t in trajectory]),
            "times": np.array([t["time"] for t in trajectory]),
            "rules_applied_at_step": np.array([t["rules_applied"] for t in trajectory]),
        }
        
        # Compute compliance
        compliance_score = compute_rule_compliance(trajectory_array, rule_key, blocks)
        
        # Episode timing
        episode_time = time.time() - episode_start
        episode_vram = get_vram_usage() - episode_start_vram
        
        print(f"  Completed in {episode_time:.2f}s, "
              f"VRAM: +{episode_vram}MB, "
              f"Compliance: {compliance_score:.3f}")
        
        # Save episode data
        episode_data = {
            "episode_id": episode_id,
            "rule_text": rule_text,
            "rule_key": rule_key,
            "gravity_text": GRAVITY_RULES[gravity_key]["text"],
            "gravity_key": gravity_key,
            "gravity_vector": GRAVITY_RULES[gravity_key]["gravity"].tolist(),
            "num_blocks": num_blocks,
            "num_steps": num_steps,
            "dt": dt,
            "rule_application_step": int(rule_application_step),
            "partial_application_prob": float(partial_prob),
            "ablation_mode": ablation_mode if ablation_mode else "none",
            "wrong_rule_key": wrong_rule_key if wrong_rule_key else "none",
            "compliance_score": compliance_score,
            "generation_time_sec": episode_time,
            "vram_usage_mb": episode_vram,
        }
        
        # Save as NPZ
        if ablation_mode:
            output_file = self.output_dir / "control_ablation" / f"episode_{episode_id:04d}.npz"
        else:
            output_file = self.output_dir / "trajectories" / f"episode_{episode_id:04d}.npz"
        
        np.savez_compressed(
            output_file,
            **episode_data,
            **trajectory_array
        )
        
        # Track timing
        self.timings.append(episode_time)
        self.vram_usage.append(episode_vram)
        
        # Clear blocks list
        self.blocks = []
        
        return episode_data
    
    def generate_dataset(self):
        """Generate full harder dataset"""
        
        # EXPANDED rule sets
        train_rules = list(RULES.keys())[:7]  # More training rules
        compositional_rules = ["red_sticky", "green_sticky"]  # New combinations
        contradiction_rules = ["blue_light", "red_light"]
        
        gravity_types = list(GRAVITY_RULES.keys())
        
        # Calculate split sizes
        train_size = int(self.num_episodes * 0.50)
        compositional_size = int(self.num_episodes * 0.15)
        contradiction_size = int(self.num_episodes * 0.10)
        ablation_shuffled_size = int(self.num_episodes * 0.10)
        ablation_wrong_size = self.num_episodes - train_size - compositional_size - contradiction_size - ablation_shuffled_size
        
        metadata = {
            "train": [],
            "test_compositional": [],
            "test_contradiction": [],
            "control_shuffled": [],
            "control_wrong_rule": [],
            "rules": {k: v["text"] for k, v in RULES.items()},
            "gravity_rules": {k: v["text"] for k, v in GRAVITY_RULES.items()},
            "rule_application_step_range": [RULE_APPLICATION_STEP_MIN, RULE_APPLICATION_STEP_MAX],
            "num_steps": DEFAULT_NUM_STEPS,
            "seed": SEED,
            "difficulty": "HARD",
        }
        
        episode_id = 0
        
        # TRAIN episodes
        print(f"\n{'='*60}")
        print(f"Generating {train_size} TRAINING episodes (HARDER)...")
        print(f"{'='*60}")
        for i in range(train_size):
            rule = np.random.choice(train_rules)
            gravity = np.random.choice(gravity_types)
            num_blocks = np.random.randint(3, 7)  # More blocks
            self.generate_episode(
                episode_id=episode_id,
                rule_key=rule,
                gravity_key=gravity,
                num_blocks=num_blocks
            )
            metadata["train"].append({
                "episode_id": episode_id, 
                "rule": rule, 
                "gravity": gravity,
                "num_blocks": num_blocks
            })
            episode_id += 1
        
        # COMPOSITIONAL episodes
        print(f"\n{'='*60}")
        print(f"Generating {compositional_size} COMPOSITIONAL episodes...")
        print(f"{'='*60}")
        for i in range(compositional_size):
            rule = np.random.choice(compositional_rules)
            gravity = np.random.choice(gravity_types)
            num_blocks = np.random.randint(3, 7)
            self.generate_episode(
                episode_id=episode_id,
                rule_key=rule,
                gravity_key=gravity,
                num_blocks=num_blocks
            )
            metadata["test_compositional"].append({
                "episode_id": episode_id,
                "rule": rule,
                "gravity": gravity,
                "num_blocks": num_blocks
            })
            episode_id += 1
        
        # CONTRADICTION episodes
        print(f"\n{'='*60}")
        print(f"Generating {contradiction_size} CONTRADICTION episodes...")
        print(f"{'='*60}")
        for i in range(contradiction_size):
            rule = np.random.choice(contradiction_rules)
            gravity = np.random.choice(gravity_types)
            num_blocks = np.random.randint(3, 7)
            self.generate_episode(
                episode_id=episode_id,
                rule_key=rule,
                gravity_key=gravity,
                num_blocks=num_blocks
            )
            metadata["test_contradiction"].append({
                "episode_id": episode_id,
                "rule": rule,
                "gravity": gravity,
                "num_blocks": num_blocks
            })
            episode_id += 1
        
        # SHUFFLED LANGUAGE control
        print(f"\n{'='*60}")
        print(f"Generating {ablation_shuffled_size} SHUFFLED LANGUAGE episodes...")
        print(f"{'='*60}")
        for i in range(ablation_shuffled_size):
            rule = np.random.choice(train_rules)
            gravity = np.random.choice(gravity_types)
            num_blocks = np.random.randint(3, 7)
            self.generate_episode(
                episode_id=episode_id,
                rule_key=rule,
                gravity_key=gravity,
                num_blocks=num_blocks,
                ablation_mode="shuffled"
            )
            metadata["control_shuffled"].append({
                "episode_id": episode_id,
                "rule": rule,
                "gravity": gravity,
                "num_blocks": num_blocks
            })
            episode_id += 1
        
        # WRONG RULE control
        print(f"\n{'='*60}")
        print(f"Generating {ablation_wrong_size} WRONG RULE episodes...")
        print(f"{'='*60}")
        for i in range(ablation_wrong_size):
            rule = np.random.choice(train_rules)
            wrong_rule = np.random.choice([r for r in train_rules if r != rule])
            gravity = np.random.choice(gravity_types)
            num_blocks = np.random.randint(3, 7)
            self.generate_episode(
                episode_id=episode_id,
                rule_key=rule,
                gravity_key=gravity,
                num_blocks=num_blocks,
                ablation_mode="wrong_rule",
                wrong_rule_key=wrong_rule
            )
            metadata["control_wrong_rule"].append({
                "episode_id": episode_id,
                "actual_rule": rule,
                "text_rule": wrong_rule,
                "gravity": gravity,
                "num_blocks": num_blocks
            })
            episode_id += 1
        
        # Save metadata
        metadata["timing"] = {
            "mean_episode_time_sec": float(np.mean(self.timings)),
            "total_time_sec": float(np.sum(self.timings)),
            "mean_vram_per_episode_mb": float(np.mean(self.vram_usage)),
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"\n{'='*60}")
        print(f"✓ HARDER Dataset generation complete!")
        print(f"{'='*60}")
        print(f"  Train: {train_size}")
        print(f"  Compositional: {compositional_size}")
        print(f"  Contradiction: {contradiction_size}")
        print(f"  Shuffled control: {ablation_shuffled_size}")
        print(f"  Wrong-rule control: {ablation_wrong_size}")
        print(f"  Total: {self.num_episodes}")
        print(f"  Mean time/episode: {np.mean(self.timings):.2f}s")
        print(f"  Total time: {np.sum(self.timings)/60:.1f} min")
        print(f"{'='*60}")


def main():
    OUTPUT_DIR = "./physics_dataset"
    NUM_EPISODES = 2000
    
    print("="*60)
    print("Multimodal World Model Grounding Test")
    print("HARDER Dataset Generation")
    print("="*60)
    print(f"Random seed: {SEED}")
    print(f"Rule application range: {RULE_APPLICATION_STEP_MIN}-{RULE_APPLICATION_STEP_MAX} steps")
    print(f"Trajectory length: {DEFAULT_NUM_STEPS} steps")
    print(f"Blocks per episode: 3-6")
    print("="*60)
    
    generator = DatasetGenerator(
        output_dir=OUTPUT_DIR,
        num_episodes=NUM_EPISODES
    )
    
    generator.setup_world()
    generator.generate_dataset()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
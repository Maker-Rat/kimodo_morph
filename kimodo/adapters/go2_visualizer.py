# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic MuJoCo motion visualizer for Morph retargeting outputs."""

import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import threading
from pathlib import Path


class RobotMotionVisualizer:
    """Real-time robot motion visualizer using MuJoCo."""
    
    def __init__(self, robot_xml_path: str = "./morph/assets/robots/unitree_go2/go2.xml", fps: int = 30):
        """
        Initialize the visualizer.
        
        Args:
            robot_xml_path: Path to robot XML model file
            fps: Playback frames per second
        """
        self.robot_xml_path = robot_xml_path
        self.fps = fps
        self.dt = 1.0 / fps
        
        self.model = None
        self.data = None
        self.viewer = None
        self.viewer_thread = None
        self.running = False
        
        # Motion data
        self.joint_angles = None
        self.base_pos = None
        self.base_rot = None
        self.actuated_joints = []
        self.motion_joints = []
        
        self.frame = 0
        self.paused = False
        self.loop = True
        
    def load_model(self):
        """Load MuJoCo model"""
        if not Path(self.robot_xml_path).exists():
            raise FileNotFoundError(f"Robot XML not found: {self.robot_xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(self.robot_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Disable gravity for kinematic playback
        self.model.opt.gravity[:] = 0
        
        # Keep actuator info for debugging only.
        self.actuated_joints = []
        for i in range(self.model.nu):
            trnid = self.model.actuator_trnid[i]
            trntype = self.model.actuator_trntype[i]
            
            if trntype == 0:  # mjTRN_JOINT
                joint_id = trnid[0]
                qpos_addr = self.model.jnt_qposadr[joint_id]
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                self.actuated_joints.append((joint_id, qpos_addr, joint_name))

        # Motion should be mapped by joint order (non-free joints), not actuator order.
        self.motion_joints = []
        for joint_id in range(self.model.njnt):
            jtype = self.model.jnt_type[joint_id]
            # Skip free/ball joints and only keep single-dof joints.
            if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qpos_addr = self.model.jnt_qposadr[joint_id]
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                self.motion_joints.append((joint_id, qpos_addr, joint_name))
        print(
            f"[Visualizer] XML={self.robot_xml_path} "
            f"motion_joints={len(self.motion_joints)} actuators={len(self.actuated_joints)}"
        )
        if self.motion_joints:
            preview = [name for _, _, name in self.motion_joints[:10]]
            print(f"[Visualizer] motion joint order preview: {preview}")
    
    def load_motion_from_pkl(self, pkl_path: str):
        """Load motion from pickle file"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.joint_angles = np.array(data['dof_pos'])
        self.base_pos = np.array(data['root_pos'])
        
        # Handle both xyzw and wxyz quaternion formats
        self.base_rot = np.array(data['root_rot'])
        # Assume xyzw format from retargeting (convert to wxyz for MuJoCo)
        if self.base_rot.shape[1] == 4:
            # Convert from xyzw to wxyz
            self.base_rot = np.column_stack([
                self.base_rot[:, 3],  # w
                self.base_rot[:, 0],  # x
                self.base_rot[:, 1],  # y
                self.base_rot[:, 2]   # z
            ])
        
        self.frame = 0
        print(f"✓ Loaded motion: {self.joint_angles.shape[0]} frames, {self.joint_angles.shape[1]} joints")
    
    def _controller(self):
        """Set MuJoCo state from motion data"""
        if self.joint_angles is None:
            return
        
        frame_idx = self.frame % len(self.joint_angles)
        
        # Set base position
        self.data.qpos[0:3] = self.base_pos[frame_idx]
        
        # Set base rotation (wxyz format)
        self.data.qpos[3:7] = self.base_rot[frame_idx]
        
        # Set joint angles by joint order (matches Morph dof_pos semantics).
        for i, (jnt_id, qpos_addr, name) in enumerate(self.motion_joints):
            if i < self.joint_angles.shape[1]:
                self.data.qpos[qpos_addr] = self.joint_angles[frame_idx, i]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def _key_callback(self, key: int):
        """Handle keyboard input"""
        if key == 32:  # Space
            self.paused = not self.paused
        elif key == ord('r'):
            self.frame = 0
        elif key == ord('l'):
            self.loop = not self.loop
    
    def _viewer_loop(self):
        """Main viewer update loop (runs in separate thread)"""
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._key_callback) as viewer:
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -15
            
            self.viewer = viewer
            self.running = True
            
            while viewer.is_running() and self.running:
                # Update state
                self._controller()
                
                # Advance frame
                if not self.paused and self.joint_angles is not None:
                    self.frame += 1
                    if self.frame >= len(self.joint_angles):
                        if self.loop:
                            self.frame = 0
                        else:
                            self.frame = len(self.joint_angles) - 1
                
                # Update viewer
                viewer.sync()
                time.sleep(self.dt)
        
        self.running = False
    
    def start(self):
        """Start the visualizer in a separate thread"""
        if self.model is None:
            self.load_model()
        
        # Don't start if already running
        if self.is_running():
            return
        
        if self.viewer_thread is None or not self.viewer_thread.is_alive():
            self.viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
            self.viewer_thread.start()
            # Give the thread time to initialize
            time.sleep(0.5)
            print("✓ MuJoCo viewer started")
    
    def stop(self):
        """Stop the visualizer"""
        self.running = False
        if self.viewer_thread is not None:
            self.viewer_thread.join(timeout=2.0)
        print("✓ MuJoCo viewer stopped")
    
    def is_running(self) -> bool:
        """Check if visualizer is running"""
        return self.running and (self.viewer_thread is not None and self.viewer_thread.is_alive())
    
    def update_motion(self, pkl_path: str):
        """Update motion data and reset playback"""
        self.load_motion_from_pkl(pkl_path)
        self.frame = 0
        self.paused = False
    
    def set_frame(self, frame: int):
        """Set current frame"""
        if self.joint_angles is not None:
            self.frame = max(0, min(frame, len(self.joint_angles) - 1))
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
    
    def reset(self):
        """Reset to frame 0"""
        self.frame = 0


# Backward-compatible alias for older Kimodo code/configs.
GO2Visualizer = RobotMotionVisualizer

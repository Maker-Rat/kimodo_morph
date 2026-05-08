# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: Kimodo G1 -> Morph teacher -> target robot."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from typing import Any, Dict, Optional

import torch

from kimodo.adapters.go2_visualizer import GO2Visualizer
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton import G1Skeleton34, global_rots_to_local_rots
from kimodo.tools import to_numpy


class KimodoRetargetingAdapter:
    """Convert Kimodo G1 output to PKL and run Morph teacher inference."""

    def __init__(
        self,
        retarget_model_dir: str,
        device: str = "cuda:0",
        enable_visualization: bool = False,
        *,
        output_root: Optional[str] = None,
        processed_dir: Optional[str] = None,
        task_family: Optional[str] = None,
        pair_id: Optional[str] = None,
        teacher_epoch: Optional[int] = None,
        reverse: Optional[bool] = None,
        go2_xml_path: Optional[str] = None,
    ):
        self.device = str(device)
        self.teacher_dir = str(retarget_model_dir)
        self.enable_visualization = enable_visualization
        self.visualizer = None

        # Morph pipeline configuration (UI-provided).
        self.output_root = os.path.abspath(str(output_root or "./morph"))
        self.processed_dir = str(processed_dir or "")
        self.task_family = str(task_family or "")
        self.pair_id = str(pair_id or "")

        self.teacher_epoch = teacher_epoch
        self.reverse = bool(reverse) if reverse is not None else False

        if self.processed_dir and not os.path.isabs(self.processed_dir):
            self.processed_dir = os.path.abspath(os.path.join(self.output_root, self.processed_dir))

        default_xml = os.path.join(self.output_root, "assets", "robots", "unitree_go2", "go2.xml")
        self.go2_xml_path = str(go2_xml_path or default_xml)

        if not os.path.exists(self.teacher_dir):
            raise FileNotFoundError(f"Teacher run directory not found: {self.teacher_dir}")
        if not self.processed_dir:
            raise ValueError("Processed dir is empty. Select a Processed Data option in the Kimodo UI.")
        if not os.path.exists(self.processed_dir):
            raise FileNotFoundError(f"Processed dir not found: {self.processed_dir}")
        if not self.task_family:
            raise ValueError("Task family is empty. Select a task in the Kimodo UI.")
        if not self.pair_id:
            raise ValueError("Pair id is empty. Select a pair in the Kimodo UI.")

        self.converter = MujocoQposConverter(G1Skeleton34())
        self.skeleton = G1Skeleton34()
        self.toe_indices = torch.tensor([7, 15, 23, 31, 32])

    def kimodo_to_morph_pkl(
        self,
        joints_pos: torch.Tensor,  # [T, 34, 3]
        joints_rot: torch.Tensor,  # [T, 34, 3, 3]
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """Convert Kimodo output tensors into Morph G1 PKL format."""
        joints_pos = torch.as_tensor(joints_pos, dtype=torch.float32)
        joints_rot = torch.as_tensor(joints_rot, dtype=torch.float32)
        root_pos = joints_pos[:, 0, :]
        local_rots_full = global_rots_to_local_rots(joints_rot, self.skeleton)  # [T, 34, 3, 3]

        mask = torch.ones(34, dtype=torch.bool)
        mask[self.toe_indices] = False

        qpos = self.converter.to_qpos(
            local_rot_mats=local_rots_full.unsqueeze(0),
            root_positions=root_pos.unsqueeze(0),
            root_quat_w_first=False,  # output xyzw
        ).squeeze(0)

        root_pos_parsed = qpos[:, :3]
        root_rot_xyzw = qpos[:, 3:7]
        dof_pos = qpos[:, 7:36]

        return {
            "fps": float(fps),
            "dof_pos": to_numpy(dof_pos),
            "root_pos": to_numpy(root_pos_parsed),
            "root_rot": to_numpy(root_rot_xyzw),
            "local_body_pos": None,
            "link_body_list": None,
        }

    def _run_morph_inference(self, input_pkl: str, output_pkl: str) -> None:
        cmd = [
            sys.executable,
            "-m",
            "csmt.pipelines.infer_teacher",
            "--output-root",
            self.output_root,
            "--processed-dir",
            self.processed_dir,
            "--task-family",
            self.task_family,
            "--pair-id",
            self.pair_id,
            "--teacher-dir",
            self.teacher_dir,
            "--input-pkl",
            input_pkl,
            "--output-pkl",
            output_pkl,
            "--device",
            self.device,
        ]

        if self.teacher_epoch is not None:
            cmd.extend(["--teacher-epoch", str(self.teacher_epoch)])
        if self.reverse:
            cmd.append("--reverse")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Morph infer_teacher failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

    def retarget(
        self,
        joints_pos: torch.Tensor,
        joints_rot: torch.Tensor,
        fps: float = 30.0,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: Kimodo G1 tensors -> Morph retargeted PKL."""
        if output_path is None:
            output_path = "./retarget_output/retargeted_go2.pkl"
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        g1_pkl = self.kimodo_to_morph_pkl(joints_pos, joints_rot, fps)
        g1_output_path = output_path.replace(".pkl", "_g1.pkl")
        with open(g1_output_path, "wb") as f:
            pickle.dump(g1_pkl, f)

        self._run_morph_inference(g1_output_path, output_path)

        with open(output_path, "rb") as f:
            out = pickle.load(f)

        if self.enable_visualization:
            if self.visualizer is None:
                self.visualizer = GO2Visualizer(go2_xml_path=self.go2_xml_path)
            if not self.visualizer.is_running():
                self.visualizer.start()
            self.visualizer.update_motion(output_path)

        return out

    def close(self) -> None:
        """Shutdown viewer resources owned by this adapter."""
        if self.visualizer is None:
            return
        try:
            self.visualizer.stop()
        except Exception:
            pass
        self.visualizer = None

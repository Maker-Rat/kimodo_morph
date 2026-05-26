# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: Kimodo G1 -> Morph teacher -> target robot."""

from __future__ import annotations

import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kimodo.adapters.go2_visualizer import RobotMotionVisualizer
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton import G1Skeleton34, global_rots_to_local_rots
from kimodo.tools import to_numpy


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return quat_xyzw[[3, 0, 1, 2]]


def _extract_yaw_xyzw(quat_xyzw: np.ndarray) -> float:
    rot = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    return float(rot.as_euler("xyz", degrees=False)[2])


def _world_vel_to_local_xy(lin_vel_world: np.ndarray, yaw: float) -> np.ndarray:
    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    vel = np.asarray(lin_vel_world, dtype=np.float32).reshape(3)
    return np.array(
        [c * vel[0] - s * vel[1], s * vel[0] + c * vel[1], vel[2]],
        dtype=np.float32,
    )


def _body_ang_vel_xyzw(prev_quat: np.ndarray, curr_quat: np.ndarray, dt: float) -> np.ndarray:
    prev_rot = Rotation.from_quat(np.asarray(prev_quat, dtype=np.float64).reshape(4))
    curr_rot = Rotation.from_quat(np.asarray(curr_quat, dtype=np.float64).reshape(4))
    rel = prev_rot.inv() * curr_rot
    return (rel.as_rotvec() / max(float(dt), 1e-8)).astype(np.float32)


def _make_ref_frame(
    dof_pos: np.ndarray,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    prev_root_pos: Optional[np.ndarray],
    prev_root_rot_xyzw: Optional[np.ndarray],
    dt: float,
    quat_convention: str,
    num_joints: int,
) -> np.ndarray:
    num_joints = int(num_joints)
    dof = np.asarray(dof_pos, dtype=np.float32).reshape(-1)[:num_joints]
    if dof.shape[0] < num_joints:
        dof = np.pad(dof, (0, num_joints - dof.shape[0])).astype(np.float32)

    root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32).reshape(4)
    if prev_root_pos is None or prev_root_rot_xyzw is None:
        lin_vel_local = np.zeros(3, dtype=np.float32)
        ang_vel_local = np.zeros(3, dtype=np.float32)
    else:
        lin_vel_world = (root_pos - np.asarray(prev_root_pos, dtype=np.float32).reshape(3)) / max(float(dt), 1e-8)
        lin_vel_local = _world_vel_to_local_xy(lin_vel_world, _extract_yaw_xyzw(root_rot_xyzw))
        ang_vel_local = _body_ang_vel_xyzw(prev_root_rot_xyzw, root_rot_xyzw, dt)

    quat = root_rot_xyzw if quat_convention == "xyzw" else _quat_xyzw_to_wxyz(root_rot_xyzw)
    return np.concatenate([dof, lin_vel_local, ang_vel_local, quat.astype(np.float32)]).astype(np.float32)


class MorphReferencePublisher:
    """Publish retargeted Morph PKL frames in the sim2sim live-reference schema."""

    def __init__(
        self,
        endpoint: str,
        robot: str,
        fps: float,
        *,
        quat_convention: str = "wxyz",
        offsets: tuple[int, ...] = (0, 1),
        warmup_sec: float = 0.25,
    ) -> None:
        import zmq

        self.endpoint = str(endpoint)
        self.robot = str(robot)
        self.fps = float(fps)
        self.dt = 1.0 / max(self.fps, 1e-8)
        self.quat_convention = str(quat_convention).lower()
        if self.quat_convention not in ("xyzw", "wxyz"):
            raise ValueError(f"Unsupported quat convention: {quat_convention}")
        self.offsets = tuple(int(x) for x in offsets)
        if len(self.offsets) == 0 or min(self.offsets) < 0:
            raise ValueError(f"Invalid publish offsets: {self.offsets}")
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.endpoint)
        self.seq = 0
        if warmup_sec > 0:
            time.sleep(float(warmup_sec))

    def close(self) -> None:
        self.socket.close(linger=0)

    def stream_pkl(self, pkl_path: str, *, realtime: bool = True) -> int:
        with open(pkl_path, "rb") as f:
            motion = pickle.load(f)
        dof = np.asarray(motion["dof_pos"], dtype=np.float32)
        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
        root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
        fps = float(motion.get("fps", self.fps))
        dt = 1.0 / max(fps, 1e-8)
        num_joints = int(dof.shape[1])
        frame_dim = int(num_joints + 10)
        max_offset = max(self.offsets)
        count = max(0, len(dof) - max_offset)
        if count == 0:
            return 0

        for anchor in range(count):
            rows = []
            for offset in self.offsets:
                idx = anchor + offset
                prev_idx = idx - 1
                rows.append(
                    _make_ref_frame(
                        dof_pos=dof[idx],
                        root_pos=root_pos[idx],
                        root_rot_xyzw=root_rot[idx],
                        prev_root_pos=None if prev_idx < 0 else root_pos[prev_idx],
                        prev_root_rot_xyzw=None if prev_idx < 0 else root_rot[prev_idx],
                        dt=dt,
                        quat_convention=self.quat_convention,
                        num_joints=num_joints,
                    )
                )
            refs = np.stack(rows, axis=0).astype(np.float32)
            packet = {
                "version": 1,
                "seq": int(self.seq),
                "timestamp": float(time.time()),
                "robot": self.robot,
                "fps": float(fps),
                "latency_frames": int(max_offset),
                "ref_offsets": list(self.offsets),
                "ref_shape": [int(len(self.offsets)), frame_dim],
                "joint_count": num_joints,
                "frame_dim": frame_dim,
                "refs_dtype": "float32",
                "quat_convention": self.quat_convention,
                "refs": refs.tolist(),
                "valid": True,
            }
            self.socket.send_pyobj(packet)
            self.seq += 1
            if realtime:
                time.sleep(dt)
        return count


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
        robot_xml_path: Optional[str] = None,
        go2_xml_path: Optional[str] = None,
        corrector_ckpt: Optional[str] = None,
        root_rotation_mode: str = "yaw",
        dst_start_height: Optional[float] = None,
        apply_root_skate_comp: bool = False,
        publish_zmq: Optional[str] = None,
        publish_quat_convention: str = "wxyz",
        publish_ref_offsets: tuple[int, ...] = (0, 1),
        publish_realtime: bool = True,
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
        self.corrector_ckpt = str(corrector_ckpt) if corrector_ckpt else ""
        self.root_rotation_mode = str(root_rotation_mode or "yaw")
        self.dst_start_height = dst_start_height
        self.apply_root_skate_comp = bool(apply_root_skate_comp)
        self.publish_zmq = str(publish_zmq) if publish_zmq else ""
        self.publish_quat_convention = str(publish_quat_convention or "wxyz")
        self.publish_ref_offsets = tuple(int(x) for x in publish_ref_offsets)
        self.publish_realtime = bool(publish_realtime)

        if not self.processed_dir:
            self.processed_dir = self._infer_processed_dir_from_teacher()

        if self.processed_dir:
            if os.path.isabs(self.processed_dir):
                if not os.path.exists(self.processed_dir):
                    self.processed_dir = self._resolve_candidate_path(self.processed_dir)
            else:
                self.processed_dir = os.path.abspath(os.path.join(self.output_root, self.processed_dir))

        default_xml = os.path.join(self.output_root, "assets", "robots", "unitree_go2", "go2.xml")
        self.robot_xml_path = str(robot_xml_path or go2_xml_path or default_xml)

        print(
            "[Retargeting][adapter-init] "
            f"teacher_dir={self.teacher_dir} "
            f"processed_dir={self.processed_dir or '<EMPTY>'} "
            f"task_family={self.task_family or '<EMPTY>'} "
            f"pair_id={self.pair_id or '<EMPTY>'} "
            f"xml={self.robot_xml_path} "
            f"corrector={self.corrector_ckpt or '<none>'} "
            f"root_rotation_mode={self.root_rotation_mode} "
            f"publish_zmq={self.publish_zmq or '<none>'}"
        )

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

    def _resolve_candidate_path(self, raw: str) -> str:
        p = Path(str(raw).strip())
        if p.is_absolute():
            # If absolute path is stale (different machine root), try remapping under current output_root.
            if p.exists():
                return str(p)
            p_posix = p.as_posix()
            marker = "/data/"
            idx = p_posix.find(marker)
            if idx != -1:
                suffix = p_posix[idx + 1 :]  # "data/..."
                remapped = (Path(self.output_root) / suffix).resolve(strict=False)
                return str(remapped)
            return str(p)
        cands = [
            (Path(self.output_root) / p),
            (Path(self.teacher_dir) / p),
            (Path(self.teacher_dir).parent / p),
            (Path.cwd() / p),
        ]
        for c in cands:
            if c.exists():
                return str(c.resolve(strict=False))
        return str(cands[0].resolve(strict=False))

    def _infer_processed_dir_from_teacher(self) -> str:
        run_dir = Path(self.teacher_dir)
        meta_path = run_dir / "refactor_teacher_run.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
            legacy_args = meta.get("legacy_args", {})
            if not isinstance(legacy_args, dict):
                legacy_args = {}
            for key in ("processed_dir", "processed_dirs", "dataset_roots"):
                val = meta.get(key)
                if not val:
                    val = legacy_args.get(key) or legacy_args.get(key.replace("_", "-"))
                if isinstance(val, str) and val.strip():
                    resolved = self._resolve_candidate_path(val.strip())
                    if Path(resolved).exists():
                        return resolved
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and item.strip():
                            resolved = self._resolve_candidate_path(item.strip())
                            if Path(resolved).exists():
                                return resolved
            for key in (
                "srcstats_path", "dststats_path",
                "src_train_path", "dst_train_path",
                "src_test_path", "dst_test_path",
                "humstats_path", "dogstats_path",
                "hum_train_path", "dog_train_path",
                "hum_test_path", "dog_test_path",
            ):
                raw = meta.get(key)
                if not raw:
                    raw = legacy_args.get(key)
                if not raw:
                    continue
                resolved = self._resolve_candidate_path(str(raw))
                p = Path(resolved)
                if p.suffix.lower() in (".npz", ".npy", ".pkl", ".pt", ".pth", ".json"):
                    return str(p.parent.resolve(strict=False))
                return str(p.resolve(strict=False))

        para_path = run_dir / "para.txt"
        if para_path.exists():
            txt = para_path.read_text().strip()
            try:
                toks = shlex.split(txt)
            except Exception:
                toks = txt.split()
            for i, tok in enumerate(toks):
                if tok in ("--processed-dir", "--processed_dir", "--data-dir", "--data_dir"):
                    if i + 1 < len(toks):
                        resolved = self._resolve_candidate_path(toks[i + 1])
                        if Path(resolved).exists():
                            return resolved

        return ""

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
            "--root-rotation-mode",
            self.root_rotation_mode,
            "--no-save-src-debug",
        ]

        if self.teacher_epoch is not None:
            cmd.extend(["--teacher-epoch", str(self.teacher_epoch)])
        if self.reverse:
            cmd.append("--reverse")
        if self.corrector_ckpt:
            cmd.extend(["--corrector-ckpt", self.corrector_ckpt])
        if self.dst_start_height is not None:
            cmd.extend(["--dst-start-height", str(float(self.dst_start_height))])
        if self.apply_root_skate_comp:
            cmd.append("--apply-root-skate-comp")

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
                self.visualizer = RobotMotionVisualizer(robot_xml_path=self.robot_xml_path)
            if not self.visualizer.is_running():
                self.visualizer.start()
            self.visualizer.update_motion(output_path)

        if self.publish_zmq:
            pair_dst = self.pair_id.split("_to_", 1)[1] if "_to_" in self.pair_id else ""
            robot = pair_dst if not self.reverse else self.pair_id.split("_to_", 1)[0]
            publisher = MorphReferencePublisher(
                endpoint=self.publish_zmq,
                robot=robot,
                fps=float(out.get("fps", fps)),
                quat_convention=self.publish_quat_convention,
                offsets=self.publish_ref_offsets,
            )
            try:
                sent = publisher.stream_pkl(output_path, realtime=self.publish_realtime)
                print(
                    "[Retargeting][ZMQ] "
                    f"published {sent} packets to {self.publish_zmq} "
                    f"(robot={robot}, offsets={list(self.publish_ref_offsets)})"
                )
            finally:
                publisher.close()

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

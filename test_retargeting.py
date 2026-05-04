#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Quick test script: Generate G1 motion via Kimodo, then retarget via Morph teacher.

Usage:
    python test_retargeting.py --prompt "A person walks forward" --output ./test_go2
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

from kimodo.adapters import KimodoRetargetingAdapter
from kimodo.model.load_model import load_model
from kimodo.model.registry import resolve_model_name
from kimodo.tools import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Generate G1 motion and retarget with Morph")
    parser.add_argument("--prompt", type=str, default="A person walks forward", help="Motion description")
    parser.add_argument("--num_frames", type=int, default=150, help="Motion duration in frames")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--diffusion_steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="./test_retarget_output", help="Output directory")
    parser.add_argument(
        "--retarget_model_dir",
        type=str,
        default="./morph/runs/teacher_loco_g1_go2",
        help="Morph teacher run directory",
    )
    parser.add_argument("--output_root", type=str, default="./morph", help="Morph output root")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="./morph/data/processed/loco_g1_go2",
        help="Morph processed dataset directory",
    )
    parser.add_argument("--task_family", type=str, default="locomotion", help="Morph task family")
    parser.add_argument("--pair_id", type=str, default="g1_to_go2", help="Morph pair id")
    parser.add_argument("--teacher_epoch", type=int, default=None, help="Optional teacher checkpoint epoch")
    parser.add_argument("--reverse", action="store_true", help="Run reverse retargeting")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    
    args = parser.parse_args()
    
    device = args.device
    seed_everything(args.seed)
    
    print(f"\n{'='*60}")
    print(f"KIMODO → RETARGETING PIPELINE TEST")
    print(f"{'='*60}\n")
    
    # Load Kimodo G1 model
    print("[1/3] Loading Kimodo-G1 model...")
    model = load_model("kimodo-g1-rp", device=device)
    print(f"✓ Model loaded: {type(model).__name__}")
    
    # Generate motion
    print(f"\n[2/3] Generating G1 motion...")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Duration: {args.num_frames} frames ({args.num_frames/30:.1f}s)")
    
    with torch.inference_mode():
        output = model(
            [args.prompt],
            num_frames=[args.num_frames],
            num_denoising_steps=args.diffusion_steps,
            num_samples=args.num_samples,
            cfg_weight=[2.0, 2.0],
        )
    
    joints_pos = output["posed_joints"][0]  # [T, 34, 3]
    joints_rot = output["global_rot_mats"][0]  # [T, 34, 3, 3]
    
    print(f"✓ Generated: {joints_pos.shape[0]} frames, {joints_pos.shape[1]} joints")
    
    # Retarget to GO2
    print(f"\n[3/3] Retargeting to GO2 (Morph teacher)...")
    
    if not os.path.exists(args.retarget_model_dir):
        print(f"✗ Teacher run directory not found: {args.retarget_model_dir}")
        print("  Please set --retarget_model_dir to your trained Morph teacher directory")
        return
    
    adapter = KimodoRetargetingAdapter(
        args.retarget_model_dir,
        device=str(device),
        output_root=args.output_root,
        processed_dir=args.processed_dir,
        task_family=args.task_family,
        pair_id=args.pair_id,
        teacher_epoch=args.teacher_epoch,
        reverse=args.reverse,
    )
    
    output_pkl_path = os.path.join(args.output, "retargeted_go2.pkl")
    go2_pkl = adapter.retarget(joints_pos, joints_rot, fps=30.0, output_path=output_pkl_path)
    
    print(f"\n{'='*60}")
    print(f"✓ RETARGETING COMPLETE")
    print(f"{'='*60}")
    print(f"  G1 input:     {joints_pos.shape[0]} frames, {joints_pos.shape[1]} joints (34 DoF)")
    print(f"  GO2 output:   {go2_pkl['dof_pos'].shape[0]} frames, {go2_pkl['dof_pos'].shape[1]} DoF")
    print(f"  Saved to:     {output_pkl_path}")
    print()


if __name__ == "__main__":
    main()

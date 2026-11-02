"""Run Pi0 inference on LIBERO ``demo_0`` at step zero."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch


def load_libero_observation(path: str | Path) -> dict[str, object]:
    """Read the fixed real LIBERO sample directly from HDF5."""
    source = Path(path).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        data = handle["data"]
        info = json.loads(data.attrs["problem_info"])
        obs = data["demo_0"]["obs"]
        state = np.concatenate(
            (
                np.asarray(obs["ee_pos"][0]),
                np.asarray(obs["ee_ori"][0]),
                np.asarray(obs["gripper_states"][0]),
            )
        ).astype(np.float32)
        return {
            "observation/image": np.ascontiguousarray(
                obs["agentview_rgb"][0][::-1, ::-1]
            ),
            "observation/wrist_image": np.ascontiguousarray(
                obs["eye_in_hand_rgb"][0][::-1, ::-1]
            ),
            "observation/state": state,
            "prompt": info["language_instruction"],
        }


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set")
    return value


def _inference_engine(device: str):
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("Pi0 inference requires a CUDA device")
    props = torch.cuda.get_device_properties(target)
    if props.major == 9 and props.minor == 0:
        from qimeng_megakernel.models.VLA.pi.pi0.H100.inference import Pi0Inference
    elif (props.major, props.minor, props.multi_processor_count) == (8, 0, 108):
        from qimeng_megakernel.models.VLA.pi.pi0.A100.inference import Pi0Inference
    else:
        raise RuntimeError(
            f"unsupported Pi0 device {props.name}: compute "
            f"{props.major}.{props.minor}, {props.multi_processor_count} SMs"
        )
    return Pi0Inference(device=target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("PI0_CKPT"))
    parser.add_argument("--observation", default=os.environ.get("PI0_LIBERO_HDF5"))
    parser.add_argument("--views", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--chunk", type=int, choices=(20, 50), default=50)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint = args.checkpoint or _env("PI0_CKPT")
    observation_path = args.observation or _env("PI0_LIBERO_HDF5")
    generator = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    noise = torch.randn(args.chunk, 32, generator=generator)
    result = _inference_engine(args.device).infer(
        checkpoint,
        load_libero_observation(observation_path),
        views=args.views,
        chunk=args.chunk,
        noise=noise,
    )
    print(json.dumps({key: value.tolist() for key, value in result.items()}))


if __name__ == "__main__":
    main()

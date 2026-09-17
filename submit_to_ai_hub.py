"""
submit_to_ai_hub.py

Uploads stakml_cifar.onnx to Qualcomm AI Hub, compiles it for a real
Snapdragon X device, and profiles it two ways on the same hardware:
  1. Default (CPU) execution
  2. NPU execution via the QNN Execution Provider

Both profile jobs run on Qualcomm's cloud-hosted device farm, so no
physical Snapdragon PC is required.

Usage:
    pip install qai-hub
    qai-hub configure --api_token YOUR_TOKEN
    python submit_to_ai_hub.py stakml_cifar.onnx
"""

import sys
import qai_hub as hub

DEVICE_NAME = "Snapdragon X Elite CRD"


def list_matching_devices():
    """Sanity check: print available Snapdragon X-class devices in case
    the exact device name has changed since this script was written."""
    devices = hub.get_devices()
    matches = [d for d in devices if "Snapdragon X" in d.name]
    print("Available Snapdragon X-class devices on AI Hub:")
    for d in matches:
        print(f"  - {d.name}")
    print()


def submit(onnx_path: str):
    list_matching_devices()
    device = hub.Device(DEVICE_NAME)

    # ── Step 1: Compile the ONNX model for this device ──────────────────────
    # AI Hub expects models to go through its compile step before profiling
    # (an uncompiled upload triggers an "unknown provenance" warning and
    # behaves inconsistently with compute-unit restriction options). Compile
    # once, then profile the same compiled artifact both ways below.
    print(f"Submitting compile job for {DEVICE_NAME}...")
    compile_job = hub.submit_compile_job(
        model=onnx_path,
        device=device,
        name="stakml_cifar_cnn",
        options="--target_runtime onnx",
    )
    compile_job.wait()
    print(f"Compile job done: {compile_job.url}")
    compiled_model = compile_job.get_target_model()

    # ── Step 2: Profile on CPU, forced via --compute_unit ────────────────────
    # --compute_unit is the documented universal flag for restricting which
    # compute unit a job runs on across all AI Hub job types. This is
    # different from --onnx_execution_providers, which only registers a
    # provider without excluding others (confirmed not to work via runtime
    # log: QNN kept running regardless of that flag).
    print("\nSubmitting CPU profile job (--compute_unit cpu)...")
    cpu_profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="stakml_cifar_cnn_cpu",
        options="--compute_unit cpu",
    )
    cpu_profile_job.wait()
    print(f"CPU profile job done: {cpu_profile_job.url}")

    # ── Step 3: Profile on NPU via the QNN execution provider ───────────────
    print("\nSubmitting NPU (QNN) profile job...")
    npu_profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="stakml_cifar_cnn_npu",
        options="--compute_unit npu",
    )
    npu_profile_job.wait()
    print(f"NPU profile job done: {npu_profile_job.url}")

    print("\nDone. Open the job URLs above in a browser to see full "
          "latency/memory/compute-unit breakdowns — screenshot these "
          "for your submission's benchmark section.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <model.onnx>")
        sys.exit(1)
    submit(sys.argv[1])
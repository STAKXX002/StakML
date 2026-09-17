"""
export_to_onnx.py

Bridges a StakML cifar_cnn checkpoint (raw float32 dump from serialize.hpp)
into an equivalent PyTorch model, then exports it to ONNX (opset 20) for
Qualcomm AI Hub.

Usage:
    python export_to_onnx.py path/to/stakml_checkpoint.bin stakml_cifar.onnx

The checkpoint must have been produced by stakml::serialize::save_model()
on the exact architecture defined in examples/cifar_cnn.cpp:

    Conv2d(3,32,3,1,1) -> ReLU -> MaxPool2d(2,2) ->
    Conv2d(32,64,3,1,1) -> ReLU -> MaxPool2d(2,2) ->
    Flatten -> Dropout(0.5) -> Linear(4096,256) -> ReLU ->
    Dropout(0.3) -> Linear(256,10)

Dropout has no learnable parameters, so it is skipped both in
serialize.hpp's output and in the state_dict below. torch.onnx.export
also just no-ops Dropout automatically since we call model.eval().
"""

import os
import sys
import numpy as np
import onnx
import torch
import torch.nn as nn


class StakMLCifarNet(nn.Module):
    """Architecture-for-architecture mirror of examples/cifar_cnn.cpp."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def load_stakml_weights(model: StakMLCifarNet, checkpoint_path: str):
    """
    Reads the flat float32 dump written by stakml::serialize::save_model()
    and assigns each chunk into the matching PyTorch parameter, in the
    exact order Sequential::parameters() produced them.

    StakML Linear stores W as {in_features, out_features} and computes
    x @ W directly, so it must be TRANSPOSED to match PyTorch's
    {out_features, in_features} + x @ W.T convention.

    Conv2d weight layout {C_out, C_in, kH, kW} is identical in both
    frameworks, so no reshape/transpose needed there.
    """
    raw = np.fromfile(checkpoint_path, dtype=np.float32)
    offset = 0

    def take(n):
        nonlocal offset
        chunk = raw[offset:offset + n]
        offset += n
        if chunk.shape[0] != n:
            raise ValueError(
                f"Checkpoint too short: expected {n} more floats at "
                f"offset {offset - n}, only got {chunk.shape[0]}. "
                f"Check the architecture matches cifar_cnn.cpp exactly."
            )
        return chunk

    # --- conv1: W{32,3,3,3}, b{32} ---
    w = take(32 * 3 * 3 * 3).reshape(32, 3, 3, 3)
    b = take(32)
    model.conv1.weight.data = torch.from_numpy(w.copy())
    model.conv1.bias.data = torch.from_numpy(b.copy())

    # --- conv2: W{64,32,3,3}, b{64} ---
    w = take(64 * 32 * 3 * 3).reshape(64, 32, 3, 3)
    b = take(64)
    model.conv2.weight.data = torch.from_numpy(w.copy())
    model.conv2.bias.data = torch.from_numpy(b.copy())

    # --- fc1: StakML W{4096,256} -> transpose -> PyTorch {256,4096}; b{256} ---
    w = take(4096 * 256).reshape(4096, 256).T
    b = take(256)
    model.fc1.weight.data = torch.from_numpy(np.ascontiguousarray(w))
    model.fc1.bias.data = torch.from_numpy(b.copy())

    # --- fc2: StakML W{256,10} -> transpose -> PyTorch {10,256}; b{10} ---
    w = take(256 * 10).reshape(256, 10).T
    b = take(10)
    model.fc2.weight.data = torch.from_numpy(np.ascontiguousarray(w))
    model.fc2.bias.data = torch.from_numpy(b.copy())

    if offset != raw.shape[0]:
        raise ValueError(
            f"Checkpoint has {raw.shape[0] - offset} leftover floats after "
            f"reading all expected parameters — architecture mismatch."
        )

    print(f"Loaded {offset} floats successfully, checkpoint fully consumed.")


def consolidate_external_data(onnx_path: str):
    """
    Loads the just-exported ONNX model (pulling in any external tensor
    data sidecar files) and re-saves it as one fully self-contained file
    with all weights embedded inline. AI Hub rejects models that expect
    a separate .data file to be uploaded alongside them.
    """
    model = onnx.load(onnx_path, load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, onnx_path, save_as_external_data=False)

    # Clean up the now-unused sidecar file, if PyTorch created one.
    sidecar = onnx_path + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)

    print(f"Consolidated {onnx_path} into a single self-contained file "
          f"({os.path.getsize(onnx_path) / 1024:.1f} KB), verified with onnx.checker.")


def export(checkpoint_path: str, onnx_path: str):
    model = StakMLCifarNet()
    load_stakml_weights(model, checkpoint_path)
    model.eval()

    # Fixed batch size of 1 — AI Hub requires static input shapes.
    dummy_input = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=20,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {onnx_path}")

    # PyTorch's dynamo-based exporter can write tensor data to a separate
    # sidecar file (<onnx_path>.data) even for small models. AI Hub expects
    # a single self-contained .onnx file, so reload with external data
    # attached and re-save everything embedded inline in one file.
    consolidate_external_data(onnx_path)

    # Sanity check: run both the raw dummy input through the exported
    # model and print output shape/sample so you can eyeball it.
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Output shape: {tuple(out.shape)}  sample logits: {out[0][:3].tolist()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <stakml_checkpoint.bin> <output.onnx>")
        sys.exit(1)
    export(sys.argv[1], sys.argv[2])
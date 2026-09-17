"""
verify_export.py

Loads the exact same test image StakML dumped (verify_input.bin) into the
exported ONNX model, and diffs the resulting logits against StakML's own
output (verify_stakml_logits.bin). If the export/weight-bridge is correct,
these should match to within float32 rounding error.

Usage:
    pip install onnxruntime
    python verify_export.py stakml_cifar.onnx verify_input.bin verify_stakml_logits.bin
"""

import sys
import numpy as np
import onnxruntime as ort


def verify(onnx_path: str, input_bin: str, stakml_logits_bin: str):
    # Same layout as StakML: {1,3,32,32} float32, already normalised
    img = np.fromfile(input_bin, dtype=np.float32).reshape(1, 3, 32, 32)
    stakml_logits = np.fromfile(stakml_logits_bin, dtype=np.float32)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: img})[0].reshape(-1)

    print(f"StakML logits: {stakml_logits}")
    print(f"ONNX logits:   {onnx_logits}")

    diff = np.abs(stakml_logits - onnx_logits)
    print(f"\nMax abs diff:  {diff.max():.6f}")
    print(f"Mean abs diff: {diff.mean():.6f}")

    stakml_pred = int(np.argmax(stakml_logits))
    onnx_pred = int(np.argmax(onnx_logits))
    print(f"\nStakML predicted class: {stakml_pred}")
    print(f"ONNX predicted class:   {onnx_pred}")

    if diff.max() < 1e-3 and stakml_pred == onnx_pred:
        print("\nMATCH — export is correct, safe to proceed to AI Hub.")
    elif diff.max() < 1e-1 and stakml_pred == onnx_pred:
        print("\nCLOSE — same prediction, some numerical drift (likely fine, "
              "but inspect before trusting benchmark numbers).")
    else:
        print("\nMISMATCH — do not proceed to AI Hub yet. Likely causes: "
              "a weight transpose/reshape bug in export_to_onnx.py, wrong "
              "parameter ordering, or the checkpoint doesn't match the "
              "architecture the script assumes.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <model.onnx> <verify_input.bin> <verify_stakml_logits.bin>")
        sys.exit(1)
    verify(sys.argv[1], sys.argv[2], sys.argv[3])

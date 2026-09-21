"""
classify_on_snapdragon.py

The actual demo: picks a handful of real CIFAR-10 test images, submits
them to the compiled StakML model running on real cloud-hosted Snapdragon
NPU hardware via Qualcomm AI Hub, and prints what it predicted.

This is different from submit_to_ai_hub.py, which only measures latency.
This script gets real classification *results* back from real Snapdragon
silicon, which is the actual use-case demo for the submission.

Usage:
    pip install qai-hub numpy
    python classify_on_snapdragon.py stakml_cifar.onnx data/cifar-10-batches-bin/test_batch.bin
"""

import sys
import numpy as np
import qai_hub as hub

DEVICE_NAME = "Snapdragon X Elite CRD"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

RECORD_BYTES = 1 + 3 * 32 * 32  # 1 label byte + 3072 pixel bytes


def load_samples(test_batch_path: str, indices):
    """
    Reads specific records directly out of the standard CIFAR-10 binary
    test_batch.bin (1 label byte + 3072 pixel bytes per record, channels
    stored as three 1024-byte planes: R, G, B). Applies the exact same
    per-channel normalisation cifar_cnn.cpp used during training.
    """
    images, labels = [], []
    with open(test_batch_path, "rb") as f:
        for idx in indices:
            f.seek(idx * RECORD_BYTES)
            record = f.read(RECORD_BYTES)
            label = record[0]
            pixels = np.frombuffer(record[1:], dtype=np.uint8).reshape(3, 32, 32).astype(np.float32)
            pixels /= 255.0
            for c in range(3):
                pixels[c] = (pixels[c] - MEAN[c]) / STD[c]
            images.append(pixels)
            labels.append(label)
    return np.stack(images), labels


def classify(onnx_path: str, test_batch_path: str, num_samples: int = 8):
    device = hub.Device(DEVICE_NAME)

    print(f"Compiling {onnx_path} for {DEVICE_NAME}...")
    compile_job = hub.submit_compile_job(
        model=onnx_path,
        device=device,
        name="stakml_cifar_cnn_classify",
        options="--target_runtime onnx",
    )
    compile_job.wait()
    compiled_model = compile_job.get_target_model()

    # Pick evenly spaced indices across the 10,000-image test set for
    # variety rather than just the first N in a row.
    indices = list(range(0, 10000, 10000 // num_samples))[:num_samples]
    images, true_labels = load_samples(test_batch_path, indices)

    print(f"\nSubmitting {num_samples} real CIFAR-10 test images for on-device "
          f"inference on real Snapdragon NPU hardware...")
    inference_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=dict(input=[images[i:i + 1] for i in range(num_samples)]),
        options="--compute_unit npu",
    )
    inference_job.wait()
    print(f"Inference job done: {inference_job.url}\n")

    outputs = inference_job.download_output_data()
    logits_list = outputs["logits"]  # list of per-sample arrays

    correct = 0
    print(f"{'True':<12} {'Predicted':<12} {'Confidence':<12} {'Result'}")
    print("-" * 50)
    for i, logits in enumerate(logits_list):
        logits = np.asarray(logits).reshape(-1)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        pred = int(np.argmax(probs))
        true = true_labels[i]
        is_correct = pred == true
        correct += is_correct
        print(f"{CIFAR10_CLASSES[true]:<12} {CIFAR10_CLASSES[pred]:<12} "
              f"{probs[pred]*100:>6.1f}%      {'correct' if is_correct else 'wrong'}")

    print(f"\n{correct}/{num_samples} correct — real inference on real Snapdragon NPU hardware.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <model.onnx> <test_batch.bin>")
        sys.exit(1)
    classify(sys.argv[1], sys.argv[2])

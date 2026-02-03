import os
from pathlib import Path
import torch
import numpy as np
from config_loader import load_configs
from models import make_model
from datasets import make_dataloaders
from utils.metrics import compute_metrics, plot_confusion_matrix


def export_onnx(model, sample_input, output_path, input_names=["input"], output_names=["output"]):
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}}
    )


class NumpyCalibrationDataReader:
    def __init__(self, dataloader, input_name):
        self.dataloader = dataloader
        self.input_name = input_name
        self.data_iter = iter(self._gen())

    def _gen(self):
        for images, _ in self.dataloader:
            # convert to numpy (N, C, H, W)
            yield {self.input_name: images.numpy()}

    def get_next(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            return None


def run_onnx_inference(onnx_path, dataloader):
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime is required to run ONNX models") from e

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    all_preds = []
    all_labels = []
    for images, labels in dataloader:
        img_np = images.numpy()
        ort_outs = sess.run(None, {input_name: img_np})
        logits = np.array(ort_outs[0])
        preds = np.argmax(logits, axis=1)
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return metrics


if __name__ == "__main__":
    cfg = load_configs()
    arch = cfg.get("arch", "tinycnn")
    dataset = cfg.get("dataset", "cifar10")
    results_dir = Path("part_2_results")
    results_dir.mkdir(exist_ok=True)

    # Create dataloaders for evaluation and calibration (small subset)
    train_loader, val_loader, num_classes = make_dataloaders(name=dataset, batch_size=32, num_workers=0, fast_dev_run=True, debug_samples=200)

    # Load trained PyTorch model
    model = make_model(arch=arch, num_classes=num_classes, pretrained=cfg.get("pretrained", False), freeze_backbone=False)
    ckpt = Path(cfg.get("results_dir", "results")) / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}, run training first")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # Export FP32 ONNX
    onnx_fp32 = results_dir / f"{arch}_{dataset}_fp32.onnx"
    sample = torch.randn(1, 3, 32, 32)
    export_onnx(model, sample, str(onnx_fp32))
    print(f"Exported FP32 ONNX to {onnx_fp32}")

    # Evaluate PyTorch FP32
    # Run on CPU using PyTorch for baseline
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    pt_metrics = compute_metrics(all_labels, all_preds)

    # Prepare summary table
    records = []
    records.append(("PyTorch FP32", pt_metrics["accuracy"], float(pt_metrics["global"].loc["macro","f1"]), float(pt_metrics["global"].loc["weighted","f1"])))

    # Try dynamic quantization using onnxruntime
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        onnx_dyn = results_dir / f"{arch}_{dataset}_int8_dynamic.onnx"
        quantize_dynamic(str(onnx_fp32), str(onnx_dyn), weight_type=QuantType.QInt8)
        print(f"Created dynamic quantized ONNX: {onnx_dyn}")
        dyn_metrics = run_onnx_inference(str(onnx_dyn), val_loader)
        records.append(("ONNX INT8 Dynamic", dyn_metrics["accuracy"], float(dyn_metrics["global"].loc["macro","f1"]), float(dyn_metrics["global"].loc["weighted","f1"])))
    except Exception as e:
        print("Dynamic quantization skipped (onnxruntime.quantization unavailable):", e)

    # Try static quantization (calibration)
    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
        onnx_static = results_dir / f"{arch}_{dataset}_int8_static.onnx"

        # Create a small calibration dataloader (already have fast_dev_run train_loader with debug_samples)
        # Convert train_loader to provide numpy batches
        # Create a CalibrationDataReader compatible wrapper
        class SimpleCalibReader(CalibrationDataReader):
            def __init__(self, dataloader, input_name):
                self.dataloader = dataloader
                self.input_name = input_name
                self.data_iter = iter(self._gen())

            def _gen(self):
                for images, _ in self.dataloader:
                    yield {self.input_name: images.numpy()}

            def get_next(self):
                try:
                    return next(self.data_iter)
                except StopIteration:
                    return None

        # Identify input name from FP32 onnx
        import onnx
        model_onnx = onnx.load(str(onnx_fp32))
        input_name = model_onnx.graph.input[0].name

        calib_reader = SimpleCalibReader(train_loader, input_name)
        quantize_static(str(onnx_fp32), str(onnx_static), calibration_data_reader=calib_reader, quant_format=None, per_channel=False, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
        print(f"Created static quantized ONNX: {onnx_static}")
        static_metrics = run_onnx_inference(str(onnx_static), val_loader)
        records.append(("ONNX INT8 Static", static_metrics["accuracy"], float(static_metrics["global"].loc["macro","f1"]), float(static_metrics["global"].loc["weighted","f1"])))
    except Exception as e:
        print("Static quantization skipped (onnxruntime.quantization or onnx unavailable):", e)

    # Write part_2_results summary
    import pandas as pd
    df = pd.DataFrame(records, columns=["Model Artifact", "Accuracy", "Macro F1", "Weighted F1"]) 
    summary_file = results_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(df.to_string(index=False))
        f.write("\n")

    print(f"Part 2: summary written to {summary_file}")
# *** End Patch
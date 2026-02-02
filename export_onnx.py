import torch
from models import make_model

def export_onnx(model, input_shape=(1,3,32,32), file_path="model.onnx"):
    """Export a loaded model to ONNX format"""
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"ONNX model exported to {file_path}")

def export_onnx_from_checkpoint(checkpoint_path, arch="tinycnn", num_classes=10, input_shape=(1,3,32,32), output_path="model.onnx"):
    """Load a saved model checkpoint and export to ONNX"""
    model = make_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    export_onnx(model, input_shape=input_shape, file_path=output_path)
    print(f"Model loaded from {checkpoint_path} and exported to {output_path}")

if __name__ == "__main__":
    # Export best model to ONNX
    export_onnx_from_checkpoint(
        checkpoint_path="results/best_model.pt",
        arch="tinycnn",
        num_classes=10,
        output_path="results/model.onnx"
    )

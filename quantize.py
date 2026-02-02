import torch
import torch.quantization as quant
from models import make_model

def dynamic_ptq(model):
    """Apply dynamic quantization to a loaded model"""
    model.eval()
    model_int8 = quant.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model_int8

def static_ptq(model, calibration_loader):
    """Apply static quantization to a loaded model"""
    model.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(model, inplace=True)
    # Run calibration
    for x, _ in calibration_loader:
        model(x)
    quant.convert(model, inplace=True)
    return model

def quantize_checkpoint_dynamic(checkpoint_path, arch="tinycnn", num_classes=10, output_path="model_quantized.pt"):
    """Load a checkpoint and apply dynamic quantization"""
    model = make_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model_quantized = dynamic_ptq(model)
    torch.save(model_quantized.state_dict(), output_path)
    print(f"Model loaded from {checkpoint_path}, quantized, and saved to {output_path}")
    return model_quantized

def quantize_checkpoint_static(checkpoint_path, calibration_loader, arch="tinycnn", num_classes=10, output_path="model_static_quantized.pt"):
    """Load a checkpoint and apply static quantization"""
    model = make_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model_quantized = static_ptq(model, calibration_loader)
    torch.save(model_quantized.state_dict(), output_path)
    print(f"Model loaded from {checkpoint_path}, quantized, and saved to {output_path}")
    return model_quantized

if __name__ == "__main__":
    # Dynamic quantization
    quantize_checkpoint_dynamic(
        checkpoint_path="results/best_model.pt",
        arch="tinycnn",
        num_classes=10,
        output_path="results/model_quantized.pt"
    )

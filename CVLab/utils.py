import torch
import onnx
import os

def load_model(model: torch.nn.Module, model_path: str, device: torch.device="cpu") -> torch.nn.Module:
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

def convert_to_onnx(model: torch.nn.Module, model_path: str) -> None:
    """
    Converts model in training mode to ONNX format to check layout.
    """
    model.eval()
    model = load_model(model, model_path)
    dummy_input = torch.randn(1, 3, 512, 512)
    onnx_name = os.path.splitext(model_path)[0] + '.onnx'
    torch.onnx.export(model, dummy_input, onnx_name,
                    export_params=True, opset_version=17, do_constant_folding=False,
                    input_names=["input"], output_names=["output"],
                    training=torch.onnx.TrainingMode.TRAINING,
                    dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}}, verbose=False)
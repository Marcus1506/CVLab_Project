import torch
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure

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

class MSE_SSIM(torch.nn.Module):
    """
    Calculates combined loss of MSE and SSIM.
    """
    def __init__(self, value_range: tuple[float, float], alpha: float=1., beta: float=1.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # Alpha and beta may need to be adjusted. SSIM is in range (-1, 1), MSE in (0, 1) if we use max normalization.
        # TODO: Unnormalize for SSIM!
        self.mse_loss = torch.nn.MSELoss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=value_range)
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1 - SSIM because 1 represents perfect similarity.
        return self.alpha * self.mse_loss(output, target) + self.beta * (1 - torch.abs(self.ssim_loss(output, target)))
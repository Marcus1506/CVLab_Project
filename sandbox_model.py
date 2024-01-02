import torch

import CVLab

from torchinfo import summary

if __name__ == "__main__":
    test_model = CVLab.models.UNet3plus(3, 1, norm='batch', activation='relu', conv_down=False)
    # test_model.init_encoder()
    
    summary(test_model, (4, 3, 512, 512))
    
    # torch.save(test_model.state_dict(), 'models/test_model.pth')
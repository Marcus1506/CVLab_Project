import torch

import CVLab

from torchinfo import summary

if __name__ == "__main__":
    test_model = CVLab.models.GUNET3plus(3, 1, norm='batch', activation='relu', conv_down=False)
    # test_model.init_encoder()
    
    summary(test_model, input_data=(torch.randn((8, 3, 512, 512)), torch.randn(8)))
    
    # torch.save(test_model.state_dict(), 'models/test_model.pth')
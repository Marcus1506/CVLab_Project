import torch
import CVLab
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    model = CVLab.models.UNet3plus(3, 1, norm='batch', activation='relu', conv_down=False)
    # model = CVLab.models.UNet(3, 1, True, conv_down=True)
    
    model.load_state_dict(torch.load('models/test_model_transfer.pt'))
    
    dataset_test = CVLab.data.CustomDataset("data/mock_data_split/Test")
    
    number_of_images = 5
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)
    
    model.eval()
    with torch.no_grad():
        for i, (test_batch, test_target_batch) in enumerate(dataset_test):
            pred = model(test_batch.unsqueeze(0))
            
            input_image = test_batch.squeeze().cpu().numpy()[0]
            pred = pred.squeeze().cpu().numpy()
            truth = test_target_batch.squeeze().cpu().numpy()
            
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            
            axs[0].imshow(input_image, vmin=0, vmax=1)
            axs[0].set_title("Focal length 0 AOS input")
            axs[0].axis('off')
            
            axs[1].imshow(pred, vmin=0, vmax=1)
            axs[1].set_title("Prediction")
            axs[1].axis('off')
            
            axs[2].imshow(truth, vmin=0, vmax=1)
            axs[2].set_title("Truth")
            axs[2].axis('off')
            
            plt.show()
            if i == number_of_images:
                break
import torch
import CVLab
import matplotlib.pyplot as plt

def psnr(true_image, pred_image):
    mse = torch.mean((true_image - pred_image) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

if __name__ == "__main__":
    
    # model = CVLab.models.UNet3plus(3, 1, norm='batch', activation='leakyrelu', conv_down=False)
    # model = CVLab.models.UNet(3, 1, True, conv_down=True)
    model = CVLab.models.GUNET3plus(3, 1, norm='batch', activation='leakyrelu', conv_down=False)
    
    model.load_state_dict(torch.load('models_finetuned/guided_transfer_batch8_leaky_full_extensive_finetuned128_MSESSIM_0.5_0.25.pt'))
    
    # dataset_test = CVLab.data.CustomDataset("data/third_data_split/Test")
    dataset_test = CVLab.data.Guided_Dataset("data/whole_data_split/Test")
    
    number_of_images = 40
    
    # model.eval()
    # with torch.no_grad():
    #     for i, (test_batch, test_target_batch) in enumerate(dataset_test):
    #         pred = model(test_batch.unsqueeze(0))
            
    #         input_image = test_batch.squeeze().cpu().numpy()[0]
    #         pred = pred.squeeze().cpu().numpy()
    #         truth = test_target_batch.squeeze().cpu().numpy()
            
    #         fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            
    #         axs[0].imshow(input_image, vmin=0, vmax=1, cmap='gray')
    #         axs[0].set_title("Focal length 0 AOS input")
    #         axs[0].axis('off')
            
    #         axs[1].imshow(pred, vmin=0, vmax=1, cmap='gray')
    #         axs[1].set_title("Prediction")
    #         axs[1].axis('off')
            
    #         axs[2].imshow(truth, vmin=0, vmax=1, cmap='gray')
    #         axs[2].set_title("Truth")
    #         axs[2].axis('off')
            
    #         plt.show()
    #         if i == number_of_images:
    #             break
    
    model.eval()
    with torch.no_grad():
        for i, (test_batch, test_target_batch) in enumerate(dataset_test):
            image_batch, temp_batch = test_batch
            pred = model(image_batch.unsqueeze(0), temp_batch)
            
            input_image = image_batch.squeeze().cpu().numpy()[0]
            pred = pred.squeeze().cpu().numpy()
            truth = test_target_batch.squeeze().cpu().numpy()
            
            psnr_value = psnr(torch.tensor(truth), torch.tensor(pred))

            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            
            axs[0].imshow(input_image, vmin=0, vmax=1, cmap='gray')
            axs[0].set_title("Focal length 0 AOS input")
            axs[0].axis('off')
            
            axs[1].imshow(pred, vmin=0, vmax=1, cmap='gray')
            axs[1].set_title(f"Prediction: PSNR={psnr_value:.2f}")
            axs[1].axis('off')
            
            axs[2].imshow(truth, vmin=0, vmax=1, cmap='gray')
            axs[2].set_title("Truth")
            axs[2].axis('off')
            
            plt.savefig(f"auxiliary_images/finetuned/GUNET3plus_finetuned_{i}.png")
            plt.close()
            if i == number_of_images:
                break
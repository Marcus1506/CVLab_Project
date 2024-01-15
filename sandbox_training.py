import torch
import CVLab

if __name__ == "__main__":
    # dataset_train = CVLab.data.CustomDataset("data/third_data_split/Train")
    # dataset_eval = CVLab.data.CustomDataset("data/third_data_split/Eval")
    # dataset_test = CVLab.data.CustomDataset("data/third_data_split/Test")

    dataset_train = CVLab.data.Guided_Dataset("data/third_data_split/Train")
    dataset_eval = CVLab.data.Guided_Dataset("data/third_data_split/Eval")
    dataset_test = CVLab.data.Guided_Dataset("data/third_data_split/Test")


    # model = CVLab.models.UNet3plus(3, 1, norm='batch', activation='leakyrelu', conv_down=False)
    # model.init_encoder()
    # model = CVLab.models.UNet(3, 1, True, conv_down=True)
    
    model = CVLab.models.GUNET3plus(3, 1, norm='batch', activation='leakyrelu', conv_down=False)
    model.unet.init_encoder()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    loss = torch.nn.MSELoss()
    # loss = CVLab.utils.MSE_SSIM((0, 1), 1., 0.125)

    CVLab.utils.std_training_loop_tuple(
        model, train_data=dataset_train, val_data=dataset_eval, num_epochs=100, model_name="test_third_model_guided_transfer_batch8_leaky_extended",
        optimizer=optimizer, loss_function=loss, minibatch_size=8, show_progress=True, try_cuda=True, early_stopping=True,
        patience=20, model_path="models_third", losses_path="losses", workers=4, pin_memory=True, prefetch_factor=3, true_random=True
        )
    
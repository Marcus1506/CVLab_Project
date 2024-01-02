import torch
import CVLab

if __name__ == "__main__":
    dataset_train = CVLab.data.CustomDataset("data/mock_data_split/Train")
    dataset_eval = CVLab.data.CustomDataset("data/mock_data_split/Eval")
    dataset_test = CVLab.data.CustomDataset("data/mock_data_split/Test")
    
    model = CVLab.models.UNet3plus(3, 1, norm='batch', activation='relu', conv_down=False)
    model.init_encoder()
    # model = CVLab.models.UNet(3, 1, True, conv_down=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    loss = torch.nn.MSELoss()
    
    CVLab.utils.std_training_loop(
        model, train_data=dataset_train, val_data=dataset_eval, num_epochs=10, model_name="test_model_transfer",
        optimizer=optimizer, loss_function=loss, minibatch_size=2, show_progress=True, try_cuda=True, early_stopping=True,
        patience=3, model_path="models", losses_path="losses", workers=3, pin_memory=True, prefetch_factor=3, true_random=True
        )
    
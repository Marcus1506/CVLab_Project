"""
This file is for computing losses of the final model on the different datasets.
"""

import torch
import CVLab

if __name__ == "__main__":
    dataset_train = CVLab.data.Guided_Dataset("data/whole_data_split/Train")
    dataset_eval = CVLab.data.Guided_Dataset("data/whole_data_split/Eval")
    dataset_test = CVLab.data.Guided_Dataset("data/whole_data_split/Test")

    model = CVLab.models.GUNET3plus(3, 1, norm='batch', activation='leakyrelu', conv_down=False)
    model.load_state_dict(torch.load('models_finetuned/guided_transfer_batch8_leaky_full_extensive_finetuned128_MSESSIM_0.5_0.25.pt'))

    mse = torch.nn.MSELoss()
    mse_ssim = CVLab.utils.MSE_SSIM((0, 1), 0.5, 0.25)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=3)
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=3)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=3)

    model.to(torch.device("cuda"))

    train_loss_mse = CVLab.utils.train_tuple(model, dataloader_train, mse, True, torch.device("cuda"), True, 1, None)
    eval_loss_mse = CVLab.utils.train_tuple(model, dataloader_eval, mse, True, torch.device("cuda"), True, 1, None)
    test_loss_mse = CVLab.utils.train_tuple(model, dataloader_test, mse, True, torch.device("cuda"), True, 1, None)

    train_loss_mse_ssim = CVLab.utils.train_tuple(model, dataloader_train, mse_ssim, True, torch.device("cuda"), True, 1, None)
    eval_loss_mse_ssim = CVLab.utils.train_tuple(model, dataloader_eval, mse_ssim, True, torch.device("cuda"), True, 1, None)
    test_loss_mse_ssim = CVLab.utils.train_tuple(model, dataloader_test, mse_ssim, True, torch.device("cuda"), True, 1, None)

    train_loss_mse = torch.mean(torch.stack(train_loss_mse))
    eval_loss_mse = torch.mean(torch.stack(eval_loss_mse))
    test_loss_mse = torch.mean(torch.stack(test_loss_mse))

    train_loss_mse_ssim = torch.mean(torch.stack(train_loss_mse_ssim))
    eval_loss_mse_ssim = torch.mean(torch.stack(eval_loss_mse_ssim))
    test_loss_mse_ssim = torch.mean(torch.stack(test_loss_mse_ssim))

    # Write losses to .txt file
    with open("losses.txt", "w") as f:
        f.write(f"Train loss MSE: {train_loss_mse}\n")
        f.write(f"Eval loss MSE: {eval_loss_mse}\n")
        f.write(f"Test loss MSE: {test_loss_mse}\n")
        f.write(f"Train loss MSE SSIM: {train_loss_mse_ssim}\n")
        f.write(f"Eval loss MSE SSIM: {eval_loss_mse_ssim}\n")
        f.write(f"Test loss MSE SSIM: {test_loss_mse_ssim}\n")
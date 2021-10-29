from reprod_log import ReprodDiffHelper


if __name__ == "__main__":
    import os
    print(os.getcwd())

    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("torch/4_step_loss_torch.npy")
    paddle_info = diff_helper.load_info("paddle/4_step_loss_pd.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="loss_check.log")
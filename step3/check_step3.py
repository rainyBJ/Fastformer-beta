from reprod_log import ReprodDiffHelper


if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("torch/best_val_torch.npy")
    paddle_info = diff_helper.load_info("paddle/best_val_pd.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="best_val.log")
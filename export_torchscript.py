import argparse

import torch

from loftr import LoFTR, default_cfg
from utils import make_student_config


def main():
    parser = argparse.ArgumentParser(description="Full LoFTR Torchscript Export")
    parser.add_argument(
        "--out_file",
        type=str,
        default="weights/outdoor_ds.torchscript",
        help="Path for the output torchscript model.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/outdoor_ds.ckpt",
        help="Path to network weights.",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="If specified the original LoFTR model will be used.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")

    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device)

    if opt.original:
        model_cfg = default_cfg
    else:
        model_cfg = make_student_config(default_cfg)

    print("Loading pre-trained network...")
    model = LoFTR(config=model_cfg)
    checkpoint = torch.load(opt.weights)

    if checkpoint is not None:
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint["model_state_dict"]
        missed_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missed_keys) > 0:
            print("Checkpoint is broken")
            return 1
        print("Successfully loaded pre-trained weights.")
    else:
        print("Failed to load checkpoint. Using random weights...")
        # return 1

    print(f"Moving model to device: {device}")
    model = model.eval().to(device=device)

    with torch.no_grad():
        # with torch.jit.optimized_execution(True):
        import cv2

        cv_img1 = cv2.imread("imgs/eiffel1.jpg", 0)
        cv_img1 = cv2.resize(cv_img1, (640, 480))
        img1 = torch.tensor(cv_img1).unsqueeze(0).unsqueeze(0).to(device=device) / 255.0

        cv_img2 = cv2.imread("imgs/eiffel2.jpg", 0)
        cv_img2 = cv2.resize(cv_img2, (640, 480))
        img2 = torch.tensor(cv_img2).unsqueeze(0).unsqueeze(0).to(device=device) / 255.0

        torch_output = model(img1, img2)

        jit_model = torch.jit.trace(model, (img1, img2))
        jit_output = jit_model(img1, img2)

        for i in range(len(torch_output)):
            assert torch.allclose(
                torch_output[i], jit_output[i], atol=1e-1
            ), f"Output {i} is not equal: {torch_output[i]} vs {jit_output[i]}"
    print("Pass JIT Tracing")

    jit_model.save(opt.out_file)
    return jit_model


if __name__ == "__main__":
    main()

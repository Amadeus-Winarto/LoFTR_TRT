import argparse

import torch
import torch.nn.utils.prune as prune
from torch.profiler import profile, record_function, ProfilerActivity

from loftr import LoFTR, default_cfg
from utils import make_student_config


def main():
    parser = argparse.ArgumentParser(description="LoFTR demo.")
    parser.add_argument(
        "--out_file",
        type=str,
        default="weights/LoFTR_teacher.onnx",
        help="Path for the output ONNX model.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/LoFTR_teacher.pt",  # weights/outdoor_ds.ckpt
        help="Path to network weights.",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="If specified the original LoFTR model will be used.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--prune", default=False, help="Do unstructured pruning")

    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device)

    if opt.original:
        model_cfg = default_cfg
    else:
        model_cfg = make_student_config(default_cfg)

    print("Loading pre-trained network...")
    model = LoFTR(config=model_cfg)
    # checkpoint = torch.load(opt.weights)
    checkpoint = None
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

    if opt.prune:
        print("Model pruning")
        for name, module in model.named_modules():
            # prune connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name="weight", amount=0.5)
                prune.remove(module, "weight")
            # prune connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=0.5)
                prune.remove(module, "weight")
        weight_total_sum = 0
        weight_total_num = 0
        for name, module in model.named_modules():
            # prune connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                weight_total_sum += torch.sum(module.weight == 0)
            # prune connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                weight_total_num += module.weight.nelement()

        print(f"Global sparsity: {100. * weight_total_sum / weight_total_num:.2f}")

    print(f"Moving model to device: {device}")
    model = model.eval().to(device=device)

    with torch.no_grad():
        dummy_image = torch.randn(
            1, 1, default_cfg["input_height"], default_cfg["input_width"], device=device
        )
        torch_output = model(dummy_image, dummy_image)

        jit_model = torch.jit.trace(
            model,
            (dummy_image, dummy_image),
        )
        jit_output = jit_model(dummy_image, dummy_image)
        assert torch.allclose(torch_output[0], jit_output[0], atol=1e-5)
    print("Pass JIT Tracing")
    return jit_model

    # model = onnx.load(opt.out_file)
    # onnx.checker.check_model(model)


if __name__ == "__main__":
    # 1. Check Tracing
    jit_model = main()

    # 2. Profile the model
    """
    Profiling the model with torch.profiler
    Default settings: 
    Input size: 1x1x240x320
    Resolution: 8, 2
    Attention: Linear
    Backbone: ResNetFPN
    Note: All measurements are in ms
    
    Attention Mechanism
    method |  CPU  |  GPU  |  JIT
    ----------------------------
    Linear | 1496  | 110.4 | 22.172
    AFT    | 1255  | 125.6 | 17.227
    

    Backbone
    method |  CPU  |  GPU  |  JIT
    ----------------------------
    ConvNeXT |  2040  |  80.735  |  20.287
    ResNetFPN |  1796  |  67.020  |  20.646
    """

    device = "cpu"
    jit_model = LoFTR(config=default_cfg).to(device=device)
    dummy_image = torch.randn(1, 1, 240, 320).to(device=device)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            jit_model(dummy_image, dummy_image)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

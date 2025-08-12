import argparse
import os
import random
import numpy as np
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", help = "target image path")
    parser.add_argument("--output_dir", type = str, default = "")
    parser.add_argument("--render_size", default = 512, type = int, help = "render size")
    parser.add_argument("--use_cpu", type = int, default = 0)
    parser.add_argument("--num_iter", type = int, default = 2000)

    parser.add_argument("--num_strokes", type = int, default = 32)
    parser.add_argument("--num_segments", type = int, default = 1)


    # attention map for stroke init
    parser.add_argument("--use_init_method", type=int, default = 1)

    args = parser.parse_args()

    test_name = os.path.splitext(os.path.basename(args.target))[0]
    output_dir = f"{args.output_dir}/{test_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    args.output_dir = output_dir
    
    use_gpu = not args.use_cpu
    if not torch.cuda.is_available():
        use_gpu = False
    if use_gpu:
        args.device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    else:
        args.device = torch.device("cpu")

    return args



if __name__ = "__main__":
    args = parse_arguments()

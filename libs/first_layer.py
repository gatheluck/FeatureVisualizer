import logging
import os
import sys

import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base)

from fft import fft_shift


def extract_target_modules(
    model, target_module_name: str = "torch.nn.Conv2d", is_orator: bool = True
) -> list:
    """
    return list of specified modules which is included in the given model.
    """
    model.eval()

    target_modules = [
        module
        for module in model.modules()
        if isinstance(module, eval(target_module_name))
    ]

    # log info
    if is_orator:
        logging.info(
            "extract_target_module: found {num} [{name}] modules.".format(
                num=len(target_modules), name=target_module_name
            )
        )
    return target_modules


def save_first_layer_weight(
    model, log_path: str, bias: int = 0.5, is_orator: bool = True, **kwargs
) -> None:
    """
    save weight of first conv as images.
    """
    model.eval()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    conv2d_modules = extract_target_modules(model, target_module_name="torch.nn.Conv2d")
    first_conv2d_weight = conv2d_modules[0].weight + bias

    torchvision.utils.save_image(first_conv2d_weight, log_path, padding=1)

    # log info
    if is_orator:
        logging.info(
            "save_first_layer_weight: images are saved under [{log_dir}]".format(
                log_dir=os.path.dirname(log_path)
            )
        )


def save_first_layer_weight_freq(
    model, log_path: str, bias: int = 0.5, is_orator: bool = True, **kwargs
) -> None:
    model.eval()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    conv2d_modules = extract_target_modules(model, target_module_name="torch.nn.Conv2d")
    first_conv2d_weight = conv2d_modules[0].weight  # + bias
    # first_conv2d_weight = first_conv2d_weight.clamp(min=0.0, max=1.0)

    freq_image = first_conv2d_weight.rfft(2, normalized=True, onesided=False)
    freq_image_shifted_rgb = fft_shift(freq_image).norm(
        dim=-1
    )  # normalize complex number: [b, c, h, w]

    freq_image_shifted_gray = freq_image_shifted_rgb.norm(dim=-3, keepdim=True)

    torchvision.utils.save_image(
        freq_image_shifted_rgb,
        os.path.splitext(log_path)[0] + "_rgb" + os.path.splitext(log_path)[1],
        padding=1,
    )
    torchvision.utils.save_image(
        freq_image_shifted_gray,
        os.path.splitext(log_path)[0] + "_gray" + os.path.splitext(log_path)[1],
        padding=1,
    )
    save_intensity_per_wave_number(freq_image_shifted_gray, log_path)

    # log info
    if is_orator:
        logging.info(
            "save_first_layer_weight_freq: images are saved under [{log_dir}]".format(
                log_dir=os.path.dirname(log_path)
            )
        )


def save_intensity_per_wave_number(input_tensor: torch.Tensor, savepath: str) -> None:
    """
    Args:
    - input_tensor:
    - savepath:
    """
    wave_num = approximate_wave_numbers(input_tensor.size(-2), input_tensor.size(-1)).float()  # [h, w]
    wave_num = wave_num[None, None, :, :].repeat(input_tensor.size(0), 1, 1, 1)  # [b, 1, h, w]

    batch_idx = torch.tensor([i for i in range(input_tensor.size(0))]).float()  # [b]
    batch_idx = batch_idx[:, None, None, None].repeat(1, 1, input_tensor.size(-2), input_tensor.size(-1))  # [b, 1, h, w]

    x = torch.cat([batch_idx, wave_num, input_tensor], dim=1)  # [b, 3, h, w]
    x = x.permute(0, 2, 3, 1).reshape(-1, 3)  # [b*h*w, 3]

    df = pd.DataFrame(x.detach().numpy())
    df = df.rename(columns={0: 'batch', 1: 'wave_number', 2: 'intensity'})
    intensity = df.groupby(['wave_number'])['intensity'].mean()
    intensity = 100.0 * intensity / intensity.sum()
    print(intensity)

    y = intensity.values.tolist()
    x = np.arange(len(y))
    xticks = intensity.index.values.tolist()

    plt.bar(x, y)
    for i, j in zip(x, y):
        plt.text(i, j, f'{j:.1f}', ha='center', va='bottom', fontsize=7)

    plt.ylabel('Rate (%)')

    plt.ylim(0, 40)

    plt.xticks(x, xticks, rotation=90)
    plt.yticks(np.linspace(0, 40, 11))

    plt.subplots_adjust(bottom=0.3)
    plt.grid(axis='y')
    plt.savefig(os.path.splitext(savepath)[0] + "_intensity" + os.path.splitext(savepath)[1])
    plt.close()


def approximate_wave_numbers(size_h: int, size_w: int) -> torch.Tensor:
    """
    Args
    - size_h: hight of tensor.
    - size_w: width of tensor.
    Returns
    - grid_wave_num: tensor of approximated wave numbers.
    """
    h = torch.tensor([i for i in range(size_h)])
    w = torch.tensor([i for i in range(size_w)])
    grid_h, grid_w = torch.meshgrid(h, w)

    grid_w = grid_w - ((size_w - 1) / 2.0)  # [h, w]
    grid_h = grid_h - ((size_h - 1) / 2.0)  # [h, w]

    grid_hw = torch.stack([grid_h, grid_w])  # [2, h, w]
    grid_wave_num = grid_hw.norm(dim=0)  # [h, w]
    grid_wave_num = grid_wave_num - grid_wave_num.min()

    return grid_wave_num


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--arch', type=str, required=True, choices=['resnet50'])
    parser.add_argument('-n', '--num_classes', type=int, required=True)
    parser.add_argument('-w', '--weight', type=str, required=True)
    parser.add_argument('-l', '--logdir', type=str, required=True)

    opt = parser.parse_args()

    # prepare model
    if opt.arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False, num_classes=opt.num_classes)
    else:
        raise NotImplementedError
    trained_weight = torch.load(opt.weight)
    model.load_state_dict(trained_weight)

    # prepare logdir
    os.makedirs(opt.logdir, exist_ok=True)

    # extract_target_modules
    modules = extract_target_modules(model)

    # save_first_layer_weight
    save_first_layer_weight(model, os.path.join(opt.logdir, 'first_layer_weight.png'))

    # test save_first_layer_weight_freq
    save_first_layer_weight_freq(model, os.path.join(opt.logdir, 'first_layer_weight_freq.png'))

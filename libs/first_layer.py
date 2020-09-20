import logging
import os
import sys

import torchvision

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
    first_conv2d_weight = conv2d_modules[0].weight + bias
    first_conv2d_weight = first_conv2d_weight.clamp(min=0.0, max=1.0)

    freq_image = first_conv2d_weight.rfft(2, normalized=True, onesided=False)
    freq_image_shifted_rgb = fft_shift(freq_image).norm(
        dim=-1
    )  # normalize complex number: [b, c, h, w]

    freq_image_shifted_gray = freq_image_shifted_rgb.sum(dim=-3, keepdim=True)

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

    # log info
    if is_orator:
        logging.info(
            "save_first_layer_weight_freq: images are saved under [{log_dir}]".format(
                log_dir=os.path.dirname(log_path)
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = torchvision.models.resnet50(pretrained=True)

    # test extract_target_modules
    modules = extract_target_modules(model)

    # test save_first_layer_weight
    save_first_layer_weight(model, "../logs/first_layer_weight.png")

    # test save_first_layer_weight_freq
    save_first_layer_weight_freq(model, "../logs/first_layer_weight_freq.png")

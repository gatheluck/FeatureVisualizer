import torch


def fft_shift(input: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of np.fftshift
    Args
    - input: (Bx)CxHxWx2

    Return
    - ret: (Bx)CxHxWx2
    """
    dims = [i for i in range(1 if input.dim() == 4 else 2, input.dim() - 1)]  # H, W
    shift = [input.size(dim) // 2 for dim in dims]
    return torch.roll(input, shift, dims)


def ifft_shift(input: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of np.ifftshift
    Args
    - input: (Bx)CxHxWx2

    Return
    - ret: (Bx)CxHxWx2
    """
    dims = [i for i in range(input.dim() - 2, 0 if input.dim() == 4 else 1, -1)]  # H, W
    shift = [input.size(dim) // 2 for dim in dims]
    return torch.roll(input, shift, dims)

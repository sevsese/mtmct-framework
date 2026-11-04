"""Print diagnostic environment info, fail early if CUDA unavailable."""
from __future__ import annotations


def main():
    import sys
    import torch
    print('Python:', sys.version.split()[0])
    print('PyTorch:', torch.__version__)
    print('PyTorch CUDA runtime:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit(
            'CUDA НЕ работает. Открой https://pytorch.org/get-started/locally/, '
            'выбери Windows / Pip / Python / CUDA и переустанови torch/torchvision/torchaudio '
            'в активном conda-окружении. Проверь драйвер командой nvidia-smi.'
        )
    device = torch.device('cuda:0')
    print('GPU:', torch.cuda.get_device_name(device))
    print('VRAM (GiB):', round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 2))
    a = torch.randn(512, 512, device=device)
    b = a @ a.T
    torch.cuda.synchronize()
    print('GPU math test:', b.shape, 'OK')

if __name__ == '__main__':
    main()

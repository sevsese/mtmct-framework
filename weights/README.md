# Модели для обработки видео

Офисный сценарий и веб-обработка используют детектор `yolo26l.pt` в корне
проекта и OSNet IBN x1.0 в этом каталоге. Для просмотра готовых результатов
достаточно [установки приложения](../docs/START_HERE.md#1-установить-и-открыть-приложение).

В PowerShell из корня проекта выберите одну сборку PyTorch. Первая команда
использует CUDA 12.4 и требует совместимого NVIDIA-драйвера, вторая — CPU.

```powershell
./.venv/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
./.venv/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

Затем установите зависимости трекинга и подготовьте веса. Эти команды скачивают
пакеты и модели при первом запуске.

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements.txt "lap>=0.5.12"
./.venv/Scripts/python.exe -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
./.venv/Scripts/python.exe -c "from ultralytics import YOLO; YOLO('yolo26l.pt')"
./.venv/Scripts/python.exe -m mtmct.weights --model osnet_ibn_x1_0
```

Если файлы весов уже есть, положите `yolo26l.pt` в корень, а checkpoint
`osnet_ibn_x1_0_msmt17_*.pth` в `weights/`. Сохраните оригинальное имя OSNet.
В активированном Conda-окружении используйте `python` вместо полного пути.
Под Linux путь к Python в виртуальном окружении — `.venv/bin/python`.
Веса остаются локальными.

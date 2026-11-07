# Запуск на компьютере

Откройте терминал VS Code в корне проекта и выполните:

```powershell
conda activate neuro_clean
python run.py app
```

Откройте http://127.0.0.1:8000/viewer . Для остановки сервера нажмите Ctrl+C.
Также можно дважды щёлкнуть `run_web.cmd` в корне: он ищет окружение neuro_clean.
Если порт уже занят запущенным приложением, откройте его существующий адрес.

Если интерфейс ещё не собран:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
cd frontend
pnpm install --frozen-lockfile
pnpm build
cd ..
python run.py app
```

Для новой машины дополнительно установите совместимый с вашей CUDA PyTorch
и ReID-зависимости согласно существующей инструкции WEB_INTERFACE.md.
В текущем neuro_clean зависимости уже установлены.

## Обработка офиса

```powershell
python run.py office --seconds 120 --run-name office_my_test_01
```

Для трёх минут задайте `--seconds 180`. Каждый запуск требует нового имени.
Вход: пять файлов `camera_new_1.mkv` … `camera_new_5.mkv` в `data/Hackaton Data/`.
Тот же каталог доступен локально как `data/hackaton/`.
Настройки алгоритма находятся в `run_hackaton_experiment.py`, конфигурации — в `configs/`.

После завершения откройте `runs/<имя>/mosaic.mp4` или импортируйте результат
в приложении: «Новый набор» → «Открыть результат из проекта» → нужный каталог `runs`.
Папка `run/` даёт быстрый доступ к двум готовым примерам.

## Работа в VS Code

Выберите интерпретатор окружения neuro_clean. Для отдельных экспериментов
откройте нужный файл из `launchers/`, измените настройки и нажмите Run Python File.
Для общего run.py используйте терминал с аргументами, как показано выше.

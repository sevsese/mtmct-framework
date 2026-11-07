# Программы запуска

Запускайте команды из корня проекта в окружении `neuro_clean`.

Основные команды: `python run.py app` и `python run.py office --seconds 120 --run-name office_new`.

В этой папке находятся отдельные экспериментальные программы:

| Программа | Назначение |
|---|---|
| run_dataset.py | Выбор датасета и параметров |
| run_epfl_lab.py, run_epfl_lab_fixed.py | EPFL |
| run_wildtrack_experiment.py | WILDTRACK |
| run_compare_all_osnet.py, run_reid_comparison.py | Сравнение ReID |
| run_nightly_optimize.py | Длительный поиск параметров |
| run_best_4cameras.py | Запуск выбранных камер NVIDIA |
| run_external_validation.py | Проверка на других датасетах |
| run_cached.py | Старый эксперимент повторной ассоциации из кэша |
| diagnose_*.py | Диагностика результатов и камер |

Пример: `python launchers/run_dataset.py`. Настройки меняются внутри выбранного
файла. Исторические инструкции с именем файла в корне следует читать с префиксом
`launchers/`. Три совместимые точки входа пока сохранены в корне:
`run_hackaton_experiment.py`, `run_wildtrack.py`, `run_wildtrack_fixed.py`.

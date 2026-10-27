# Модель вероятности дефолта (UCI Credit Card Default)

Сессионный проект по курсу «Автоматизация процессов разработки и тестирования моделей ML». Модель оценивает вероятность дефолта клиента по кредитной карте, а конвейер вокруг неё готовит данные, проверяет их, обучает модель и отдаёт прогноз через API.

[![CI](https://github.com/a-kapset/MLOps_DevAndTest_SessionProject/actions/workflows/ci.yml/badge.svg)](https://github.com/a-kapset/MLOps_DevAndTest_SessionProject/actions/workflows/ci.yml)

## Что сделано

Конвейер проходит путь от исходного CSV до сервиса в Docker, каждый этап запускается одной командой:

1. `validate`: Great Expectations проверяет исходные данные 70 правилами и останавливает пайплайн при аномалиях.
2. `prepare`: очистка, удаление дубликатов, стратифицированное разбиение на train и test.
3. `eda`: графики разведочного анализа, выводы и решения для модели в [reports/eda.md](reports/eda.md).
4. `train`: sklearn Pipeline с доменными признаками, подбор гиперпараметров логистической регрессии и случайного леса через GridSearchCV, трекинг всех экспериментов в MLflow, сохранение лучшей модели.
5. REST API на FastAPI: `POST /predict` возвращает класс и вероятность дефолта.
6. Docker-образ с API и моделью, скрипты сборки и запуска для PowerShell и bash.

Стадии 1–4 описаны в `dvc.yaml`, данные и модель версионируются DVC в хранилище на DagsHub. Unit-тесты (pytest) покрывают очистку, разбиение, правила валидации, признаки, Pipeline, метрики и API. CI в GitHub Actions на каждый push проверяет форматирование, линтинг и типы, запускает тесты и валидирует реальные данные.

Итог: лучшая модель, случайный лес, даёт ROC-AUC 0,787 на кросс-валидации и 0,777 на тестовой выборке. Сравнение моделей приведено в разделе «Эксперименты MLflow».

## Демонстрация

**Без запуска.** Результаты лежат в репозитории и на DagsHub:

- [reports/eda.md](reports/eda.md): разведочный анализ с графиками и решениями для модели;
- [reports/metrics.json](reports/metrics.json) и [reports/roc_curve.png](reports/roc_curve.png): параметры, метрики и ROC-кривая лучшей модели;
- [reports/validation.json](reports/validation.json): результат каждого правила валидации;
- [эксперименты MLflow на DagsHub](https://dagshub.com/artem_vk/MLOps_DevAndTest_SessionProject.mlflow/#/experiments/0): 2 родительских и 7 вложенных runs с параметрами, метриками, ROC-кривыми и моделями, вход не нужен;
- [запуски CI](https://github.com/a-kapset/MLOps_DevAndTest_SessionProject/actions/workflows/ci.yml).

**С запуском.** Команды ниже проверены на свежем клоне в PowerShell; справа указан ожидаемый вывод. Данные получаются одним из двух способов из раздела «Данные», здесь показан первый, с токеном DagsHub.

```powershell
git clone https://github.com/a-kapset/MLOps_DevAndTest_SessionProject.git
cd MLOps_DevAndTest_SessionProject
poetry install
poetry run dvc remote modify --local origin auth basic
poetry run dvc remote modify --local origin user <логин DagsHub>
poetry run dvc remote modify --local origin password <токен DagsHub>
poetry run dvc pull                  # 4 files fetched: исходный CSV, train.csv, test.csv, модель
poetry run dvc status                # Data and pipelines are up to date.
poetry run pytest                    # 32 passed
.\scripts\docker_build.ps1           # Built image: credit-default-api
.\scripts\docker_run.ps1             # health: ok
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/client.json"
                                     # {"prediction":0,"probability":0.125050130403816}
docker stop credit-default-api
```

После `dvc pull` все стадии актуальны, и `dvc repro` ничего не пересчитывает. Чтобы увидеть пайплайн в работе, `poetry run dvc repro -f` перезапускает все стадии принудительно, это занимает около минуты:

```
validate: success=True, checks=70, failed=0
prepare: train=23972, test=5993, default_rate=0.2213
EDA: figures=3, dir=reports/figures
train: best=random_forest, cv_roc_auc=0.7867, test_roc_auc=0.7765
```

Runs этого запуска сохраняются в локальном MLflow и открываются командой `poetry run mlflow ui`.

## Стек

| Инструмент | Зачем |
|---|---|
| Python 3.12, Poetry | окружение и зависимости; `poetry.lock` фиксирует точные версии всех пакетов, включая транзитивные |
| pandas, scikit-learn | обработка данных, Pipeline модели, подбор гиперпараметров |
| joblib | сохранение обученного Pipeline в файл |
| matplotlib | графики разведочного анализа и ROC-кривая |
| FastAPI, pydantic, uvicorn | REST API модели, проверка входных данных по схеме, ASGI-сервер |
| Docker | образ сервиса: Python 3.12, зависимости основной группы из `poetry.lock`, модель |
| MLflow | трекинг экспериментов: параметры, метрики, ROC-кривая и модель; сервер MLflow на DagsHub |
| Great Expectations | правила валидации исходных данных |
| DVC, DagsHub | версии данных, воспроизводимый пайплайн, удалённое хранилище |
| pytest, pytest-cov | unit-тесты и покрытие |
| ruff | линтинг (правила flake8: pycodestyle и pyflakes, а также isort, pyupgrade, bugbear) и форматирование, совместимое с black |
| mypy | проверка типов в строгом режиме |
| pre-commit, gitleaks | проверки перед каждым коммитом, поиск секретов |
| GitHub Actions | CI |

## Структура

```
├── .github/workflows/ci.yml        # CI: линтинг, проверка типов, тесты, валидация данных
├── data/
│   ├── raw/                        # исходный CSV (в git только указатель .dvc)
│   └── processed/                  # train.csv и test.csv, выходы стадии prepare
├── models/                         # model.joblib, выход стадии train (в git только через dvc.lock)
├── reports/
│   ├── validation.json             # результат каждой проверки стадии validate
│   ├── eda.md                      # разведочный анализ: выводы и решения для модели
│   ├── figures/                    # графики стадии eda
│   ├── metrics.json                # параметры и метрики лучшей модели
│   └── roc_curve.png               # ROC-кривая лучшей модели на тестовой выборке
├── src/credit_default/
│   ├── config.py                   # пути проекта и чтение params.yaml
│   ├── data/validation.py          # стадия validate: правила Great Expectations
│   ├── data/make_dataset.py        # стадия prepare: загрузка, очистка, разбиение
│   ├── analysis/eda.py             # стадия eda: графики по обучающей выборке
│   ├── features/build_features.py  # доменные признаки
│   ├── models/pipeline.py          # sklearn Pipeline: признаки, предобработка, классификатор
│   ├── models/evaluate.py          # метрики и ROC-кривая
│   ├── models/train.py             # стадия train: GridSearchCV и MLflow
│   ├── api/schemas.py              # схемы запроса и ответа API
│   └── api/app.py                  # приложение FastAPI: /health, /predict
├── examples/client.json            # пример запроса к /predict
├── scripts/                        # сборка и запуск Docker-образа (.ps1 и .sh)
├── Dockerfile, .dockerignore       # образ сервиса
├── tests/                          # unit-тесты
├── dvc.yaml, dvc.lock              # описание пайплайна и зафиксированные хэши
├── params.yaml                     # параметры стадий, включая сетки гиперпараметров
├── .env.example                    # переменные для сервера MLflow на DagsHub
└── pyproject.toml, poetry.lock     # зависимости и настройки инструментов
```

Код оформлен пакетом `credit_default` внутри `src/`. Пакет устанавливается в окружение, поэтому импорты одинаково работают в тестах и стадиях DVC без правки `sys.path`. Анализ данных оформляется скриптами `.py`, а не ноутбуками: скрипты проверяют ruff, mypy и CI, а результаты воспроизводятся одной командой.

## Установка

Нужны Python 3.12 и Poetry 2.x.

```bash
git clone https://github.com/a-kapset/MLOps_DevAndTest_SessionProject.git
cd MLOps_DevAndTest_SessionProject
poetry install
```

`poetry install` ставит все группы зависимостей: основную (pandas, scikit-learn, joblib, PyYAML, FastAPI, uvicorn), на которой работает API, `train` (DVC, matplotlib, Great Expectations, MLflow), `test` (pytest, httpx) и `lint` (ruff, mypy, pre-commit). Отдельные группы ставятся флагом `--only`, например `poetry install --only main,train,test`.

## Данные

Набор данных: [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset), 30 000 клиентов тайваньского банка, 23 признака и целевая переменная. В git хранится только указатель `data/raw/UCI_Credit_Card.csv.dvc` с MD5 файла `940b416bb13a9b24bb5c9e1589284005`, сам файл лежит в хранилище DVC на DagsHub.

DagsHub не отдаёт данные без входа даже из публичного репозитория (ответ 401), поэтому есть два способа их получить.

**Способ 1. Через DVC с аккаунтом DagsHub.** Аккаунт бесплатный, вход возможен через GitHub. Токен берётся в настройках профиля DagsHub (Settings → Tokens). Учётные данные записываются в `.dvc/config.local`, этот файл не попадает в git.

```bash
poetry run dvc remote modify --local origin auth basic
poetry run dvc remote modify --local origin user <логин DagsHub>
poetry run dvc remote modify --local origin password <токен DagsHub>
poetry run dvc pull
```

`dvc pull` скачивает исходный CSV, готовые `train.csv` и `test.csv` и обученную модель `models/model.joblib`.

**Способ 2. Без аккаунта DagsHub.** Скачать `UCI_Credit_Card.csv` с Kaggle по ссылке выше, положить в `data/raw/` и собрать выборки заново:

```bash
poetry run dvc repro
```

Файл с Kaggle совпадает с версией в DVC по MD5. `dvc repro` выполняет `prepare` и `train` (около двух минут), а `validate` и `eda` пропускает: их результаты уже есть в git. Модель обучается заново, runs записываются в локальный MLflow. В проверочном прогоне на свежем клоне выборки, модель и метрики совпали с версией в хранилище DVC побайтно.

## Пайплайн

```bash
poetry run dvc repro     # пересчитать устаревшие стадии
poetry run dvc status    # показать, что изменилось с последнего запуска
poetry run dvc dag       # граф стадий: validate → prepare → eda и prepare → train
```

Стадия `validate` (`src/credit_default/data/validation.py`) проверяет исходный CSV набором из 70 правил Great Expectations:

- полнота: все 25 колонок на месте, лишних нет, пропусков нет;
- типы: во всех колонках только числа;
- уникальность: `ID` не повторяется;
- диапазоны: `AGE` от 18 до 100, `LIMIT_BAL` больше нуля, суммы платежей `PAY_AMT1…6` не отрицательные;
- категории: `SEX` 1–2, `EDUCATION` 0–6, `MARRIAGE` 0–3, статусы `PAY_*` от −2 до 8, целевая переменная 0 или 1. Коды `EDUCATION` 0, 5, 6 и `MARRIAGE` 0 в описании набора отсутствуют, но приходят из источника, поэтому правила их допускают, а `prepare` перекодирует.

Результат каждой проверки записывается в `reports/validation.json`. Если хотя бы одна проверка не прошла, команда завершается с кодом 1 и печатает названия упавших правил. `prepare` зависит от этого отчёта, поэтому DVC запускает стадии по порядку и при провале валидации не доходит до очистки. Проверку можно запустить и без DVC: `poetry run python -m credit_default.data.validation`.


Стадия `prepare` (`src/credit_default/data/make_dataset.py`):

- переименовывает `PAY_0` в `PAY_1`, чтобы нумерация совпадала с `BILL_AMT1…6` и `PAY_AMT1…6`, а целевую колонку `default.payment.next.month` в `default`;
- удаляет `ID`;
- перекодирует коды, которых нет в описании набора данных: `EDUCATION` 0, 5, 6 в 4 («прочее»), `MARRIAGE` 0 в 3 («прочее»);
- удаляет 35 полных дубликатов до разбиения, чтобы одна строка не попала в обе выборки;
- делит данные со стратификацией по целевой переменной: 23 972 строки в train и 5 993 в test, доля дефолтов 0,2213 в обеих.

Параметры разбиения (`test_size`, `random_state`) заданы в `params.yaml`. DVC перезапускает стадию при изменении этих параметров, кода стадии или исходных данных.

Стадия `eda` (`src/credit_default/analysis/eda.py`) читает только `train.csv`, тестовая выборка в анализ не входит. Она сохраняет три графика в `reports/figures/`: баланс классов, распределения числовых признаков по классам и долю дефолтов по статусу последнего платежа `PAY_1`. Графики хранятся в git, а не в кэше DVC (`cache: false`), поэтому GitHub показывает их прямо в [reports/eda.md](reports/eda.md). После изменения данных или кода `dvc repro` строит графики заново.

Стадия `train` (`src/credit_default/models/train.py`) обучает модель на `train.csv`. Модель собрана как sklearn Pipeline (`src/credit_default/models/pipeline.py`):

1. Два доменных признака: `LIMIT_UTILIZATION` (доля использованного лимита в последнем месяце, `BILL_AMT1 / LIMIT_BAL`) и `MAX_DELAY` (максимальная просрочка за полгода по `PAY_1…6`). Обоснование признаков приведено в [reports/eda.md](reports/eda.md).
2. Предобработка: числовые колонки заполняются медианой и масштабируются, категориальные `SEX`, `EDUCATION`, `MARRIAGE` заполняются модой и кодируются one-hot. Неизвестная категория на инференсе кодируется нулями.
3. Классификатор.

Признаки и предобработка входят в Pipeline, поэтому при кросс-валидации медианы, моды и параметры масштабирования считаются только по обучающим фолдам, а на инференсе применяются те же преобразования.

GridSearchCV перебирает сетки из `params.yaml` на стратифицированной кросс-валидации из 5 фолдов по ROC-AUC. Метрика выбрана потому, что модель оценивает вероятность дефолта, а ROC-AUC оценивает ранжирование по вероятности и не зависит от порога. Precision, Recall и F1 считаются для класса «дефолт» при пороге 0,5. Лучшая модель выбирается по ROC-AUC на кросс-валидации, тестовая выборка в выборе не участвует.

Выходы стадии:

- `models/model.joblib`: лучший Pipeline, хранится в кэше DVC и на DagsHub, в git попадает только его хэш в `dvc.lock`;
- `reports/metrics.json`: модель, параметры, ROC-AUC на кросс-валидации и метрики на test;
- `reports/roc_curve.png`: ROC-кривая на test.

Метрики и график хранятся в git (`cache: false`), поэтому результат виден без запуска.

## Эксперименты MLflow

Каждый запуск `train` записывает в MLflow эксперимент `credit-default`:

- родительский run на каждую модель: лучшие параметры, ROC-AUC на кросс-валидации, метрики на test с префиксом `test_`, ROC-кривая и сама модель (`mlflow.sklearn.log_model` с примером входа и выведенной сигнатурой);
- вложенный run на каждого кандидата сетки: параметры, среднее и стандартное отклонение ROC-AUC по фолдам.

Итог последнего запуска (2 родительских и 7 вложенных runs):

| Модель | Лучшие параметры | ROC-AUC, CV | ROC-AUC, test | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Логистическая регрессия | `C=1.0` | 0,749 | 0,741 | 0,684 | 0,262 | 0,379 |
| Случайный лес, 200 деревьев | `max_depth=16`, `min_samples_leaf=20` | 0,787 | 0,777 | 0,671 | 0,354 | 0,464 |

Стадия сохраняет случайный лес: ROC-AUC на кросс-валидации у него выше на 0,04. На test он тоже лучше по ROC-AUC и находит больше дефолтов при пороге 0,5 (Recall 0,354 против 0,262). Внутри сеток разброс небольшой: у регрессии ROC-AUC 0,747–0,749 при `C` от 0,01 до 1, у леса 0,783–0,787. Разница между кандидатами одной модели меньше стандартного отклонения по фолдам (0,008–0,010), поэтому на этих сетках выбор модели дал больше, чем настройка её параметров.

Runs доступны без входа на сервере MLflow в DagsHub: [эксперимент credit-default](https://dagshub.com/artem_vk/MLOps_DevAndTest_SessionProject.mlflow/#/experiments/0). Вложенные runs раскрываются под родительским, модели собраны на вкладке Models.

Куда пишет MLflow, определяет переменная `MLFLOW_TRACKING_URI`. Без неё runs сохраняются локально в `mlflow.db` и `mlruns/` (оба в `.gitignore`) и открываются командой:

```bash
poetry run mlflow ui     # http://127.0.0.1:5000, переключатель Model training, эксперимент credit-default
```

Чтобы писать на свой сервер MLflow, нужно скопировать `.env.example` в `.env` и заполнить адрес, логин и токен. `train` читает `.env` при запуске, сам файл в git не попадает.

## API

API отдаёт прогноз модели из `models/model.joblib`. Файл появляется после `dvc pull` или `dvc repro`, другой путь задаёт переменная `MODEL_PATH`. Без файла модели сервис не запускается и сообщает, как его получить.

```bash
poetry run uvicorn credit_default.api.app:app --port 8000
```

| Метод | Путь | Ответ |
|---|---|---|
| GET | `/health` | `{"status": "ok"}`, когда сервис запущен и модель загружена |
| POST | `/predict` | класс (`1`, если вероятность дефолта не меньше 0,5) и вероятность дефолта |
| GET | `/docs` | интерактивная документация Swagger UI |

Запрос к `/predict` содержит 23 признака клиента с теми же именами, что в очищенных данных (`PAY_1…6`, а не `PAY_0`). Пример лежит в `examples/client.json`. В PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/client.json"
```

Ответ: `{"prediction":0,"probability":0.12505013040381596}`. Последние знаки вероятности могут отличаться между запусками: случайный лес считает деревья параллельно, и порядок сложения меняется.

Схема запроса (`src/credit_default/api/schemas.py`) отклоняет недопустимые значения ответом 422 с описанием ошибки:

- `LIMIT_BAL` больше нуля: признак `LIMIT_UTILIZATION` делит на него;
- `AGE` от 18 до 100, статусы `PAY_1…6` от −2 до 8, суммы платежей `PAY_AMT1…6` не отрицательные;
- `SEX` 1–2, `EDUCATION` 1–4, `MARRIAGE` 1–3. Коды `EDUCATION` 0, 5, 6 и `MARRIAGE` 0 из исходных данных стадия `prepare` перекодирует в «прочее» (4 и 3), поэтому API ждёт уже перекодированные значения;
- лишние поля, например `ID`, не принимаются.

## Docker

Образ содержит API, зависимости основной группы и модель, поэтому перед сборкой нужен файл `models/model.joblib` (`dvc pull` или `dvc repro`, см. «Данные»). Без него скрипт сборки останавливается и подсказывает, как получить модель.

PowerShell:

```powershell
.\scripts\docker_build.ps1
.\scripts\docker_run.ps1
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/client.json"
docker stop credit-default-api
```

bash:

```bash
./scripts/docker_build.sh
./scripts/docker_run.sh
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @examples/client.json
docker stop credit-default-api
```

`docker_run` запускает контейнер `credit-default-api` в фоне, ждёт до 30 секунд ответа `/health` и печатает адрес API. Если сервис не поднялся, скрипт выводит логи контейнера и удаляет его. Другой порт задаётся параметром: `.\scripts\docker_run.ps1 -Port 8080` или `./scripts/docker_run.sh credit-default-api 8080`. Если PowerShell не разрешает запуск скриптов, достаточно один раз выполнить `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Как устроен образ (`Dockerfile`):

- база `python:3.12-slim`, та же версия Python, что в проекте и CI;
- Poetry 2.4.1 ставится в отдельное окружение и устанавливает только основную группу из `poetry.lock`, поэтому версии пакетов совпадают с проверенными в CI, а DVC, MLflow, matplotlib и инструменты разработки в образ не попадают;
- сначала копируются `pyproject.toml` и `poetry.lock`, потом код: при изменении кода Docker берёт слой с зависимостями из кэша;
- кэш Poetry удаляется в том же слое, где ставятся пакеты;
- сервис работает от пользователя без прав root;
- `.dockerignore` пропускает в контекст сборки только нужные файлы, поэтому `.env` с токенами и данные в образ не попадают.

Модель копируется в образ при сборке, поэтому после переобучения образ нужно собрать заново.

## Тесты и проверки кода

```bash
poetry run pytest                 # unit-тесты с отчётом о покрытии
poetry run ruff format --check .  # форматирование
poetry run ruff check .           # линтинг
poetry run mypy                   # проверка типов (strict)
poetry run pre-commit install     # те же проверки перед каждым коммитом
```

CI запускает три задания на каждый push в `main` и на pull request: `Lint and type check` (ruff и mypy), `Unit tests` (pytest) и `Data validation`. Последнее скачивает исходный CSV из DagsHub через `dvc pull` и проверяет его правилами Great Expectations. DagsHub не отдаёт данные без входа, поэтому логин и токен хранятся в секретах репозитория `DAGSHUB_USER` и `DAGSHUB_TOKEN`. В форке секретов нет, и это задание там не пройдёт.

## Запуск на Windows

- Пути внутри кэша DVC длинные. Если репозиторий лежит в глубокой папке, `dvc repro` падает с `[WinError 3]` из-за ограничения Windows в 260 символов. Решение: включить длинные пути (PowerShell от администратора, затем новый терминал):

  ```powershell
  New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
  ```

- После `dvc repro` или `dvc pull` git может показать изменёнными `dvc.lock`, `.dvc`-файлы и `.gitignore` в папках с данными. DVC на Windows записывает их с CRLF, содержимое не меняется: `git diff --ignore-cr-at-eol` пуст.

# Модель вероятности дефолта (UCI Credit Card Default)

Сессионный проект по курсу «Автоматизация процессов разработки и тестирования моделей ML». Модель оценивает вероятность дефолта клиента по кредитной карте, а конвейер вокруг неё готовит данные, проверяет их, обучает модель и отдаёт прогноз через API.

Проект в разработке: README описывает только готовые части и дополняется вместе с проектом.

## Что готово

- Исходные данные версионируются DVC, хранилище размещено на DagsHub.
- DVC-стадия `validate` проверяет исходные данные набором правил Great Expectations и останавливает пайплайн при аномалиях.
- DVC-стадия `prepare` очищает данные и делит их на обучающую и тестовую выборки.
- DVC-стадия `eda` строит графики разведочного анализа, выводы собраны в [reports/eda.md](reports/eda.md).
- Unit-тесты очистки, разбиения и правил валидации (pytest).
- CI в GitHub Actions: форматирование, линтинг и проверка типов, unit-тесты, валидация реальных данных.

## Стек

| Инструмент | Зачем |
|---|---|
| Python 3.12, Poetry | окружение и зависимости; `poetry.lock` фиксирует точные версии всех пакетов, включая транзитивные |
| pandas, scikit-learn | обработка данных и модель |
| matplotlib | графики разведочного анализа |
| Great Expectations | правила валидации исходных данных |
| DVC, DagsHub | версии данных, воспроизводимый пайплайн, удалённое хранилище |
| pytest, pytest-cov | unit-тесты и покрытие |
| ruff | линтинг (правила flake8: pycodestyle и pyflakes, а также isort, pyupgrade, bugbear) и форматирование, совместимое с black |
| mypy | проверка типов в строгом режиме |
| pre-commit, gitleaks | проверки перед каждым коммитом, поиск секретов |
| GitHub Actions | CI |

## Структура

```
├── .github/workflows/ci.yml        # CI: линтинг, проверка типов, тесты
├── data/
│   ├── raw/                        # исходный CSV (в git только указатель .dvc)
│   └── processed/                  # train.csv и test.csv, выходы стадии prepare
├── reports/
│   ├── validation.json             # результат каждой проверки стадии validate
│   ├── eda.md                      # разведочный анализ: выводы и решения для модели
│   └── figures/                    # графики стадии eda
├── src/credit_default/
│   ├── config.py                   # пути проекта и чтение params.yaml
│   ├── data/validation.py          # стадия validate: правила Great Expectations
│   ├── data/make_dataset.py        # стадия prepare: загрузка, очистка, разбиение
│   └── analysis/eda.py             # стадия eda: графики по обучающей выборке
├── tests/                          # unit-тесты
├── dvc.yaml, dvc.lock              # описание пайплайна и зафиксированные хэши
├── params.yaml                     # параметры стадий
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

`poetry install` ставит все группы зависимостей: основную (pandas, scikit-learn), `train` (DVC, PyYAML, matplotlib, Great Expectations), `test` (pytest) и `lint` (ruff, mypy, pre-commit). Отдельные группы ставятся флагом `--only`, например `poetry install --only main,train,test`.

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

`dvc pull` скачивает исходный CSV и готовые `train.csv` и `test.csv`.

**Способ 2. Без аккаунта DagsHub.** Скачать `UCI_Credit_Card.csv` с Kaggle по ссылке выше, положить в `data/raw/` и собрать выборки заново:

```bash
poetry run dvc repro
```

Файл с Kaggle совпадает с версией в DVC по MD5, поэтому стадия `prepare` даёт те же `train.csv` и `test.csv` побайтно.

## Пайплайн

```bash
poetry run dvc repro     # пересчитать устаревшие стадии
poetry run dvc status    # показать, что изменилось с последнего запуска
poetry run dvc dag       # граф стадий: validate → prepare → eda
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

- После `dvc repro` или `dvc pull` git может показать `dvc.lock` и `.dvc`-файлы изменёнными. DVC на Windows записывает их с CRLF, содержимое не меняется: `git diff --ignore-cr-at-eol` пуст.

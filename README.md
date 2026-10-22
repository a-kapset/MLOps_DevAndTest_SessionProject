# Модель вероятности дефолта (UCI Credit Card Default)

Сессионный проект по курсу «Автоматизация процессов разработки и тестирования моделей ML». Модель оценивает вероятность дефолта клиента по кредитной карте, а конвейер вокруг неё готовит данные, проверяет их, обучает модель и отдаёт прогноз через API.

Проект в разработке: README описывает только готовые части и дополняется вместе с проектом.

## Что готово

- Исходные данные версионируются DVC, хранилище размещено на DagsHub.
- DVC-стадия `prepare` очищает данные и делит их на обучающую и тестовую выборки.
- Unit-тесты очистки и разбиения (pytest).
- CI в GitHub Actions: форматирование, линтинг и проверка типов, отдельно unit-тесты.

## Структура

```
├── .github/workflows/ci.yml        # CI: линтинг, проверка типов, тесты
├── data/
│   ├── raw/                        # исходный CSV (в git только указатель .dvc)
│   └── processed/                  # train.csv и test.csv, выходы стадии prepare
├── src/credit_default/
│   ├── config.py                   # пути проекта и чтение params.yaml
│   └── data/make_dataset.py        # стадия prepare: загрузка, очистка, разбиение
├── tests/                          # unit-тесты
├── dvc.yaml, dvc.lock              # описание пайплайна и зафиксированные хэши
├── params.yaml                     # параметры стадий
└── pyproject.toml, poetry.lock     # зависимости и настройки инструментов
```

## Установка

Нужны Python 3.12 и Poetry 2.x.

```bash
git clone https://github.com/a-kapset/MLOps_DevAndTest_SessionProject.git
cd MLOps_DevAndTest_SessionProject
poetry install
```

`poetry install` ставит все группы зависимостей: основную (pandas, scikit-learn), `train` (DVC, PyYAML), `test` (pytest) и `lint` (ruff, mypy, pre-commit). Отдельные группы ставятся флагом `--only`, например `poetry install --only main,train,test`.

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
```

Стадия `prepare` (`src/credit_default/data/make_dataset.py`):

- переименовывает `PAY_0` в `PAY_1`, чтобы нумерация совпадала с `BILL_AMT1…6` и `PAY_AMT1…6`, а целевую колонку `default.payment.next.month` в `default`;
- удаляет `ID`;
- перекодирует коды, которых нет в описании набора данных: `EDUCATION` 0, 5, 6 в 4 («прочее»), `MARRIAGE` 0 в 3 («прочее»);
- удаляет 35 полных дубликатов до разбиения, чтобы одна строка не попала в обе выборки;
- делит данные со стратификацией по целевой переменной: 23 972 строки в train и 5 993 в test, доля дефолтов 0,2213 в обеих.

Параметры разбиения (`test_size`, `random_state`) заданы в `params.yaml`. DVC перезапускает стадию при изменении этих параметров, кода стадии или исходных данных.

## Тесты и проверки кода

```bash
poetry run pytest                 # unit-тесты с отчётом о покрытии
poetry run ruff format --check .  # форматирование
poetry run ruff check .           # линтинг
poetry run mypy                   # проверка типов (strict)
poetry run pre-commit install     # те же проверки перед каждым коммитом
```

CI запускает два задания на каждый push в `main` и на pull request: `Lint and type check` (ruff и mypy) и `Unit tests` (pytest).

## Запуск на Windows

- Пути внутри кэша DVC длинные. Если репозиторий лежит в глубокой папке, `dvc repro` падает с `[WinError 3]` из-за ограничения Windows в 260 символов. Решение: включить длинные пути (PowerShell от администратора, затем новый терминал):

  ```powershell
  New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
  ```

- После `dvc repro` или `dvc pull` git может показать `dvc.lock` и `.dvc`-файлы изменёнными. DVC на Windows записывает их с CRLF, содержимое не меняется: `git diff --ignore-cr-at-eol` пуст.

## Технические решения

Несколько мест отличаются от формулировок задания.

- **Poetry вместо `requirements.txt`.** `poetry.lock` фиксирует точные версии всех пакетов, включая транзитивные, а группы зависимостей разделяют окружения: Docker-образ получит только основную группу, CI ставит только нужные заданию группы.
- **ruff вместо flake8 и black.** `ruff check` реализует правила flake8 (pycodestyle, pyflakes) и дополнительные наборы, `ruff format` совместим с black. Один инструмент работает быстрее и настраивается в одном месте, `pyproject.toml`. Проверку типов добавляет mypy в строгом режиме, её нет ни в flake8, ни в black.
- **Без папки `notebooks/`.** Анализ оформляется скриптами `.py`: их проверяют ruff, mypy и CI, а результаты воспроизводятся одной командой.
- **Пакет `src/credit_default/` вместо модулей прямо в `src/`.** Пакет устанавливается в окружение, поэтому импорты одинаково работают в тестах, стадиях DVC и будущем API без правки `sys.path`.

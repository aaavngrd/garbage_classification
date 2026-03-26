# Linting

## Обрані інструменти та причини вибору

| Інструмент | Версія | Призначення |
|---|---|---|
| **flake8** | 7.x | Стиль коду (PEP 8): пробіли, довжина рядків, невикористані імпорти |
| **pylint** | 3.x | Глибокий аналіз якості: docstrings, перевизначення змінних, складність |
| **bandit** | 1.9.x | Безпека: небезпечна десеріалізація, subprocess, hardcoded secrets |

**Чому саме ці інструменти:**
- `flake8` — найшвидший, зручно інтегрується в CI/CD, мінімум налаштувань
- `pylint` — дає числовий рейтинг 0–10, легко відстежувати прогрес між запусками
- `bandit` — специфічний для ML/API проєктів: `torch.load()` є реальним вектором атаки через pickle-десеріалізацію

---

## Встановлення

```bash
pip install flake8 pylint bandit
```

---

## Конфігураційні файли

### `.flake8`
```ini
[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude =
    .git,
    __pycache__,
    .venv,
    venv,
    data/,
    raw_dataset/
```

### `.pylintrc`
```ini
[MASTER]
ignore=data,raw_dataset,.venv,venv

[MESSAGES CONTROL]
disable=C0114,R0903

[FORMAT]
max-line-length=100

[BASIC]
good-names=i,j,k,p,x,n,lr,bs,e
```

---

## Запуск лінтерів

```bash
# Стиль та форматування
flake8 train.py server.py inference.py split_dataset.py > flake8_report_after.txt

# Якість коду (рейтинг 0–10)
pylint train.py server.py inference.py split_dataset.py > pylint_report_after.txt

# Безпека (тільки файли проєкту, без venv)
bandit train.py server.py inference.py split_dataset.py -f txt -o bandit_report_after.txt
```

---

## Правила та їх пояснення

| Правило | Інструмент | Пояснення |
|---|---|---|
| `max-line-length = 100` | flake8 | ML-код має довгі виклики — 79 символів замало |
| `E302` | flake8 | Між функціями обов'язково 2 порожні рядки — стандарт PEP 8 |
| `E261` | flake8 | Перед inline-коментарем `#` потрібно мінімум 2 пробіли |
| `F401` | flake8 | Невикористаний імпорт збільшує час завантаження та заплутує код |
| `C0116` | pylint | Відсутній docstring — без нього важко розуміти призначення функції |
| `R0402` | pylint | `import torch.nn as nn` → краще `from torch import nn` (явніше) |
| `W0621` | pylint | Перевизначення імені з зовнішньої області видимості приховує баги |
| `W0612` | pylint | Невикористана змінна — результат обчислюється, але ніде не використовується |
| `W0718` | pylint | `except Exception` — занадто широкий обробник винятків |
| `B614` | bandit | `torch.load()` використовує pickle — небезпечно для недовірених файлів |
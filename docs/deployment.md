# Інструкція з розгортання у виробничому середовищі

**Проєкт:** Garbage Classification API  
**Версія:** 1.0  
**Стек:** Python 3, FastAPI, PyTorch (MobileNetV2), Uvicorn  

---

## 1. Вимоги до апаратного забезпечення

| Ресурс | Мінімум | Рекомендовано |
|--------|---------|---------------|
| Архітектура | x86_64 | x86_64 |
| CPU | 2 ядра, 2.0 GHz | 4 ядра, 3.0+ GHz |
| RAM | 4 GB | 8 GB |
| Диск | 5 GB SSD | 20 GB SSD |
| ОС | Windows 10 (64-bit) | Windows 10/11 (64-bit) |

> **Примітка:** При першому запуску модель MobileNetV2 завантажує ваги (~14 MB) з інтернету. Переконайтесь, що сервер має доступ до мережі або ваги вже кешовані локально (`%USERPROFILE%\.cache\torch\hub\`).

---

## 2. Необхідне програмне забезпечення

### Python 3.11

1. Завантажити інсталятор: https://www.python.org/downloads/release/python-3110/
2. При встановленні обов'язково поставити галочку **"Add Python to PATH"**
3. Перевірити встановлення:

```cmd
python --version
```

### Git

Завантажити та встановити: https://git-scm.com/download/win

### Python-залежності

```cmd
pip install fastapi==0.111.0 uvicorn[standard]==0.30.1 torch==2.3.0 torchvision==0.18.0 Pillow==10.3.0
```

> **CPU-only сервер** (без GPU) — легша версія PyTorch:
> ```cmd
> pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 3. Налаштування мережі

API слухає порт `8000`. Необхідно відкрити його у Windows Firewall:

```cmd
netsh advfirewall firewall add rule ^
  name="Garbage Classification API" ^
  dir=in action=allow protocol=TCP localport=8000
```

Або через інтерфейс: **Панель керування → Windows Defender Firewall → Додаткові параметри → Правила для вхідних підключень → Створити правило → Порт → TCP 8000**.

---

## 4. Розгортання коду

### Структура директорій

```
C:\garbage_classification\
├── server.py
├── inference.py
├── train.py
├── split_dataset.py
├── best_model.pt      ← навчена модель (критичний файл!)
└── venv\              ← Python virtual environment
```

### Встановлення

```cmd
:: 1. Клонуємо репозиторій
git clone https://github.com/aaavngrd/garbage_classification.git C:\garbage_classification
cd C:\garbage_classification

:: 2. Створюємо віртуальне середовище
python -m venv venv
venv\Scripts\activate

:: 3. Встановлюємо залежності
pip install --upgrade pip
pip install fastapi uvicorn[standard] torch torchvision Pillow
```

### Копіювання моделі

Файл `best_model.pt` **не входить до репозиторію** — його потрібно скопіювати вручну з артефактів навчання:

```cmd
copy \\network-share\models\best_model.pt C:\garbage_classification\best_model.pt
```

### Запуск сервера

```cmd
cd C:\garbage_classification
venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Автозапуск через Task Scheduler

Щоб сервер стартував автоматично при завантаженні Windows:

1. Створити файл `C:\garbage_classification\start_api.bat`:

```bat
@echo off
cd C:\garbage_classification
call venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

2. Відкрити **Task Scheduler** (`taskschd.msc`) → **Create Basic Task**:
   - Trigger: **When the computer starts**
   - Action: **Start a program** → `C:\garbage_classification\start_api.bat`
   - Увімкнути **"Run whether user is logged on or not"**

### Оновлення коду

```cmd
cd C:\garbage_classification
git pull origin master
venv\Scripts\activate
pip install --upgrade fastapi uvicorn torch torchvision Pillow
:: Перезапустити сервер вручну або через Task Scheduler
```

### Оновлення моделі

```cmd
:: Резервна копія
copy C:\garbage_classification\best_model.pt C:\garbage_classification\best_model.pt.bak

:: Нова версія
copy \\network-share\models\new_model.pt C:\garbage_classification\best_model.pt

:: Перезапустити сервер
```

---

## 5. Перевірка працездатності

### Health-check

Відкрити у браузері або виконати в cmd:

```cmd
curl http://localhost:8000/
```

Очікувана відповідь:

```json
{
  "status": "API is running",
  "device": "cpu",
  "classes": ["glass", "metal", "organic", "paper", "plastic"]
}
```

### Тестовий запит на класифікацію

```cmd
curl -X POST http://localhost:8000/predict -F "file=@C:\test_image.jpg"
```

Очікувана відповідь:

```json
{
  "predicted_class": "plastic",
  "confidence": 0.9342
}
```

> `confidence` > 0.7 вважається надійним результатом.

### Чеклист після деплою

- [ ] Сервер запущений, у консолі видно `Uvicorn running on http://0.0.0.0:8000`
- [ ] `GET /` повертає статус `"API is running"` та список класів
- [ ] `POST /predict` з тестовим зображенням повертає `predicted_class` та `confidence`
- [ ] Порт 8000 відкритий у Windows Firewall
- [ ] Файл `best_model.pt` присутній у директорії проєкту

---

## Типові помилки та вирішення

| Помилка | Причина | Вирішення |
|---------|---------|-----------|
| `FileNotFoundError: best_model.pt` | Модель не скопійована | Скопіювати `best_model.pt` до `C:\garbage_classification\` |
| `address already in use :8000` | Порт зайнятий іншим процесом | `netstat -ano \| findstr :8000`, потім `taskkill /PID <id> /F` |
| `500 Internal Server Error` | Неправильний формат зображення | Переконатись, що файл є валідним JPEG або PNG |
| `CUDA not available` | Немає GPU або драйверів NVIDIA | Нормально для CPU-серверів; модель працює на CPU |
| `uvicorn: command not found` | Venv не активовано | Виконати `venv\Scripts\activate` перед запуском |

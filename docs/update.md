# Інструкція з оновлення проєкту

**Проєкт:** Garbage Classification API  
**ОС:** Windows 10/11 (64-bit)  
**Стек:** Python 3, FastAPI, PyTorch (MobileNetV2), Uvicorn  

---

## 1. Підготовка до оновлення

### 1.1 Створення резервних копій

Перед будь-яким оновленням збережіть поточний стан проєкту.

**Резервна копія моделі** — найкритичніший файл:

```cmd
copy C:\garbage_classification\best_model.pt ^
     C:\garbage_classification\best_model.pt.bak
```

**Резервна копія всієї директорії проєкту:**

```cmd
xcopy C:\garbage_classification C:\garbage_classification_backup_%date:~-4,4%%date:~-7,2%%date:~0,2% /E /I /H
```

Це створить папку виду `C:\garbage_classification_backup_20260327`.

---

### 1.2 Перевірка сумісності

Перед оновленням перевірте, чи нова версія коду або моделі сумісна з поточним середовищем.

**Перевірити поточні версії залежностей:**

```cmd
cd C:\garbage_classification
venv\Scripts\activate
pip freeze
```

**Перевірити Python:**

```cmd
python --version
```

Мінімально підтримувана версія: **Python 3.10+**

**Якщо оновлюється `best_model.pt`** — переконайтесь, що нова модель навчалась на тих самих класах і з тією самою архітектурою (MobileNetV2). Кількість класів у `checkpoint["class_to_idx"]` має збігатись з попередньою версією. Перевірити можна так:

```cmd
python -c "import torch; ck=torch.load('best_model.pt', map_location='cpu'); print(ck['class_to_idx'])"
```

---

### 1.3 Планування часу простою

Цей проєкт є stateless REST API без бази даних, тому оновлення коду або залежностей потребує лише короткого перезапуску процесу (~5–15 секунд).

| Тип оновлення | Час простою |
|---------------|-------------|
| Оновлення коду (`git pull`) | ~5–10 сек (перезапуск Uvicorn) |
| Оновлення залежностей (`pip install`) | ~1–3 хв |
| Заміна моделі (`best_model.pt`) | ~5–10 сек (перезапуск Uvicorn) |

Рекомендований час проведення оновлення — в години мінімального навантаження.

---

## 2. Процес оновлення

### 2.1 Зупинка сервера

Знайдіть PID процесу Uvicorn і завершіть його:

```cmd
netstat -ano | findstr :8000
```

У виводі знайдіть рядок зі станом `LISTENING` та відповідний PID. Потім:

```cmd
taskkill /PID <знайдений_PID> /F
```

Або, якщо сервер запущений у видимому вікні cmd — просто натисніть `Ctrl+C` у тому вікні.

---

### 2.2 Розгортання нового коду

```cmd
cd C:\garbage_classification

:: Отримати останні зміни
git fetch origin
git pull origin master
```

Якщо є конфлікти злиття — вирішіть їх вручну або скиньте до стану репозиторію:

```cmd
:: Увага: це відкине всі локальні зміни
git reset --hard origin/master
```

---

### 2.3 Оновлення конфігурацій

**Оновлення Python-залежностей** (якщо змінились):

```cmd
venv\Scripts\activate
pip install --upgrade fastapi uvicorn[standard] torch torchvision Pillow
```

**Заміна моделі** (якщо виходить нова версія `best_model.pt`):

```cmd
:: Резервна копія вже зроблена в п. 1.1, просто замінюємо файл
copy \\network-share\models\best_model_v2.pt C:\garbage_classification\best_model.pt
```

**Запуск сервера після оновлення:**

```cmd
cd C:\garbage_classification
venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 3. Перевірка після оновлення

```cmd
:: Health-check
curl http://localhost:8000/

:: Тестова класифікація
curl -X POST http://localhost:8000/predict -F "file=@C:\test_image.jpg"
```

Очікуваний результат `GET /`:

```json
{
  "status": "API is running",
  "device": "cpu",
  "classes": ["glass", "metal", "organic", "paper", "plastic"]
}
```

### Чеклист після оновлення

- [ ] Сервер запущений (`Uvicorn running on http://0.0.0.0:8000`)
- [ ] `GET /` повертає актуальний список класів
- [ ] `POST /predict` повертає коректний результат
- [ ] Версії залежностей (`pip freeze`) відповідають очікуваним

---

## 4. Відкат до попередньої версії

Якщо після оновлення виникли критичні помилки:

```cmd
:: Зупинити сервер
taskkill /PID <PID> /F

:: Відкотити код
cd C:\garbage_classification
git reset --hard HEAD~1

:: Відновити модель з резервної копії
copy C:\garbage_classification\best_model.pt.bak ^
     C:\garbage_classification\best_model.pt

:: Або відновити всю директорію з backup-папки
:: xcopy C:\garbage_classification_backup_20260327 C:\garbage_classification /E /I /H /Y

:: Запустити сервер
venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

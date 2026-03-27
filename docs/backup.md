# Резервне копіювання

> Описує стратегію та процедури резервного копіювання для сервісу `garbage_classification`.
> Система не має СУБД — резервуються три артефакти: **модель**, **залежності** та **версія коду**.

---

## Що резервується

| Артефакт | Файл / джерело | Критичність |
|---|---|---|
| Чекпоінт моделі | `best_model.pt` | Критична — без неї сервіс не запуститься |
| Список залежностей | `pip freeze` | Висока — потрібна для відтворення середовища |
| Версія коду | `git rev-parse HEAD` | Висока — потрібна для відкату коду |

> Датасет (`data\`) та `venv\` **не резервуються** — датасет зберігається окремо, venv відтворюється з pip freeze.

---

## Розміщення бекапів

```
C:\backups\garbage_classification\
├── best_model_20240315_143022.pt       # чекпоінт моделі
├── requirements_20240315_143022.txt    # залежності
├── commit_20240315_143022.txt          # git-коміт
├── best_model_20240320_090000.pt
├── requirements_20240320_090000.txt
└── commit_20240320_090000.txt
```

Директорію бекапів слід тримати на **окремому диску або мережевому сховищі** (NAS, S3, тощо).

---

## Ручне резервне копіювання

```cmd
C:\garbage_classification\docs\scripts\backup.bat
```

Або покроково у `cmd`:

```cmd
:: Отримати поточну дату/час у форматі YYYYMMDD_HHMMSS
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set BACKUP_DATE=%DT:~0,8%_%DT:~8,6%

set DEPLOY_DIR=C:\garbage_classification
set BACKUP_DIR=C:\backups\garbage_classification

if not exist %BACKUP_DIR% mkdir %BACKUP_DIR%

:: 1. Зберегти чекпоінт моделі
copy %DEPLOY_DIR%\best_model.pt %BACKUP_DIR%\best_model_%BACKUP_DATE%.pt

:: 2. Зберегти список залежностей
%DEPLOY_DIR%\venv\Scripts\pip freeze > %BACKUP_DIR%\requirements_%BACKUP_DATE%.txt

:: 3. Зберегти поточний git-коміт
git -C %DEPLOY_DIR% rev-parse HEAD > %BACKUP_DIR%\commit_%BACKUP_DATE%.txt

echo Бекап збережено: %BACKUP_DATE%
dir %BACKUP_DIR%
```

---

## Автоматичне резервне копіювання (Task Scheduler)

Для щоденного автоматичного бекапу о 02:00 виконайте у `cmd` від імені адміністратора:

```cmd
schtasks /create /tn "GarbageAPI Backup" ^
  /tr "C:\garbage_classification\docs\scripts\backup.bat" ^
  /sc DAILY /st 02:00 ^
  /ru SYSTEM /f
```

Лог виконання записується скриптом до `C:\backups\garbage_classification\backup.log`.

---

## Ротація бекапів

Зберігати всі бекапи нескінченно — нераціонально. Рекомендована стратегія:

- Зберігати **останні 7 щоденних** бекапів.
- Зберігати **бекап до кожного оновлення** окремо (позначати вручну або тегом).

Видалення застарілих бекапів (старші 7 днів) — вбудовано у `backup.bat` через `forfiles`:

```cmd
forfiles /p C:\backups\garbage_classification /m *.pt      /d -7 /c "cmd /c del @path"
forfiles /p C:\backups\garbage_classification /m requirements_*.txt /d -7 /c "cmd /c del @path"
forfiles /p C:\backups\garbage_classification /m commit_*.txt      /d -7 /c "cmd /c del @path"
```

---

## Перевірка цілісності бекапу

Після збереження чекпоінту переконайтесь, що він валідний:

```cmd
set MODEL_PATH=C:\backups\garbage_classification\best_model_<DATE>.pt
C:\garbage_classification\venv\Scripts\python ^
    C:\garbage_classification\docs\scripts\validate_model.py
```

Очікуваний вивід:

```
Чекпоінт валідний.
Класи: ['glass', 'metal', 'organic', 'paper', 'plastic']
```

---

## Відновлення з бекапу

Відновлення виконується в рамках процедури відкату — докладніше у [`update_guide.md`](update_guide.md#4-відкат-до-попередньої-версії).

Швидке відновлення лише моделі:

```cmd
set BACKUP_DATE=<YYYYMMDD_HHMMSS>

:: 1. Зупинити сервер
taskkill /IM python.exe /F

:: 2. Відновити модель
copy C:\backups\garbage_classification\best_model_%BACKUP_DATE%.pt ^
     C:\garbage_classification\best_model.pt

:: 3. Запустити сервер
cd C:\garbage_classification
venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8000

:: 4. Перевірити
curl http://localhost:8000/
```

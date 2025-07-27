# Импорт необходимых библиотек
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # Для валидации данных
from minio import Minio  # Для работы с MinIO (объектное хранилище)
from ultralytics import YOLO  # YOLO модель для детекции объектов
import cv2  # OpenCV для работы с изображениями
import numpy as np  # Для работы с массивами
from io import BytesIO  # Для работы с байтовыми потоками
from datetime import datetime  # Для работы с датой и временем
import psycopg2  # Для работы с PostgreSQL
import os  # Для работы с переменными окружения
from dotenv import load_dotenv  # Для загрузки переменных окружения из .env файла
from typing import List, Optional  # Для аннотации типов

# Загружаем переменные окружения из .env файла
load_dotenv()

# Создаем экземпляр FastAPI приложения
app = FastAPI()

# Настраиваем CORS (Cross-Origin Resource Sharing) middleware
# Это позволяет фронтенду на localhost:5173 делать запросы к нашему API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Разрешенные источники
    allow_credentials=True,  # Разрешить куки и заголовки авторизации
    allow_methods=["*"],  # Разрешить все HTTP методы
    allow_headers=["*"],  # Разрешить все заголовки
)

# Инициализируем клиент MinIO для работы с объектным хранилищем
minio_client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT"),  # Адрес MinIO сервера
    access_key=os.getenv("MINIO_ACCESS_KEY"),  # Ключ доступа
    secret_key=os.getenv("MINIO_SECRET_KEY"),  # Секретный ключ
    secure=False  # Использовать HTTPS (False для локальной разработки)
)

# Имя бакета в MinIO, где будут храниться изображения
bucket_name = "detections"

# Загружаем предобученную YOLO модель для детекции объектов
model = YOLO("best100.pt")

# Цвета для отрисовки bounding box'ов разных классов
class_colors = {
    0: (0, 0, 255),    # Красный
    1: (0, 255, 0),    # Зеленый
    2: (255, 0, 0),    # Синий
    3: (0, 255, 255),  # Желтый
    4: (255, 0, 255),  # Пурпурный
    5: (255, 255, 0),  # Голубой
    6: (0, 165, 255)   # Оранжевый
}

# Словарь для переименования классов с английского на русский
class_rename_map = {
    "bulk carrier": "балкер",
    "container ship": "контейнеровоз",
    "sailboat": "парусное судно",
    "fishing boat": "рыбацкая лодка",
    "liner": "лайнер",
    "warship": "военный корабль",
    "canoe": "яхта",
}

# Модель запроса для детекции изображения
class DetectRequest(BaseModel):
    image_key: str  # Ключ изображения в MinIO
    classes: Optional[List[str]] = None  # Опциональный список классов для фильтрации

# Эндпоинт для загрузки изображения в MinIO
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Читаем содержимое файла
        contents = await file.read()
        
        # Генерируем уникальный ключ для изображения с timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_key = f"uploads/{timestamp}_{file.filename}"

        # Загружаем изображение в MinIO
        minio_client.put_object(
            bucket_name,
            image_key,
            BytesIO(contents),
            length=len(contents),
            content_type=file.content_type
        )

        return {
            "message": "Файл успешно загружен",
            "image_key": image_key
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {str(e)}")

# Эндпоинт для получения списка детекций с возможностью фильтрации
@app.get("/detections")
def get_detections(
    detection_id: Optional[int] = Query(None),  # Фильтр по ID детекции
    start_date: str = Query(None),  # Начальная дата для фильтрации
    end_date: str = Query(None),    # Конечная дата для фильтрации
    min_objects: int = Query(None), # Минимальное количество объектов
    max_objects: int = Query(None), # Максимальное количество объектов
    # Далее идут фильтры по количеству объектов каждого класса
    min_container_ship: int = Query(None),
    max_container_ship: int = Query(None),
    min_liner: int = Query(None),
    max_liner: int = Query(None),
    min_warship: int = Query(None),
    max_warship: int = Query(None),
    min_fishing_boat: int = Query(None),
    max_fishing_boat: int = Query(None),
    min_bulk_carrier: int = Query(None),
    max_bulk_carrier: int = Query(None),
    min_sailboat: int = Query(None),
    max_sailboat: int = Query(None),
    min_canoe: int = Query(None),
    max_canoe: int = Query(None)
):
    try:
        # Подключаемся к PostgreSQL
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        cursor = conn.cursor()

        # Базовый SQL запрос для получения детекций
        sql = """
            SELECT 
                id, 
                detection_time, 
                object_count, 
                object_classes, 
                input_image_key, 
                output_image_key, 
                settings_classes
            FROM detections
        """
        filters = []  # Список для условий WHERE
        params = []   # Список параметров для SQL запроса

        # Добавляем условия фильтрации в зависимости от переданных параметров
        if detection_id is not None:
            filters.append("id = %s")
            params.append(detection_id)

        if start_date:
            filters.append("detection_time >= %s")
            params.append(start_date)
        if end_date:
            filters.append("detection_time <= %s")
            params.append(end_date)

        if min_objects is not None:
            filters.append("object_count >= %s")
            params.append(min_objects)
        if max_objects is not None:
            filters.append("object_count <= %s")
            params.append(max_objects)

        # Вспомогательная функция для добавления фильтров по классам
        def add_class_filter(class_name, min_count, max_count):
            if min_count is not None or max_count is not None:
                filter_parts = []
                if min_count is not None:
                    filter_parts.append("coalesce(array_length(array_positions(object_classes, %s), 1), 0) >= %s")
                    params.extend([class_name, min_count])
                if max_count is not None:
                    filter_parts.append("coalesce(array_length(array_positions(object_classes, %s), 1), 0) <= %s")
                    params.extend([class_name, max_count])
                filters.append(f"({' AND '.join(filter_parts)})")

        # Добавляем фильтры для каждого класса
        add_class_filter("контейнеровоз", min_container_ship, max_container_ship)
        add_class_filter("лайнер", min_liner, max_liner)
        add_class_filter("военный корабль", min_warship, max_warship)
        add_class_filter("рыбацкая лодка", min_fishing_boat, max_fishing_boat)
        add_class_filter("балкер", min_bulk_carrier, max_bulk_carrier)
        add_class_filter("парусное судно", min_sailboat, max_sailboat)
        add_class_filter("яхта", min_canoe, max_canoe)

        # Если есть фильтры, добавляем их в SQL запрос
        if filters:
            sql += " WHERE " + " AND ".join(filters)

        # Сортируем по времени детекции (новые сначала)
        sql += " ORDER BY detection_time DESC"

        # Выполняем запрос
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        # Формируем список детекций для ответа
        detections = []
        for row in rows:
            detections.append({
                "id": row[0],
                "detection_time": row[1].isoformat(),
                "object_count": row[2],
                "object_classes": row[3],
                "input_image_key": row[4],
                "output_image_key": row[5],
                "settings_classes": row[6]
            })

        # Закрываем соединение с БД
        cursor.close()
        conn.close()

        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинт для обработки изображения и детекции объектов
@app.post("/detect-image")
async def detect_image(req: DetectRequest):
    try:
        # Получаем изображение из MinIO по ключу
        response = minio_client.get_object(bucket_name, req.image_key)
        image_bytes = response.read()
        response.close()
        response.release_conn()

        # Декодируем изображение из байтов в numpy массив
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Ошибка декодирования изображения")

        # Выполняем детекцию объектов с помощью YOLO модели
        results = model(image)[0]

        filtered_boxes = []  # Для отфильтрованных bounding box'ов
        filtered_classes_detected = []  # Для отфильтрованных классов

        # Если указаны классы для фильтрации, создаем множество
        requested_classes = set(req.classes) if req.classes else None

        # Фильтруем результаты детекции по классам (если указаны)
        for box in results.boxes:
            cls_id = int(box.cls[0])
            orig_class = model.names[cls_id]
            class_name = class_rename_map.get(orig_class, orig_class)

            # Пропускаем классы, которые не входят в фильтр
            if requested_classes is not None and class_name not in requested_classes:
                continue

            filtered_boxes.append(box)
            filtered_classes_detected.append(class_name)

        # Рисуем bounding box'ы на изображении
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            color = class_colors.get(cls_id, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Кодируем обработанное изображение в JPEG
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise HTTPException(status_code=500, detail="Ошибка кодирования результата")

        # Генерируем ключ для сохранения обработанного изображения
        processed_key = f"processed/{req.image_key}"

        # Сохраняем обработанное изображение в MinIO
        minio_client.put_object(
            bucket_name,
            processed_key,
            BytesIO(encoded_image.tobytes()),
            length=len(encoded_image),
            content_type="image/jpeg"
        )

        # Сохраняем информацию о детекции в PostgreSQL
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detections (
                detection_time, object_count, object_classes, input_image_key, output_image_key, settings_classes
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            datetime.now(),
            len(filtered_boxes),
            filtered_classes_detected,
            req.image_key,
            processed_key,
            req.classes or []
        ))
        detection_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()

        return {
            "detection_id": detection_id,
            "object_count": len(filtered_boxes),
            "object_classes": filtered_classes_detected,
            "processed_image_key": processed_key
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск приложения с помощью uvicorn (если файл запущен напрямую)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:main", host="0.0.0.0", port=8000, reload=True)
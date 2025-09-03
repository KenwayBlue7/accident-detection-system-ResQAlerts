import firebase_admin
from firebase_admin import credentials, messaging, storage, firestore
import uuid
import os
import time
from datetime import datetime

# Initialize Firebase
service_account_path = r"D:/College/Mini Project/accident_detection/config/firebase_service_account.json"
cred = credentials.Certificate(service_account_path)

if not firebase_admin._apps:
    print(f"[{datetime.now()}] 🔑 Initializing Firebase with service account: {service_account_path}")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'divine-beanbag-454003-a1.firebasestorage.app'
    })
    print(f"[{datetime.now()}] ✅ Firebase initialized successfully.")

db = firestore.client()
bucket = storage.bucket()
print(f"[{datetime.now()}] 📦 Storage bucket fetched: {bucket.name}")

# Logging Setup
last_upload_time = 0
THROTTLE_SECONDS = 10
LOG_FILE = "output/logs/firebase_upload_log.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_message(message: str):
    timestamped = f"[{datetime.now()}] {message}"
    print(timestamped)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(timestamped + "\n")

def upload_accident_data(image_path, coordinates, accident_type="accident"):
    global last_upload_time
    current_time = time.time()

    log_message(f"🕒 Checking upload throttle. Current Time: {current_time}, Last Upload Time: {last_upload_time}")
    if current_time - last_upload_time < THROTTLE_SECONDS:
        log_message("⏱ Upload skipped (within 10s throttle window)")
        return None, None

    last_upload_time = current_time
    log_message(f"📤 Starting image upload process → Image Path: {image_path}")

    image_filename = os.path.basename(image_path)
    unique_id = str(uuid.uuid4())
    blob_path = f"accident_images/{unique_id}_{image_filename}"
    blob = bucket.blob(blob_path)

    log_message(f"🗂 Generated Blob Path: {blob_path}")
    log_message(f"🆔 Generated Accident ID: {unique_id}")

    try:
        blob.upload_from_filename(image_path)
        log_message(f"📁 Image upload to Firebase Storage successful → Blob Path: {blob_path}")
        
        blob.make_public()
        image_url = blob.public_url
        log_message(f"🌐 Image made public → URL: {image_url}")

        data = {
            "accident_id": unique_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "local_time": datetime.now().isoformat(),
            "location": {
                "latitude": coordinates[0],
                "longitude": coordinates[1]
            },
            "image_url": image_url,
            "maps_link": f"https://maps.google.com/?q={coordinates[0]},{coordinates[1]}",
            "accident_type": accident_type,
            "status": "unresolved",
            "responder_id": None
        }

        log_message(f"📝 Firestore Data Payload: {data}")

        db.collection("accidents-info").document(unique_id).set(data)
        log_message(f"✅ Firestore document created successfully → Document ID: {unique_id}")

        return image_url, data["maps_link"]

    except Exception as e:
        log_message(f"❌ Upload failed due to exception: {e}")
        return None, None

def send_accident_notification(title, body, image_url=None, maps_link=None, topic="ambulance_alert"):
    try:
        full_body = f"{body}\nLocation: {maps_link}" if maps_link else body
        log_message(f"📨 Preparing notification → Title: {title}, Body: {full_body}, Topic: {topic}")

        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=full_body,
                image=image_url
            ),
            topic=topic
        )

        response = messaging.send(message)
        log_message(f"✅ FCM notification sent successfully → Response ID: {response}")

    except Exception as e:
        log_message(f"❌ Notification sending failed due to exception: {e}")

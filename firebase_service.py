import firebase_admin
from firebase_admin import credentials, messaging, storage, firestore
import uuid
import os
import time
from datetime import datetime

def initialize_firebase():
    """Initialize Firebase with service account credentials from environment or config file."""
    if firebase_admin._apps:
        print(f"[{datetime.now()}] ✅ Firebase already initialized.")
        return
    
    # Try to get service account path from environment variable first
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
    
    # Fallback to default config path (relative to project root)
    if not service_account_path:
        service_account_path = os.path.join(os.path.dirname(__file__), "config", "firebase_service_account.json")
    
    # Check if service account file exists
    if not os.path.exists(service_account_path):
        raise FileNotFoundError(
            f"Firebase service account file not found at: {service_account_path}\n"
            f"Please ensure the file exists or set FIREBASE_SERVICE_ACCOUNT_PATH environment variable."
        )
    
    try:
        print(f"[{datetime.now()}] 🔑 Initializing Firebase with service account")
        cred = credentials.Certificate(service_account_path)
        
        # Get storage bucket from environment variable or use default
        storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET', 'your-project-id.firebasestorage.app')
        
        firebase_admin.initialize_app(cred, {
            'storageBucket': storage_bucket
        })
        print(f"[{datetime.now()}] ✅ Firebase initialized successfully.")
        
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Firebase initialization failed: {e}")
        raise

# Initialize Firebase on import
initialize_firebase()

# Initialize Firestore and Storage clients
try:
    db = firestore.client()
    bucket = storage.bucket()
    print(f"[{datetime.now()}] 📦 Storage bucket ready: {bucket.name}")
except Exception as e:
    print(f"[{datetime.now()}] ❌ Failed to initialize Firebase clients: {e}")
    db = None
    bucket = None

# Configuration
THROTTLE_SECONDS = int(os.getenv('UPLOAD_THROTTLE_SECONDS', '10'))
LOG_FILE = os.path.join("output", "logs", "firebase_upload_log.txt")

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Global variables
last_upload_time = 0

def log_message(message: str):
    """Log messages to both console and file with timestamp."""
    timestamped = f"[{datetime.now()}] {message}"
    print(timestamped)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(timestamped + "\n")
    except Exception as e:
        print(f"[{datetime.now()}] ⚠️ Failed to write to log file: {e}")

def check_firebase_connection():
    """Check if Firebase services are properly initialized."""
    if not db or not bucket:
        log_message("❌ Firebase services not properly initialized")
        return False
    return True

def upload_accident_data(image_path, coordinates, accident_type="accident"):
    """
    Upload accident data to Firebase Storage and Firestore.
    
    Args:
        image_path (str): Path to the accident image
        coordinates (tuple): (latitude, longitude) coordinates
        accident_type (str): Type of accident detected
        
    Returns:
        tuple: (image_url, maps_link) or (None, None) if failed
    """
    global last_upload_time
    
    if not check_firebase_connection():
        return None, None
        
    current_time = time.time()

    log_message(f"🕒 Checking upload throttle. Time since last upload: {current_time - last_upload_time:.1f}s")
    if current_time - last_upload_time < THROTTLE_SECONDS:
        log_message(f"⏱ Upload skipped (within {THROTTLE_SECONDS}s throttle window)")
        return None, None

    last_upload_time = current_time
    log_message(f"📤 Starting image upload process → Image Path: {image_path}")

    # Validate input parameters
    if not os.path.exists(image_path):
        log_message(f"❌ Image file not found: {image_path}")
        return None, None
        
    if not coordinates or len(coordinates) != 2:
        log_message(f"❌ Invalid coordinates: {coordinates}")
        return None, None

    try:
        image_filename = os.path.basename(image_path)
        unique_id = str(uuid.uuid4())
        blob_path = f"accident_images/{unique_id}_{image_filename}"
        blob = bucket.blob(blob_path)

        log_message(f"🗂 Generated Blob Path: {blob_path}")
        log_message(f"🆔 Generated Accident ID: {unique_id}")

        # Upload image to Firebase Storage
        blob.upload_from_filename(image_path)
        log_message(f"📁 Image upload to Firebase Storage successful → Blob Path: {blob_path}")
        
        # Make image publicly accessible
        blob.make_public()
        image_url = blob.public_url
        log_message(f"🌐 Image made public → URL: {image_url}")

        # Prepare Firestore document data
        data = {
            "accident_id": unique_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "local_time": datetime.now().isoformat(),
            "location": {
                "latitude": float(coordinates[0]),
                "longitude": float(coordinates[1])
            },
            "image_url": image_url,
            "maps_link": f"https://maps.google.com/?q={coordinates[0]},{coordinates[1]}",
            "accident_type": accident_type,
            "status": "unresolved",
            "responder_id": None,
            "created_by": "accident_detection_system"
        }

        log_message(f"📝 Firestore Data Payload: {data}")

        # Save to Firestore
        db.collection("accidents-info").document(unique_id).set(data)
        log_message(f"✅ Firestore document created successfully → Document ID: {unique_id}")

        return image_url, data["maps_link"]

    except Exception as e:
        log_message(f"❌ Upload failed due to exception: {str(e)}")
        return None, None

def send_accident_notification(title, body, image_url=None, maps_link=None, topic=None):
    """
    Send accident notification via Firebase Cloud Messaging.
    
    Args:
        title (str): Notification title
        body (str): Notification body
        image_url (str, optional): URL of accident image
        maps_link (str, optional): Google Maps link to location
        topic (str, optional): FCM topic to send to
        
    Returns:
        bool: True if notification sent successfully, False otherwise
    """
    if not check_firebase_connection():
        return False
        
    # Get topic from environment variable or use default
    if not topic:
        topic = os.getenv('FCM_TOPIC', 'ambulance_alert')
    
    try:
        # Construct full notification body
        full_body = f"{body}\nLocation: {maps_link}" if maps_link else body
        log_message(f"📨 Preparing notification → Title: {title}, Body: {full_body}, Topic: {topic}")

        # Create FCM message
        message_data = {
            "notification": messaging.Notification(
                title=title,
                body=full_body
            ),
            "topic": topic
        }
        
        # Add image if provided
        if image_url:
            message_data["notification"].image = image_url

        message = messaging.Message(**message_data)

        # Send notification
        response = messaging.send(message)
        log_message(f"✅ FCM notification sent successfully → Response ID: {response}")
        return True

    except Exception as e:
        log_message(f"❌ Notification sending failed due to exception: {str(e)}")
        return False

def get_firebase_config():
    """Get current Firebase configuration for debugging."""
    if not check_firebase_connection():
        return None
        
    return {
        "storage_bucket": bucket.name if bucket else None,
        "firestore_project": db._client.project if db else None,
        "throttle_seconds": THROTTLE_SECONDS,
        "log_file": LOG_FILE
    }

# Health check function
def test_firebase_connection():
    """Test Firebase connection and log status."""
    try:
        if check_firebase_connection():
            # Test Firestore
            test_doc = db.collection("system_health").document("connection_test")
            test_doc.set({"test": True, "timestamp": datetime.now().isoformat()})
            
            log_message("✅ Firebase connection test successful")
            return True
        else:
            log_message("❌ Firebase connection test failed")
            return False
            
    except Exception as e:
        log_message(f"❌ Firebase connection test failed: {str(e)}")
        return False

# Optional: Test connection on import (only in development)
if os.getenv('ENVIRONMENT') == 'development':
    test_firebase_connection()

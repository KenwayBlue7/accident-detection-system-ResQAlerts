from firebase_admin import credentials, initialize_app, storage
import os

cred = credentials.Certificate(r"D:/College/Mini Project/accident_detection/config/firebase_service_account.json")
initialize_app(cred, {
    'storageBucket': 'divine-beanbag-454003-a1.appspot.com'
})

bucket = storage.bucket()
blob = bucket.blob('test_upload/testfile.jpg')

try:
    blob.upload_from_filename('test_image.jpg')  # 👈 place a sample file in the same folder
    blob.make_public()
    print("✅ Upload Success:", blob.public_url)
except Exception as e:
    print("❌ Upload Failed:", e)

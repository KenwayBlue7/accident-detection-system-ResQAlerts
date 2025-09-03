# 📦 YOLO Model Files - Download Required

## 🚫 Model Files Not Included

Due to GitHub's 100MB file size limit, the YOLO model files required for this accident detection system are hosted externally.

## 🔗 Download Link

**📁 Google Drive**: [**Download Model Files**](https://drive.google.com/drive/folders/16gM9xko2g0lbqjx2yg1MDF3Q-vojON2q?usp=sharing)

## 📋 Required Files

Please download the following files and place them in this `models/` folder:

| File Name | Size | Description |
|-----------|------|-------------|
| `custom2_1280.onnx` | 139.59 MB | Custom trained YOLO model for accident detection |
| `yolov7_5classes_640x640.onnx` | 139.36 MB | YOLOv7 model trained on 5 traffic classes |
| `working.onnx` | 139.59 MB | Additional working model file |

## 📁 Final Folder Structure

After downloading, your `models/` folder should look like this:

```
models/
├── README.md (this file)
├── custom2_1280.onnx
├── yolov7_5classes_640x640.onnx
└── working.onnx
```

## ⚠️ Important Notes

- **⚡ The system will NOT work without these model files**
- 🔧 All files are in ONNX format for optimal performance
- 📊 Total download size: **~400MB**
- 🌐 Ensure stable internet connection for download

## 🚀 Quick Setup

1. **📎 Click the Google Drive link above**
2. **⬇️ Download all `.onnx` files**
3. **📂 Place them in this `models/` directory**
4. **▶️ Run `python main.py` to start the system**

## 🔧 Troubleshooting

### **🔒 File Access Issues?**
- Ensure you have access to the Google Drive folder
- Check your internet connection
- Try downloading files individually if batch download fails

### **❌ Model Loading Errors?**
- Verify all three `.onnx` files are present
- Check file sizes match the expected values
- Ensure files are not corrupted during download

### **⚙️ Performance Issues?**
- Use GPU acceleration with `onnxruntime-gpu` for faster inference
- Ensure sufficient RAM (minimum 4GB recommended)

## 📊 Model Information

### **Detection Classes**
The models are trained to detect:
- 🚗 Cars
- 🚌 Buses  
- 🏍️ Motorcycles
- 🚴 Cyclists
- 🚶 Pedestrians

### **Model Specifications**
- **Framework**: YOLOv7
- **Format**: ONNX Runtime compatible
- **Input Resolution**: 640x640 / 1280x1280
- **Precision**: FP32

## 📞 Support

If you encounter issues accessing the model files:

1. 🔍 **Check the project's [GitHub Issues](../../issues)**
2. 📧 **Contact the repository maintainer**
3. 🔗 **Verify the Google Drive link is still active**
4. 💬 **Join the [GitHub Discussions](../../discussions)**

## 📝 License

These model files are provided for research and educational purposes. Please refer to the main repository license for usage terms.

---

**📅 Last Updated**: September 2025  
**🔖 File Version**: 1.0  
**👨‍💻 Maintained by**: Accident Detection System Team

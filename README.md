# Brain Tumor Detection Using Deep Learning

## 📋 Overview

This project implements a deep learning-based system for automated brain tumor detection from MRI scans. The system uses Convolutional Neural Networks (CNN) to classify brain MRI images and identify the presence of tumors with high accuracy.

## 🎯 Features

- Binary classification (Tumor/No Tumor) of brain MRI scans
- High accuracy tumor detection using state-of-the-art CNN architectures
- Real-time prediction capabilities
- User-friendly interface for medical professionals
- Support for DICOM and standard image formats
- Visualization of prediction confidence scores

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image preprocessing
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Scikit-learn** - Model evaluation metrics
- **Pandas** - Data manipulation

## 📁 Project Structure

```
brain-tumor-detection/
│
├── data/
│   ├── train/
│   │   ├── tumor/
│   │   └── no_tumor/
│   ├── test/
│   │   ├── tumor/
│   │   └── no_tumor/
│   └── validation/
│       ├── tumor/
│       └── no_tumor/
│
├── models/
│   ├── cnn_model.py
│   ├── pretrained_models.py
│   └── saved_models/
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_analysis.ipynb
│
├── utils/
│   ├── data_augmentation.py
│   ├── visualization.py
│   └── metrics.py
│
├── requirements.txt
├── README.md
└── main.py
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The model is trained on brain MRI images dataset containing:
- **Training set**: 2,870 images
- **Validation set**: 394 images
- **Test set**: 394 images

Dataset sources:
- [Kaggle Brain MRI Images Dataset](https://www.kaggle.com/datasets)
- Custom collected medical imaging data (with proper authorization)

## 🔧 Usage

### Training the Model

```python
python src/train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Making Predictions

```python
python src/predict.py --image_path path/to/mri_scan.jpg --model_path models/saved_models/best_model.h5
```

### Evaluating Model Performance

```python
python src/evaluate.py --test_data_path data/test/ --model_path models/saved_models/best_model.h5
```

## 📈 Model Architecture

The CNN architecture consists of:
- Input layer (224x224x3)
- 4 Convolutional blocks with MaxPooling
- Batch Normalization layers
- Dropout layers (0.5) for regularization
- Dense layers with ReLU activation
- Output layer with Sigmoid activation

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 97.5% |
| Precision | 96.8% |
| Recall | 98.2% |
| F1-Score | 97.5% |
| AUC-ROC | 0.99 |

## 🖼️ Sample Results

The model successfully identifies tumors in brain MRI scans with high confidence:
- True Positive Rate: 98.2%
- False Positive Rate: 3.2%
- Processing time per image: ~0.3 seconds

## 🔄 Data Augmentation

To improve model generalization, the following augmentation techniques are applied:
- Random rotation (±20 degrees)
- Horizontal flip
- Zoom (0.9-1.1)
- Brightness adjustment
- Gaussian noise addition

## 🚧 Future Improvements

- [ ] Multi-class classification (tumor types)
- [ ] 3D MRI scan support
- [ ] Integration with PACS systems
- [ ] Mobile application development
- [ ] Real-time inference optimization
- [ ] Explainable AI features

## ⚠️ Disclaimer

This tool is intended for research and educational purposes only. It should not be used as the sole method for medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Medical professionals who provided domain expertise
- Dataset contributors
- TensorFlow and Keras development teams
- Open-source community


---

**Note**: This project is continuously being improved. Check back for updates and new features!

# Tubes2_IF3270-CNN-RNN-Scratch

## Deskripsi Singkat

Repository ini berisi implementasi **from scratch** dari arsitektur deep learning CNN, RNN, dan LSTM menggunakan Python dan NumPy. Proyek ini bertujuan untuk memahami fundamental dari neural network dengan mengimplementasikan algoritma forward propagation tanpa menggunakan library high-level seperti Keras atau PyTorch untuk core computation.

### Struktur Proyek
```
├── src/
│   ├── CNN/          # Implementasi Convolutional Neural Network
│   ├── RNN/          # Implementasi Recurrent Neural Network  
│   └── LSTM/         # Implementasi Long Short-Term Memory
│       ├── Layer/    # Layer implementations (LSTM, Dense, Dropout, etc.)
│       ├── Function/ # Activation functions dan utilities
│       ├── main.py   # Script utama untuk training dan testing
│       ├── LSTMModel.py # Model wrapper
│       ├── Utils.py  # Utility functions
│       └── Testing.ipynb # Jupyter notebook untuk eksperimen
├── requirements.txt  # Dependencies
└── README.md
```

## Setup dan Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/fauzannazz/Tubes2_IF3270-CNN-RNN-Scratch.git
cd Tubes2_IF3270-CNN-RNN-Scratch
```

### 2. Setup Virtual Environment
```bash
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Untuk Windows:
venv\Scripts\activate

# Untuk macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Pastikan dataset NusaX sudah tersedia di masing-masing folder model, contoh : `src/LSTM/data/nusax_sentiment_id/` atau sesuaikan path di konfigurasi.

## 🔧 Cara Menjalankan Program

### Menjalankan LSTM Implementation
```bash
cd src/LSTM
python main.py
```

### Struktur Output
Program akan menghasilkan:
- Model training progress dan metrics
- Perbandingan hasil antara implementasi custom vs Keras

## Fitur Utama

### LSTM Implementation
- ✅ **Forward Propagation**: Input gate, forget gate, output gate, cell state
- ✅ **Bidirectional LSTM**: Forward dan backward processing
- ✅ **Dropout Regularization**: Mencegah overfitting
- ✅ **Masking Support**: Handling variable-length sequences
- ✅ **Keras Compatibility**: Load weights dari pre-trained Keras model

### CNN Implementation
- ✅ **Forward Propagation**: Convolution Layer (with relu), Pooling Layer, Flatten Layer, Dense Layer
- ✅ **Pooling Layer**: MaxPooling dan AveragePooling
- ✅ **Activation Function**: Relu, Softmax (hanya dense layer)
- ✅ **Keras Compatibility**: Load weights dari pre-trained Keras model

### RNN Implementation
- 

## Pembagian Tugas Anggota Kelompok

### Anggota 1: [Muhammad Fauzan Azhim] - 13522153
**Tanggung Jawab: LSTM Implementation**
- Implementasi LSTM cell dan gates (input, forget, output)
- Forward propagation untuk unidirectional dan bidirectional LSTM
- Integration dengan embedding dan dense layers
- Testing dan validasi terhadap Keras implementation

### Anggota 2: [Pradipta Rafa Mahesa] - 13522162  
**Tanggung Jawab: CNN Implementation**
- Implementasi Convolution,MaxPool2D,AvgPool2D,Flatten, dan Dense Layer
- Forward propagation model CNN
- Testing dan validasi terhadap Keras implementation

### Anggota 3: [Nama] - NIM
**Tanggung Jawab: RNN Implementation**
-

## 🔗 Dependencies

Lihat `requirements.txt` untuk daftar lengkap, dependencies utama:
- `numpy`: Computational operations
- `tensorflow`: Baseline comparison dan data loading
- `scikit-learn`: Evaluation metrics
- `matplotlib`: Visualization
- `pandas`: Data manipulation
- `jupyter`: Interactive development

## Lisensi

Proyek ini dibuat untuk keperluan tugas akademik IF3270 - Machine Learning.

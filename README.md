# Anti-Spoofing Model Training

Repository này được thiết kế để huấn luyện và đánh giá các mô hình học sâu phục vụ cho tác vụ Face Anti-Spoofing (Chống giả mạo khuôn mặt).

## Cấu trúc thư mục

Thư mục dự án được tổ chức như sau:

```text
├── data/                  # Thư mục chứa dữ liệu
│   ├── raw/               # Dữ liệu gốc chưa qua xử lý
│   └── processed/         # Dữ liệu đã được tiền xử lý
├── models/                # Thư mục lưu checkpoint và weights của mô hình
├── notebooks/             # Thư mục chứa các file Jupyter Notebook để thử nghiệm
├── src/                   # Source code chính của dự án
│   ├── configs/           # Cấu hình huấn luyện (YAML/JSON)
│   ├── data/              # Dataset class và Dataloader
│   ├── models/            # Các kiến trúc mô hình (CNN, ViT, v.v.)
│   ├── utils/             # Hàm tiện ích (logging, metrics, visualization)
│   ├── train.py           # File chính để chạy huấn luyện
│   └── evaluate.py        # File để đánh giá mô hình đã train
├── .gitignore             # Quy định các file/thư mục không được đẩy lên Git
├── environment.yml        # Định nghĩa môi trường Conda
├── requirements.txt       # Danh sách các thư viện python cần thiết
└── README.md              # Tài liệu hướng dẫn sử dụng
```

## Khởi tạo và thiết lập môi trường

### 1. Cài đặt môi trường bằng Conda

Nếu bạn sử dụng Conda để quản lý môi trường:

```bash
# Tạo môi trường mới từ file environment.yml
conda env create -f environment.yml

# Kích hoạt môi trường
conda activate anti-spoofing
```

Hoặc nếu bạn muốn tự tạo môi trường và cài đặt qua `pip`:

```bash
# Tạo môi trường conda mới
conda create -n anti-spoofing python=3.10 -y
conda activate anti-spoofing

# Cài đặt thư viện thông qua requirements.txt
pip install -r requirements.txt
```

### 2. Sử dụng trong Jupyter Notebook

Để đăng ký môi trường conda này với Jupyter Notebook:

```bash
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=anti-spoofing --display-name "Python (anti-spoofing)"
```

## Cách chạy

### Huấn luyện mô hình

```bash
python src/train.py --epochs 10 --batch-size 32 --lr 1e-4
```

### Đánh giá mô hình

```bash
python src/evaluate.py --model-path models/best_model.pth
```

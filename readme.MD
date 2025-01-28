# ASR + Diarization + Summarization

## Giới thiệu
Dự án này kết hợp các công nghệ nhận dạng giọng nói tự động (ASR), phân tách giọng nói (Diarization), và tóm tắt văn bản (Summarization) để xử lý và phân tích file âm thanh một cách hiệu quả. Dự án sử dụng các mô hình và công cụ sau:

- **PhoWhisper**: Nhận dạng giọng nói tự động (ASR).
- **Pyannote**: Phân tách người nói (Speaker Diarization).
- **BartPho**: Mô hình tóm tắt văn bản zero-shot dành cho tiếng Việt.

## Cách cài đặt

### 1. Clone Repository
```bash
git clone https://github.com/ducdatit2002/Speech2Summary.git
cd Speech2Summary
```

### 2. Cài đặt Python dependencies
```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### 1. Chạy ứng dụng Python
Sau khi cài đặt, chạy ứng dụng bằng lệnh:
```bash
python app.py
```
Ứng dụng sẽ mở giao diện Gradio tại địa chỉ `http://0.0.0.0:7860`.

### 2. Chạy bằng Docker
Nếu bạn không muốn cài đặt Python thủ công, bạn có thể sử dụng Docker:

#### Kéo Docker image:
```bash
docker pull miavox/ichigo
```

#### Chạy container:
```bash
docker run --gpus all -it -p 7860:7860 --name ichigo_container miavox/ichigo /bin/bash
```

Trong container, chạy ứng dụng:
```bash
python app.py
```

Sau khi chạy, giao diện Gradio sẽ được mở tại `http://localhost:7860`.

## Tính năng chính

1. **Transcribe & Diarize**:
   - Chuyển giọng nói thành văn bản (ASR) và phân tách người nói.
   - Hiển thị kết quả từng phần theo thời gian thực (streaming).

2. **Summarize**:
   - Tóm tắt nội dung văn bản bằng mô hình BartPho.
   - Hỗ trợ chia văn bản thành các chunk nhỏ để tránh lỗi tràn bộ nhớ.

## Cách hoạt động

1. **Pipeline xử lý âm thanh**:
   - Đọc file âm thanh.
   - Chuyển đổi sang mono và resample về 16kHz.
   - Phân tách thành các đoạn nhỏ (chunk).

2. **Diarization**:
   - Xác định người nói và phân chia các đoạn âm thanh tương ứng.

3. **ASR**:
   - Chuyển đoạn âm thanh thành văn bản.

4. **Tóm tắt văn bản**:
   - Tóm tắt nội dung các đoạn văn bản lớn bằng cách chia nhỏ thành các chunk tối ưu.

## Demo
Upload file âm thanh hoặc sử dụng microphone để ghi âm, sau đó chọn các tính năng trên giao diện Gradio:
- Transcribe & Diarize
- Summarize

## Mô hình và công cụ sử dụng

1. **PhoWhisper**:
   - Mô hình ASR dành cho tiếng Việt.

2. **Pyannote**:
   - Công cụ phân tách giọng nói (Diarization).

3. **BartPho**:
   - Mô hình tóm tắt văn bản tiếng Việt zero-shot.

## Yêu cầu hệ thống
- **GPU** (tùy chọn): Sử dụng GPU để tăng tốc độ xử lý.
- **Python**: Phiên bản >= 3.11.8.
- **Docker** (nếu dùng): Docker Engine >= 20.10.

## Liên hệ
Nếu bạn có câu hỏi hoặc cần hỗ trợ, vui lòng liên hệ qua email hoặc GitHub của chúng tôi.
Email: ducdatit2002@gmail.com

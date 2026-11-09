# 🎓 Tool EOS FE - Bộ Công Cụ Tạo Đề Thi & Luyện Tập Trắc Nghiệm

Hệ thống tự động quét ảnh chụp câu hỏi từ các nền tảng (FuOverflow, ngân hàng đề thi FPT), sử dụng **Thị giác máy tính (Computer Vision)** và **Mô hình OCR (Deep Learning)** để bóc tách câu hỏi, lọc bình luận sinh viên, tự động xác định đáp án đúng và xuất ra các trang web luyện thi phong cách EOS chạy độc lập offline/online.

---

## 🏆 Tích Hợp Bảng Xếp Hạng Online (Google Sheets API)
* **Nhập tên định danh**: Khi vào web, người dùng nhập Tên/Nickname một lần duy nhất (tự động lưu vào trình duyệt).
* **Ghi nhận điểm số tự động**: Sau khi nộp bài, điểm số, thời gian làm và ngày nộp được gửi tự động về Google Sheets của bạn.
* **Bảng Vàng Vinh Danh (Leaderboard)**: Nút **[🏆 BẢNG XẾP HẠNG]** trên Header cho phép xem Top thí sinh điểm cao nhất kèm huy chương 🥇🥈🥉, lọc theo đề hiện tại hoặc toàn bộ các đề.

---

## 🌟 2 Chế Độ Luyện Thi Chuyên Biệt

Bộ công cụ cung cấp 2 chế độ học tập tùy chọn theo nhu cầu:

### 1. 🏛️ Chế độ Thi Thử Chuẩn EOS (`index.html`)
* **Mô phỏng chân thực phần mềm thi EOS của FPT**: Danh sách chọn đáp án dọc bên trái cho câu hỏi hiện tại, hình ảnh câu hỏi sạch bên phải (đã tự động cắt bỏ phần bình luận).
* **Nộp bài & Chấm điểm**: Có **đồng hồ đếm thời gian làm bài (Stopwatch)** tính từ lúc bắt đầu đến khi nộp bài. Sau khi bấm **Nộp bài**, hệ thống hiển thị tổng thời gian làm bài, điểm số thang 10 và xếp loại.
* **Xem lại câu sai (Review Mode)**: Chế độ lọc riêng các câu làm sai, lật mở ảnh gốc kèm đầy đủ comment thảo luận của sinh viên trên FuOverflow và đối soát lý thuyết giáo trình.
* **Cách tạo**: Chạy file `tao_de_eos.cmd`.

### 2. ⚡ Chế độ Luyện Tập Phản Xạ Tức Thì (`luyen_tap.html`)
* **Kiểm tra ngay sau mỗi câu**: Tích chọn đáp án và ấn **`Enter`**:
  * 🟢 **Đúng**: Ô đáp án sáng **Xanh lá**, tự động nhảy sang câu tiếp theo sau 0.6s.
  * 🔴 **Sai**: Ô chọn báo **Đỏ**, ô đúng sáng **Xanh lá**, hiển thị ảnh gốc FuOverflow kèm phân tích giải thích để bạn đối chiếu kiến thức ngay lập tức. Sau đó ấn **`Enter`** hoặc **`➡`** để sang câu tiếp theo.
* **Thống kê thời gian thực**: Có **đồng hồ bấm giờ (Stopwatch)** theo dõi thời gian luyện tập, đi kèm bộ đếm câu Đúng/Sai, hỗ trợ nút lọc **"Câu làm SAI"** và **"Làm lại câu này (Phím R)"**.
* **Cách tạo**: Chạy file `tao_de_luyen_tap.cmd`.

---

## ⌨️ Hệ Thống Phím Tắt Siêu Tốc

| Phím | Chức năng |
| :---: | :--- |
| **`1` `2` `3` `4` `5`** | Chọn nhanh phương án **A, B, C, D, E** |
| **`A` `B` `C` `D` `E`** | Chọn nhanh phương án tương ứng |
| **`Enter`** | Kiểm tra đáp án / Chuyển câu (ở chế độ Luyện tập) |
| **`➡` (Mũi tên phải)** | Chuyển sang câu tiếp theo (ở câu 60 sẽ **tự động vòng về câu 1**) |
| **`⬅` (Mũi tên trái)** | Quay lại câu trước (ở câu 1 sẽ vòng đến câu cuối cùng) |
| **`R`** | Làm lại câu hỏi hiện tại (ở chế độ Luyện tập) |

---

## 🧠 Cơ Chế Thuật Toán Đọc Ảnh & Bầu Chọn Đáp Án

1. **Phân vùng ảnh thích ứng (Adaptive Layout Splitting)**: Quét ma trận pixel tối để tự động phân định ranh giới giữa ảnh đề thi bên trái (nền trắng) và cột thảo luận bên phải (nền tối), cắt xuất ảnh câu hỏi sạch vào thư mục `questions/`.
2. **Lọc màu pixel BGR tách biệt User và Comment**: Trên FuOverflow, tên user có màu xanh lam và nội dung comment có màu trắng. Thuật toán đo hiệu số kênh màu `Blue - Red > 45` để loại bỏ 100% tên user, tránh đếm nhầm các chữ cái trong nickname (như `KhanhNgan`, `Asaki0556`...).
3. **Mô hình OCR hiện đại (RapidOCR + ONNX Runtime)**: Sử dụng DBNet và SVTR để bóc tách văn bản tiếng Việt và nhận diện chuẩn xác các ký tự đơn lẻ (A, B, C, D).
4. **Phân tích ngữ cảnh & Biểu quyết (Weighted Consensus)**:
   * Loại trừ các câu nghi vấn / phủ định (*"sao k phải là D"*, *"A sai nhé"*).
   * Nhân trọng số x2.5 cho các comment chốt hạ (*"--> B"*, *"=> A"*).
   * Tự động nhận diện tổ hợp câu chọn nhiều đáp án (*"ABD"*, *"ABC"*) và câu có 5 phương án (*"E. Tất cả phương án"*).
   * Đối soát nội dung ngữ nghĩa với 4 phương án nếu comment giải thích bằng chữ.

---

## 🚀 Hướng Dẫn Sử Dụng

### Yêu Cầu Môi Trường
* Máy tính đã cài đặt **Python 3.8+** ([Tải tại đây](https://www.python.org/downloads/) - *Lưu ý tích chọn "Add python.exe to PATH"*).
* Các thư viện cần thiết (`rapidocr-onnxruntime`, `opencv-python`) sẽ được file `.cmd` **tự động cài đặt ngầm trong lần đầu chạy**.

### Các Cách Chạy Tool

#### 🔹 Cách 1: Chạy từ Menu danh sách (Khuyên dùng)
1. Click đúp chuột vào file:
   * **`tao_de_eos.cmd`**: Nếu muốn tạo đề thi thử chuẩn EOS (`index.html`).
   * **`tao_de_luyen_tap.cmd`**: Nếu muốn tạo đề luyện phản xạ tức thì (`luyen_tap.html`).
2. Giao diện Console hiển thị danh sách toàn bộ các đề có trong thư mục:
   * Nhập số thứ tự đề muốn tạo (ví dụ: `1` hoặc nhiều đề: `1, 3, 5`).
   * Nhập **`NEW`** để tạo cho tất cả các đề chưa được tạo.
   * Nhập **`ALL`** để tạo toàn bộ danh sách.

#### 🔹 Cách 2: Kéo - Thả (Drag & Drop)
* Kéo một folder đề thi bất kỳ thả thẳng vào biểu tượng file `.cmd`, tool sẽ tự xử lý riêng cho folder đó.

#### 🔹 Cách 3: Đặt trực tiếp trong folder đề
* Copy file `.cmd` vào bên trong thư mục chứa ảnh đề (`Q001.png`, `Q002.png`...) rồi click đúp chạy.

---

## 📂 Cấu Trúc Thư Mục Một Đề Sau Khi Tạo

Sau khi chạy tool, bên trong mỗi folder đề thi sẽ tự động sinh ra:
```text
📁 Đề Thi FE MLN122 - SP26 - C2FE/
├── 📁 questions/             # 60 ảnh câu hỏi sạch (đã cắt bỏ comment để ôn thi)
├── 📄 Q001.png ... Q060.png  # Ảnh gốc FuOverflow (hiển thị khi xem lại câu sai)
├── 📄 extracted_data.json    # Dữ liệu bóc tách OCR thô & thống kê vote
├── 📄 quiz_data.json         # Dữ liệu chuẩn hóa câu hỏi, đáp án đúng & giải thích
├── 🌐 index.html             # Web thi thử phong cách EOS FPT
└── 🌐 luyen_tap.html         # Web luyện tập phản xạ (Enter kiểm tra đúng/sai luôn)
```

---

## 👥 Chia Sẻ Cho Bạn Bè Trong Nhóm

Các bạn trong nhóm **KHÔNG CẦN** cài đặt Python:
1. **Làm Offline**: Nén thư mục đề thành file `.zip` gửi qua Zalo/Drive. Bạn bè tải về, giải nén và click đúp vào file `index.html` hoặc `luyen_tap.html` là làm được ngay trên trình duyệt (Chrome, Edge, Cốc Cốc...).
2. **Làm Online (Cả điện thoại & Máy tính)**: Kéo thả thư mục đề vào [app.netlify.com/drop](https://app.netlify.com/drop) (hoặc Vercel / GitHub Pages), bạn sẽ nhận được một đường link web miễn phí để cả nhóm cùng truy cập luyện thi mọi lúc mọi nơi!

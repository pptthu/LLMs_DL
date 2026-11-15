# Khuôn model card (v2)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-16T01:40:17+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-16T01:40:17+07:00
- Trạng thái: **Duyệt: Toàn, 2026-11-16**
- Dựa trên: `MODEL_CARD_TEMPLATE_v1.md`; model card mẫu `Lab-DL/labs/lab1/docs/models/mlp_baseline_v1.md` (sơ đồ mermaid + bảng tham số từng lớp).
- Thay đổi so với v1: **sơ đồ kiến trúc bắt buộc** (mermaid) kèm bảng số tham số từng thành phần; thêm phần "Log ghi gì về model". v1 giữ nguyên, không dùng.
- Dùng cho: mỗi model / tokenizer bài dùng. Bản đã điền lưu ở `docs/models/<tên>_v<N>.md`.

## Cách dùng
- **Thông tin model** lấy từ model card và `config.json` trên Hugging Face; ghi ngày tra.
- **Sơ đồ kiến trúc**: vẽ bằng mermaid, đi từ input đến output, ghi shape tensor ở mỗi bước chính. Thứ tự lớp phải có nguồn (paper, model card, hoặc mã `transformers`).
- **Bảng tham số**: tự tính từ `config.json`, rồi đối chiếu với số đếm bằng code trong notebook (`sum(p.numel() for p in model.parameters())`).
- **Cách lấy** phải chạy được đúng như viết (kiểm ở smoke run).
- Bài không train nên không có phần "Trained model"; thay bằng "Cách dùng trong bài".

## How to do: từng bước
1. Mở trang model trên Hugging Face: đọc model card, `config.json`, `generation_config.json`; ghi ngày tra.
2. Điền "Thông tin model" (nguồn, revision, giấy phép, kiến trúc, số tham số, ngữ cảnh, từ điển).
3. Vẽ sơ đồ mermaid từ input đến output, ghi shape ở mỗi bước chính; ghi nguồn thứ tự lớp.
4. Tự tính bảng tham số từ `config.json`; đối chiếu với số trong model card.
5. Viết "Cách lấy" (lệnh `from_pretrained`), chạy thử ở smoke run.
6. Viết "Cách dùng trong bài"; sau smoke run, đối chiếu số tham số đếm bằng code với bảng ở bước 4.

## Khuôn

    # Model card — <tên>  (v<N>)
    - Tác giả, ngày tạo | Người cập nhật, ngày cập nhật | Ngày tra nguồn
    ## Purpose
    <1–2 câu: dùng để làm gì trong bài, so với model nào>
    ## Thông tin model
    - Nguồn: <link Hugging Face> | Revision: <commit hash> | Giấy phép: <…> | Cần token: <có / không>
    - Kiến trúc: <…> | Số tham số: <…> | Ngữ cảnh: <…> | Từ điển: <…>
    - Giai đoạn huấn luyện: <pretraining / post-training>
    ## Sơ đồ kiến trúc  (bắt buộc)
    ```mermaid
    flowchart TB
        In["Input [shape]"] --> ... --> Out["Output [shape]"]
    ```
    | Thành phần | Cách tính | Số tham số |
    |---|---|---:|
    Nguồn thứ tự lớp: <…>
    ## Cách lấy
    - Lệnh: `<from_pretrained(...)>` | Dung lượng: <…> | Lưu ở: bộ nhớ đệm Hugging Face (không commit)
    ## Cách dùng trong bài
    - Mục / phần notebook: <…>
    - Tham số: dtype <…>, do_sample <…>, max_new_tokens <…>
    - Kết quả: <outputs/results/…>
    - Ghi chú / gotcha: <…>
    ## Log ghi gì về model
    - Tên + revision, dtype, device_map, kiến trúc in từ `model.config`, số tham số đếm bằng code (RUN_LOG_TEMPLATE_v2 §2.1)

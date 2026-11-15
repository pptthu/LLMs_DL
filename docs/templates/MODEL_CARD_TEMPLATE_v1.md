# Khuôn model card (v1)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00
- Dựa trên: `Lab-DL/labs/lab1/docs/templates/MODEL_CARD_TEMPLATE_v1.md` (Thư). Lab 1 có phần "Trained model" vì nhóm tự train; bài này **không train**, nên thay bằng phần "Cách dùng trong bài".
- Dùng cho: mỗi model / tokenizer bài dùng. Bản đã điền lưu ở `docs/models/<tên>_v<N>.md`.

## Cách dùng
- Phần **Thông tin model** lấy từ model card và `config.json` trên Hugging Face; ghi ngày tra.
- Phần **Cách lấy** phải chạy được đúng như viết (kiểm ở smoke run).
- Phần **Cách dùng trong bài** ghi mục, phần notebook, tham số, file kết quả.

## Khuôn

    # Model card — <tên>  (v<N>)
    - Tác giả, ngày tạo | Người cập nhật, ngày cập nhật | Ngày tra nguồn
    ## Purpose
    <1–2 câu: dùng để làm gì trong bài, so với model nào>
    ## Thông tin model
    - Nguồn: <link Hugging Face> | Giấy phép: <…> | Cần token: <có / không>
    - Kiến trúc: <…> | Số tham số: <…> | Ngữ cảnh: <…> | Từ điển: <…>
    - Giai đoạn huấn luyện: <pretraining / post-training>
    - Sơ đồ: <mermaid nếu cần>
    ## Cách lấy
    - Lệnh: `<from_pretrained(...)>` | Dung lượng: <…> | Lưu ở: bộ nhớ đệm Hugging Face (không commit)
    ## Cách dùng trong bài
    - Mục / phần notebook: <…>
    - Tham số: dtype <…>, do_sample <…>, max_new_tokens <…>
    - Kết quả: <outputs/results/…>
    - Ghi chú / gotcha: <…>

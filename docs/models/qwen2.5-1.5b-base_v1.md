# Model card — Qwen2.5-1.5B (bản base)  (v1)
- Toàn (AI hỗ trợ), 2026-11-15 | Ngày tra nguồn: 2026-11-14/15 | Trạng thái: **[đề xuất]**

## Purpose
Chỉ dùng ở mục 5, để so với `Qwen2.5-1.5B-Instruct`: cùng kiến trúc, cùng số tham số, khác giai đoạn huấn luyện.

## Thông tin model
- **Nguồn:** https://huggingface.co/Qwen/Qwen2.5-1.5B | **Giấy phép:** Apache-2.0 | **Cần token:** không (`gated: false`)
- **Kiến trúc và thông số:** giống bản Instruct (xem `qwen2.5-1.5b-instruct_v1.md`).
- **Giai đoạn huấn luyện:** chỉ pretraining. Model card ghi **không khuyến nghị** dùng bản base để hội thoại.

## Cách lấy
```python
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16, device_map="auto")
base.eval()
```
- Tải ở phần 5, không tải ở phần 0.
- Thiếu bộ nhớ GPU thì `del model; torch.cuda.empty_cache()` trước, hoặc nạp lần lượt.
- **Chưa chạy thử.**

## Cách dùng trong bài
- **Mục / phần:** 5.
- **Đầu vào:** văn bản thô, **không** dùng chat template (bản base không được huấn luyện với định dạng hội thoại). Ghi rõ trong markdown của notebook.
- **Tham số sinh:** giống hệt bản Instruct, truyền tường minh (`do_sample=False`, cùng `max_new_tokens`). Bản base mặc định `do_sample: false`, bản Instruct mặc định `true`, nên không truyền thì so sánh không công bằng.
- **Kết quả:** `outputs/results/base_vs_instruct.csv`.

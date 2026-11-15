# Model card — GPT-2 tokenizer  (v1)
- Toàn (AI hỗ trợ), 2026-11-15 | Ngày tra nguồn: 2026-11-15 | Trạng thái: **[đề xuất]**

## Purpose
Chỉ dùng ở mục 3, để so số token của câu tiếng Việt và tiếng Anh với tokenizer của Qwen2.5. **Chỉ tải tokenizer, không tải model.**

## Thông tin
- **Nguồn:** https://huggingface.co/openai-community/gpt2 | **Giấy phép:** MIT | **Cần token:** không (`gated: false`)
- **Loại:** byte-level BPE; từ điển 50,257 = 256 byte + 50,000 lần gộp + 1 token kết thúc văn bản (Hugging Face, *Summary of the tokenizers*).
- **File tokenizer:** `vocab.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json`.
- Được huấn luyện chủ yếu trên văn bản tiếng Anh → **có thể** tách tiếng Việt vụn hơn. Chưa đo; số thật lấy từ notebook phần 3.

## Cách lấy
```python
from transformers import AutoTokenizer
gpt2_tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
```
Vài MB, chạy được trên CPU. **Chưa chạy thử.**

## Cách dùng trong bài
- **Mục / phần:** 3.
- **Đếm token:** `len(gpt2_tok(text, add_special_tokens=False)["input_ids"])`, cùng cách cho tokenizer Qwen để so công bằng.
- **Kết quả:** `outputs/results/token_counts.csv`.

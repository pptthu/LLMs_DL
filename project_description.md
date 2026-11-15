# Presentation LLMs — tổng quan dự án

- **Created**: 2026-11-15T16:30:00+07:00 (Toàn, AI hỗ trợ) | Trạng thái: **[đề xuất]**, chờ Thư duyệt
- Desc hiện hành: [`docs/presentation_description_v1.md`](docs/presentation_description_v1.md) · Plan: [`docs/plans/plan_presentation_llm_v2.md`](docs/plans/plan_presentation_llm_v2.md) · Mục lục: [`docs/00_README_v1.md`](docs/00_README_v1.md)

## 1. Define

| Mục | Nội dung |
|---|---|
| Đề | Tìm hiểu mô hình ngôn ngữ lớn (LLMs) — 10 mục, mỗi mục có yêu cầu minh hoạ |
| Mục tiêu của cô | Giải thích cách LLM học và sinh văn bản; phân biệt các giai đoạn huấn luyện; nhận biết khả năng và hạn chế |
| Sản phẩm nộp | (1) Slide PDF/PPTX; (2) notebook + hướng dẫn chạy; (3) bộ câu thử nghiệm, kết quả, phân tích lỗi; (4) tài liệu tham khảo |
| Thời lượng | 15–20 phút, khoảng 21 slide; mỗi người khoảng 5 câu Q&A |
| Yêu cầu thêm | Slide nêu thư viện và môi trường; demo trên slide; slide có thực nghiệm; giải thích từng dòng notebook; mở notebook khi slide không có code |
| Phạm vi | Không fine-tune; code chỉ trong notebook; 1 model chính 1.5B |
| Chờ cô xác nhận | Cách demo mục 5 (base vs instruct) và mục 8 (không RAG vs có RAG); dùng UIT-VSFC. Đến Ngày 4 chưa trả lời thì chạy theo phương án hiện tại và ghi rõ trên slide |

**Yêu cầu minh hoạ → phần notebook → slide**

| # | Yêu cầu minh hoạ của đề | Phần | Slide | Người |
|---|---|---|---|---|
| 1 | Ví dụ các tác vụ: hỏi đáp, tóm tắt, dịch, sinh mã | 1 | 4 | Muội |
| 2 | Sơ đồ từ token đầu vào đến xác suất token tiếp theo | 2 | 5–6 | Muội |
| 3 | Tokenize một câu tiếng Việt và một câu tiếng Anh | 3 | 7 | Muội |
| 4 | Tạo các cặp ngữ cảnh → token mục tiêu từ một câu | 4 | 8–9 | Phúc |
| 5 | Sơ đồ mục tiêu và dữ liệu của từng giai đoạn | 5 | 10–11 | Phúc |
| 6 | Cùng một prompt với các cấu hình sinh khác nhau | 6 | 12–13 | Phúc |
| 7 | So sánh zero-shot và few-shot trên cùng tác vụ | 7 | 14 | Toàn |
| 8 | Bảng so sánh dữ liệu cần có và trường hợp sử dụng | 8 | 15–16 | Muội |
| 9 | Một câu trả lời sai và cách kiểm chứng | 9 | 17–18 | Phúc |
| 10 | Bảng kết quả, ví dụ đầu ra và phân tích lỗi | 10 | 19–20 | Toàn |

## 2. Design

### 2.1 Môi trường và model
- Kaggle, GPU T4, `torch.float16`.
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (chính), `Qwen/Qwen2.5-1.5B` (mục 5), tokenizer `openai-community/gpt2` (mục 3).
- Dữ liệu: UIT-VSFC (mục 7, 10).
- Cách lấy từng thứ: `README.md` §Setup; model card ở `docs/models/`.

### 2.2 Notebook (phần 0–10)
```text
Phần 0  Setup (Phúc): cấu hình, thư viện, tải model + tokenizer + UIT-VSFC, env.json
  │
  ├─ Phần 1–3 (Muội):  tác vụ mẫu · kiến trúc + causal mask + top-5 · tokenization
  ├─ Phần 4–6 (Phúc):  cặp ngữ cảnh → đích + loss · base vs instruct · 5 cấu hình sinh
  ├─ Phần 7   (Toàn):  zero-shot vs 3-shot × 3 bộ, 500 câu UIT-VSFC
  ├─ Phần 8   (Muội):  RAG trên slide môn học (TF-IDF)
  ├─ Phần 9   (Phúc):  hallucination · thiên lệch · ngữ cảnh dài
  └─ Phần 10  (Toàn):  metric, McNemar, confusion matrix, bảng tổng hợp mọi phần
```

### 2.3 File dữ liệu và kết quả mỗi phần [đề xuất]
| Phần | Đầu vào trong `data/` | Đầu ra trong `outputs/` |
|---|---|---|
| 0 | — | `results/env.json` |
| 1 | `tasks_prompts.json` | `results/tasks_outputs.csv` |
| 2 | — | `results/model_config.json`, `results/top5_next_tokens.csv`, `figures/causal_mask.png`, `figures/attention_qwen.png` |
| 3 | — | `results/token_counts.csv` |
| 4 | — | `results/per_token_loss.csv` |
| 5 | `base_vs_instruct.json` | `results/base_vs_instruct.csv` |
| 6 | — | `results/decoding_samples.csv` |
| 7 | `test_500_ids.json`, `fewshot_pools.json` | `results/icl_zero_shot.csv`, `results/icl_3shot_set{1,2,3}.csv` |
| 8 | `rag_questions.csv` | `results/rag_results.csv` |
| 9 | `hallucination.csv`, `bias_prompts.json`, `long_context.json` | `results/hallucination_results.csv`, `results/bias_results.csv`, `results/long_context_results.csv` |
| 10 | — | `results/icl_metrics.json`, `results/summary_table.csv`, `figures/confusion_matrix_*.png` |

### 2.4 Quy ước notebook
Theo `docs/rules/RULES_v1.md` §2:
- markdown trước mỗi cell code;
- comment tiếng Việt cho dòng không hiển nhiên;
- cell "Phân tích" sau output;
- hàm dùng lại định nghĩa ngay trước chỗ dùng, kèm `assert`;
- tham số sinh truyền tường minh;
- `SMOKE` ở cell đầu.

### 2.5 Rủi ro kỹ thuật và cách xử lý
| Rủi ro | Cách xử lý |
|---|---|
| Bản Instruct mặc định lấy mẫu ngẫu nhiên → kết quả không lặp lại được | Luôn truyền `do_sample=False` cho greedy; in `generation_config` |
| bf16 chạy chậm trên T4 | Dùng `float16`; fp16 lạ thì `float32` |
| Script tải UIT-VSFC không chạy với `datasets` ≥ 4.0 | Đọc Parquet bằng `pandas` |
| Rò dữ liệu: câu few-shot trùng câu test | Ví dụ lấy từ split train, test từ split test, `assert` không trùng |
| Số trên slide lệch notebook | Chỉ lấy số từ `outputs/results/` của lần Save & Run All; ghi đường dẫn dưới bảng |
| Xung đột khi nhiều người sửa một `.ipynb` | Mỗi người một file trong `notebook/parts/`; Toàn ghép |
| Không tải được model khi trình bày | Notebook nộp đã có output; xuất HTML dự phòng |

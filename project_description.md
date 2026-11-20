# Presentation LLMs — tổng quan dự án

- **Created**: 2026-11-15T16:30:00+07:00 (Toàn, AI hỗ trợ) | **Last Updated**: 2026-11-21 (Toàn, AI hỗ trợ) | Trạng thái: **Duyệt: Toàn, 2026-11-16**; cập nhật theo hiện trạng 21/11 (Thư review)
- Desc hiện hành: [`docs/presentation_description_v3.md`](docs/presentation_description_v3.md) · Khung slide: [`docs/plans/khung_presentation_llms_v5.md`](docs/plans/khung_presentation_llms_v5.md) · Mục lục: [`docs/00_README_v3.md`](docs/00_README_v3.md)
- Giải thích lý thuyết, code, output từng cell: [`docs/giai_thich_du_an_v4.md`](docs/giai_thich_du_an_v4.md) · Kịch bản và thứ tự ôn: [`docs/scripts/kich_ban_toan_v2.md`](docs/scripts/kich_ban_toan_v2.md)

## 1. Define

| Mục | Nội dung |
|---|---|
| Đề | Tìm hiểu mô hình ngôn ngữ lớn (LLMs) — 10 mục, mỗi mục có yêu cầu minh hoạ |
| Mục tiêu của cô | Giải thích cách LLM học và sinh văn bản; phân biệt các giai đoạn huấn luyện; nhận biết khả năng và hạn chế |
| Sản phẩm nộp | (1) Slide PDF/PPTX; (2) notebook + hướng dẫn chạy; (3) bộ câu thử nghiệm, kết quả, phân tích lỗi; (4) tài liệu tham khảo |
| Thời lượng | 15–20 phút, 16 slide, gồm cả lúc mở notebook |
| Người trình bày | **Toàn trình bày cả 10 mục** và viết toàn bộ code (đổi phân công 21/11); Thư chủ repo, review |
| Yêu cầu thêm | Slide nêu thư viện và môi trường; demo trên slide; slide có thực nghiệm; giải thích từng dòng notebook; mở notebook khi slide không có code |
| Phạm vi | Không fine-tune; code chỉ trong notebook; 1 model chính 1.5B |

**Yêu cầu minh hoạ → phần notebook → slide**

| # | Yêu cầu minh hoạ của đề | Phần (cell) | Slide |
|---|---|---|---|
| 1 | Ví dụ các tác vụ: hỏi đáp, tóm tắt, dịch, sinh mã | 1 (1.1–1.5) | 3 |
| 2 | Sơ đồ từ token đầu vào đến xác suất token tiếp theo | 2 (sơ đồ + 2.1–2.5) | 4 |
| 3 | Tokenize một câu tiếng Việt và một câu tiếng Anh | 3 (3.1, 3.1b–3.5) | 5 |
| 4 | Tạo các cặp ngữ cảnh → token mục tiêu từ một câu | 4 (4.1–4.5) | 6 |
| 5 | Sơ đồ mục tiêu và dữ liệu của từng giai đoạn | 5 (5.0 sơ đồ; 5.1–5.4 base vs Instruct) | 7 |
| 6 | Cùng một prompt với các cấu hình sinh khác nhau | 6 (6.1–6.4) | 8 |
| 7 | So sánh zero-shot và few-shot trên cùng tác vụ | 7 (7.1–7.5) | 9 |
| 8 | Bảng so sánh dữ liệu cần có và trường hợp sử dụng | 8 (bảng + 8.1–8.4) | 10 |
| 9 | Một câu trả lời sai và cách kiểm chứng | 9 (9.1–9.3; thêm 9.4 thiên lệch, 9.5 ngữ cảnh, 9.6 chất lượng) | 11 |
| 10 | Bảng kết quả, ví dụ đầu ra và phân tích lỗi | 10 (10.1–10.4) | 12–13 |

## 2. Design

### 2.1 Môi trường và model
- Kaggle, GPU T4, `torch.float16`, seed 42. Lần chạy chính thức: **`20261121_010447_normal`** (9.2 phút).
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (chính), `Qwen/Qwen2.5-1.5B` (mục 5), tokenizer `openai-community/gpt2` (mục 3).
- Dữ liệu: UIT-VSFC (mục 7, 9.6, 10); UIT-ViQuAD 2.0 (mục 8, 9.1–9.3, 9.5); WinoGender và Nemotron-Personas-Vietnam (mục 9.4).
- Cách lấy từng thứ: `README.md` §Setup và cell markdown thứ 2 của notebook; model card ở `docs/models/*_v2.md`.
- Input lấy từ đâu, output lưu ở đâu: `docs/rules/FOLDER_STRUCTURE_v3.md`. Không train nên không có trọng số tự tạo.

### 2.2 Notebook (phần 0–10)
Code và trình bày mọi phần: **Toàn**.
```text
Phần 0  Setup: cấu hình, log, show_table / show_input, tải model + tokenizer + UIT-VSFC, env.json
  │
  ├─ Phần 1:  4 tác vụ mẫu, một model, chỉ đổi prompt
  ├─ Phần 2:  thông số · causal mask · attention thật · top-5 token tiếp theo
  ├─ Phần 3:  token, token ID, byte UTF-8 · gom theo từ · Qwen vs GPT-2 · embedding
  ├─ Phần 4:  cặp ngữ cảnh → mục tiêu · cross-entropy từng token · perplexity
  ├─ Phần 5:  sơ đồ 3 giai đoạn · base vs Instruct
  ├─ Phần 6:  sinh từng bước · temperature, top-p · 5 cấu hình × 3 lần
  ├─ Phần 7:  zero-shot vs few-shot (4 ví dụ), 2,999 câu UIT-VSFC
  ├─ Phần 8:  không RAG vs có RAG (TF-IDF top-3), UIT-ViQuAD 2.0
  ├─ Phần 9:  hallucination + kiểm chứng · thực thể không tồn tại · thiên lệch · giới hạn ngữ cảnh · chất lượng theo tác vụ, dữ liệu
  └─ Phần 10: accuracy, McNemar, confusion matrix, phân tích lỗi, bảng tổng hợp
```

### 2.3 File dữ liệu và kết quả mỗi phần
Mỗi phần còn ghi log `outputs/logs/<YYYYMMDD_HHMMSS>_<smoke|normal>/partNN_<tên>.log` và `_records.csv`; cả lần chạy có `run.log`, `config.json`, `env.json` (khuôn `docs/templates/RUN_LOG_TEMPLATE_v3.md`).

| Phần | Đầu vào | Đầu ra trong `outputs/` |
|---|---|---|
| 0 | Hugging Face Hub (model), UIT-VSFC (Parquet) | `results/env.json` |
| 1 | `data/tasks_prompts.json` | `results/tasks_outputs.csv` |
| 2 | câu viết trong cell | `results/model_config.json`, `results/top5_next_tokens.csv`, `figures/causal_mask.png`, `figures/attention_qwen.png`, `figures/top5_next_tokens.png` |
| 3 | câu viết trong cell | `results/token_counts.csv`, `figures/embedding_qwen.png` |
| 4 | câu viết trong cell | `results/per_token_loss.csv`, `results/loss_summary.csv`, `figures/context_target_pairs.png`, `figures/per_token_loss.png` |
| 5 | `data/base_vs_instruct.json` | `results/base_vs_instruct.csv`, `figures/training_stages.png` |
| 6 | prompt viết trong cell | `results/decoding_steps.csv`, `results/temperature_top_p.csv`, `results/decoding_samples.csv`, `figures/temperature_distribution.png` |
| 7 | UIT-VSFC train / validation / test | `results/vsfc_test_ids.csv`, `results/fewshot_examples.json`, `results/prompt_selection.csv`, `results/icl_zero_shot.csv`, `results/icl_few_shot.csv` |
| 8 | UIT-ViQuAD 2.0 validation | `results/rag_questions_used.csv`, `results/rag_results.csv` |
| 9 | UIT-ViQuAD 2.0; 3 câu tiền đề sai (trong cell); WinoGender (GitHub); Nemotron-Personas-Vietnam (shard 0) | `results/hallucination_results.csv`, `results/hallucination_fictitious.csv`, `results/bias_en_winogender.csv`, `results/bias_vi_nemotron.csv`, `results/bias_summary.csv`, `results/context_needle.csv`, `results/context_cost.csv`, `results/quality_by_task.csv`, `results/quality_by_data.csv`, `figures/bias_scatter.png`, `figures/context_cost.png`, `figures/quality_by_data.png` |
| 10 | kết quả phần 1–9 | `results/icl_metrics.json`, `results/error_analysis_sample.csv`, `results/summary_table.csv`, `figures/confusion_matrix_icl.png`, `figures/accuracy_icl.png` |

### 2.4 Quy ước notebook
Theo `docs/rules/RULES_v3.md` §2:
- markdown trước mỗi cell code (mục đích, đầu vào, đầu ra);
- khung **ĐẦU VÀO** (`show_input()`) trước output của mỗi demo; bảng in bằng `show_table()` (kẻ vạch);
- comment tiếng Việt cho dòng không hiển nhiên;
- cell "Phân tích" sau output;
- hàm dùng lại định nghĩa ngay trước chỗ dùng, kèm `assert`;
- tham số sinh truyền tường minh;
- `SMOKE` ở cell đầu.

### 2.5 Rủi ro kỹ thuật và cách xử lý
| Rủi ro | Cách xử lý |
|---|---|
| Bản Instruct mặc định lấy mẫu ngẫu nhiên → kết quả không lặp lại được | Luôn truyền `do_sample=False` cho greedy; in `generation_config` |
| bf16 chạy chậm trên T4 | Dùng `float16` |
| Script tải UIT-VSFC không chạy với `datasets` ≥ 4.0 | Đọc Parquet bằng `pandas` |
| Rò dữ liệu: câu few-shot trùng câu test | Ví dụ lấy từ split train, test từ split test, `assert` không trùng |
| Hết bộ nhớ GPU khi đo 8,192 token (cell 9.5) | Dọn bộ nhớ trước mỗi độ dài; bắt `OutOfMemoryError`, ghi `OOM` thay vì dừng notebook |
| Số trên slide lệch notebook | Chỉ lấy số từ `outputs/results/` của run chính thức; ghi `run_id` dưới bảng |
| Không giải thích được code khi bị hỏi | Ôn theo `docs/scripts/kich_ban_toan_v2.md` Phần A; giải thích từng cell ở `docs/giai_thich_du_an_v4.md` |
| Không tải được model khi trình bày | Notebook nộp đã có output; không chạy lại khi trình bày |

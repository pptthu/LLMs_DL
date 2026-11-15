# Detail plan notebook: từng bước phần 0–10 (v1)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-16T02:00:00+07:00 | **Duyệt: Toàn, 2026-11-16**
- Vai trò trong quy trình (`rules/RULES_v2.md` §1): đây là bước 3 **Detail plan**, AI chi tiết hoá từ `presentation_description_v2.md`, `plan_presentation_llm_v3.md` và các file `research/noi_dung_*`. Người phụ trách mục đối chiếu với dàn ý desc mục mình (`docs/parts/`, chưa có) rồi Toàn duyệt.
- Code mọi phần: **Toàn**, trong `notebook/llm_presentation.ipynb`. Mỗi bước dưới đây = **một small prompt** = một hoặc vài cell; làm xong bước nào chạy kiểm bước đó rồi mới sang bước sau.
- Ký hiệu cell: `X.Y` = phần X, cell code thứ Y. Trước mỗi cell code có markdown (mục đích, đầu vào, đầu ra); sau output có cell "Phân tích".
- Log mỗi phần: `outputs/logs/<ts>_<mode>/partNN_<tên>.log` và `_records.csv`, cột theo `templates/RUN_LOG_TEMPLATE_v2.md` §3. Đường dẫn smoke: thay `outputs/` bằng `outputs/_smoke/`.

---

## Phần 0 — Setup (lý thuyết + code: Toàn, chốt 16/11)
**Mục tiêu:** cô hỏi "thư viện gì, chạy ở đâu, model lấy ở đâu, lấy thế nào" thì chỉ được ngay phần này.
**Input:** Hugging Face Hub (model, tokenizer, UIT-VSFC). **Output:** `results/env.json`, `logs/<ts>_<mode>/{run.log, config.json, env.json, part00_setup.log}`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 0.1 | Cell cấu hình: `SMOKE`, `SEED = 42`, tên 3 model, `N_TEST = 500` (smoke: 10), đường dẫn `OUT_DIR` (`outputs/` hoặc `outputs/_smoke/`), `RUN_ID = <YYYYMMDD_HHMMSS>_<mode>` theo giờ Việt Nam | In ra `RUN_ID`, `OUT_DIR` |
| 2 | 0.2 | Hàm log: tạo thư mục `OUT_DIR/logs/RUN_ID/`, hàm `log(part, text)` ghi thêm dòng vào `run.log` và `partNN_*.log`; ghi phần đầu `run.log` | `assert` thư mục tồn tại, `run.log` có dòng `RUN START` |
| 3 | 0.3 | Import thư viện; in phiên bản Python, `torch`, `transformers`, `accelerate`, `pandas`, `scikit-learn`, `statsmodels`; tên GPU, CUDA, VRAM | `assert torch.cuda.is_available()` |
| 4 | 0.4 | Tải tokenizer + `Qwen2.5-1.5B-Instruct` (`torch_dtype=torch.float16`, `device_map="auto"`), `model.eval()`; in `model.generation_config` | In `do_sample` mặc định (kỳ vọng `True`) |
| 5 | 0.5 | In kiến trúc từ `model.config` (số block, hidden, head Q/KV, FFN, vocab, context); đếm `sum(p.numel() for p in model.parameters())`; ghi vào log | `assert` số tham số = 1,543,714,304 |
| 6 | 0.6 | Tải tokenizer GPT-2 | `gpt2_tok.vocab_size == 50257` |
| 7 | 0.7 | Đọc UIT-VSFC train/test bằng `pd.read_parquet`; bỏ nhãn neutral | Kỳ vọng 11,426 / 3,166 dòng trước khi bỏ neutral (`research_presentation_llm_v1.md`) |
| 8 | 0.8 | Ghi `env.json`, `config.json`; dòng tổng kết phần 0 vào `run.log` | File tồn tại, đọc lại được bằng `json.load` |

## Phần 1 — Tác vụ mẫu (mục 1; lý thuyết: Muội)
**Input:** `data/tasks_prompts.json` (Muội soạn, 4 prompt theo `research/noi_dung_muc01_gioi_thieu_v1.md` §5.2). **Output:** `results/tasks_outputs.csv`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 1.1 | Hàm `chat_generate(model, tok, prompt, max_new_tokens)`: `apply_chat_template(..., add_generation_prompt=True)`, `generate(do_sample=False, ...)`, cắt phần prompt khỏi output | `assert` output là chuỗi không rỗng với 1 prompt thử |
| 2 | 1.2 | Chạy 4 tác vụ (hỏi đáp, tóm tắt, dịch, sinh mã); mỗi tác vụ 1 dòng log | 4 dòng trong `_records.csv` |
| 3 | 1.3 | Chạy hàm `is_even` do model sinh với n = 0, 1, 2, −3 (cell riêng, ghi rõ là code của model) | So kết quả với đáp án |
| 4 | 1.4 | Chấm tay theo cột "Cách kiểm output" đã ghi sẵn; ghi cột `judged` | Đủ 4 dòng có chấm |

## Phần 2 — Kiến trúc (mục 2; lý thuyết: Muội)
**Input:** model chính; `google-bert/bert-base-multilingual-cased`, `Qwen/Qwen2-VL-2B-Instruct`, `data/vlm_image.png`, `data/compare_prompts.json`. **Output:** `results/model_config.json`, `results/top5_next_tokens.csv`, `figures/causal_mask.png`, `figures/attention_qwen.png`, `results/compare_bert_qwen.csv`, `results/compare_vlm.csv`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 2.1 | Ghi `model_config.json` (lấy từ phần 0, bước 5) | Khớp bảng trong `models/qwen2.5-1.5b-instruct_v2.md` |
| 2 | 2.2 | `causal_mask(n)` cho câu "Hôm nay trời rất đẹp."; vẽ heatmap | `assert` ma trận tam giác dưới |
| 3 | 2.3 | Nạp lại model với `attn_implementation="eager"`, lấy trọng số attention 1 layer, 1 head (`output_attentions=True`); vẽ heatmap | Phần trên đường chéo bằng 0 |
| 4 | 2.4 | `top_k_next_tokens(model, tok, "Hôm nay trời rất", k=5)`: softmax ở vị trí cuối | Tổng xác suất top-5 ≤ 1 |
| 5 | 2.5 | Demo A: nạp `google-bert/bert-base-multilingual-cased`, `pipeline("fill-mask")` cho câu có `[MASK]` ở giữa; kiểm đáp án là 1 token của tokenizer mBERT | In top-5 + xác suất |
| 6 | 2.6 | Demo A: Qwen chỉ nhận phần bên trái, top-5 token tiếp theo; ghi `results/compare_bert_qwen.csv` | Đủ 2 model × câu mẫu |
| 7 | 2.7 | Demo B: giải phóng model eager; nạp `Qwen/Qwen2-VL-2B-Instruct` (`AutoProcessor`, `Qwen2VLForConditionalGeneration`) | In VRAM; lỗi `KeyError: 'qwen2_vl'` thì ghi phiên bản `transformers` |
| 8 | 2.8 | Demo B: cùng câu hỏi về `data/vlm_image.png`; Qwen2.5 chỉ nhận chữ, Qwen2-VL nhận ảnh + chữ; `do_sample=False`; chấm theo đáp án ghi sẵn; ghi `results/compare_vlm.csv` | Mỗi câu 2 dòng |

## Phần 3 — Tokenization (mục 3; lý thuyết: Muội)
**Input:** tokenizer Qwen, tokenizer GPT-2, 2 câu ví dụ. **Output:** `results/token_counts.csv`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 3.1 | `tokenize_text(tok, câu)`: bảng token + token ID cho câu Việt, câu Anh | — |
| 2 | 3.2 | `compare_token_counts(...)` cho 2 câu × 2 tokenizer, đếm với `add_special_tokens=False` | Bảng 2×2 đủ số |
| 3 | 3.3 | In `vocab_size` hai tokenizer | GPT-2 = 50,257 |
| 4 | 3.4 | Shape embedding: `model.get_input_embeddings().weight.shape` | (151936, 1536) |
| 5 | 3.5 | `tok.decode(ids)` trả lại câu gốc | `assert` bằng câu gốc |

## Phần 4 — Nguyên lý học (mục 4; lý thuyết: Phúc)
**Input:** model chính, câu ví dụ. **Output:** `results/per_token_loss.csv`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 4.1 | `make_context_target_pairs(ids)`: bảng ngữ cảnh → token mục tiêu | Số cặp = số token − 1 |
| 2 | 4.2 | `per_token_loss(model, tok, câu)` trong `torch.no_grad()`: P(token đúng), −log p từng vị trí | Mỗi loss ≥ 0 |
| 3 | 4.3 | Trung bình loss tự tính so với `model(ids, labels=ids).loss` | `torch.allclose(..., atol=1e-3)` |
| 4 | 4.4 | Perplexity = exp(loss), so với mốc đoán bừa | — |
| 5 | 4.5 | (Tuỳ chọn) loss câu xáo trộn từ | Loss cao hơn câu gốc (ghi đúng như thật) |

## Phần 5 — Base vs Instruct (mục 5; lý thuyết: Phúc; **chờ cô xác nhận**)
**Input:** `data/base_vs_instruct.json` (Phúc soạn, 4–6 prompt), `Qwen/Qwen2.5-1.5B`. **Output:** `results/base_vs_instruct.csv`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 5.1 | Nếu thiếu VRAM: `del` model eager ở phần 2, `torch.cuda.empty_cache()`; tải bản base | In VRAM còn trống |
| 2 | 5.2 | In `generation_config` của cả hai bản | Base `do_sample=False`, instruct `True` (mặc định) |
| 3 | 5.3 | Chạy cùng prompt: base nhận văn bản thô, instruct qua chat template; cả hai `do_sample=False`, cùng `max_new_tokens` | Mỗi prompt 2 dòng output |
| 4 | 5.4 | Chấm tay theo thang ghi sẵn (làm theo chỉ dẫn / không) | Đủ mọi prompt |

## Phần 6 — Sinh văn bản (mục 6; lý thuyết: Phúc)
**Input:** model chính, 1 prompt. **Output:** `results/decoding_samples.csv`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 6.1 | Sinh từng bước (vòng lặp lấy token argmax, in từng token) | In 5–10 bước |
| 2 | 6.2 | Định nghĩa 5 cấu hình (greedy, temp0.7_top_p0.9, temp1.0, temp1.5, top_p0.5), đặt **đủ** `top_k`, `repetition_penalty` | In dict cấu hình |
| 3 | 6.3 | Mỗi cấu hình 5 mẫu, seed `SEED + i` | 25 dòng |
| 4 | 6.4 | Chạy greedy 2 lần | `assert` 2 output y hệt |

## Phần 7 — Zero-shot / few-shot (mục 7; lý thuyết: Toàn)
**Input:** UIT-VSFC (phần 0), `data/test_500_ids.json`, `data/fewshot_pools.json`. **Output:** `results/icl_zero_shot.csv`, `results/icl_3shot_set{1,2,3}.csv`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 7.1 | Lấy 500 câu test (250/250, seed 42) → ghi `test_500_ids.json` lần đầu, các lần sau đọc lại | Đếm 250/250 |
| 2 | 7.2 | Lấy 3 bộ × 3 ví dụ từ split train → `fewshot_pools.json` | `assert` không trùng câu test |
| 3 | 7.3 | `build_prompt(text, examples)` theo template | In 1 prompt zero-shot, 1 prompt 3-shot |
| 4 | 7.4 | `parse_label(output)` theo luật `noi_dung_toan_muc07_muc10_v2.md` §6.4: nhận positive/negative và tích cực/tiêu cực | `assert` cho cả nhãn Anh, nhãn Việt, có cả hai nhãn, không có nhãn |
| 5 | 7.5 | Thử nhiều mẫu prompt (nhãn Anh / Việt, cách diễn đạt khác) trên split **validation** (vài chục câu), không dùng test; ghi từng mẫu + accuracy + tỉ lệ không parse được vào log; chốt một mẫu | Ghi mẫu đã chốt vào desc |
| 6 | 7.6 | Chạy 500 câu × 4 lượt, greedy, `max_new_tokens` khoảng 5; mỗi câu 1 dòng log (`acc_cumulative`) | 2,000 dòng ở normal run |

## Phần 8 — RAG (mục 8; lý thuyết: Muội; **chờ cô xác nhận**)
**Input:** `data/rag_questions.csv` (Muội soạn, 15–20 câu), slide môn học (Kaggle Dataset riêng, không commit). **Output:** `results/rag_results.csv`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 8.1 | Chia slide thành đoạn (mỗi slide một đoạn) | In số đoạn |
| 2 | 8.2 | `TfidfVectorizer(ngram_range=(1, 2))`, tìm k = 3 đoạn | Tỉ lệ câu có đoạn đúng trong top-3 |
| 3 | 8.3 | Hỏi không RAG và có RAG, cùng greedy | Mỗi câu 2 output |
| 4 | 8.4 | Chấm tự động (chuẩn hoá, khớp đáp án, token F1) theo `research/noi_dung_muc08_*_v2.md` §3.4; xuất file cho 2 người chấm mù | Đủ mọi câu, có cột nhãn tự động |
| 5 | 8.5 | Đọc lại nhãn của 2 người; tính % đồng ý, Cohen's kappa, McNemar exact, context recall@3, số câu từ chối đúng | In các số; ghi `rag_results.csv` |

## Phần 9 — Hạn chế (mục 9; lý thuyết: Phúc)
**Input:** `data/hallucination.csv`, `data/bias_prompts.json`, `data/long_context.json` (Phúc soạn). **Output:** `results/hallucination_results.csv`, `bias_results.csv`, `long_context_results.csv`.

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 9.1 | Hallucination: khoảng 20 câu, greedy; chấm tự động rồi 2 người chấm mù theo SimpleQA (Correct / Incorrect / Not attempted), `research/noi_dung_muc09_*_v2.md` §5.1b; tính %, F-score, kappa | Đủ mọi câu |
| 2 | 9.2 | Thiên lệch: prompt nghề nghiệp, nhiều mẫu, đếm từ chỉ giới tính | Bảng đếm |
| 3 | 9.3 | Ngữ cảnh dài: `make_long_context(fact, filler, position)` ở đầu / giữa / cuối; ghi số token | Mỗi vị trí 1 dòng đúng/sai |

## Phần 10 — Tổng hợp (mục 10; lý thuyết: Toàn)
**Input:** file kết quả phần 7 (và các phần khác cho bảng tổng hợp). **Output:** `results/icl_metrics.json`, `results/summary_table.csv`, `figures/confusion_matrix_*.png`; `docs/experiments/error_analysis_v1.md` (viết tay sau khi chạy).

| Bước | Cell | Làm gì | Kiểm |
|---|---|---|---|
| 1 | 10.1 | Accuracy, tỉ lệ không parse được cho 4 lượt | Số câu = 500 mỗi lượt |
| 2 | 10.2 | Confusion matrix 2×2 + cột "không parse được"; vẽ hình | Tổng mỗi hàng = 250 |
| 3 | 10.3 | McNemar (`exact=True`) zero-shot vs từng bộ 3-shot; Bonferroni (ngưỡng 0.05/3 ≈ 0.017), báo cả p chưa hiệu chỉnh | In b, c, p-value |
| 4 | 10.4 | Bảng tổng hợp mọi phần đọc từ file kết quả | Không chép số tay |
| 5 | 10.5 | Ghi phần cuối `run.log` (thời gian, file, lỗi) | Có dòng `RUN END` |

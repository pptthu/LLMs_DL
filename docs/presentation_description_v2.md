# Presentation LLMs — Tìm hiểu mô hình ngôn ngữ lớn  (v2)
- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-16T01:40:17+07:00
- Trạng thái: **Duyệt: Toàn, 2026-11-16** (Thư review) | Trạng thái công việc: In Progress
- Khuôn: `templates/DESC_TEMPLATE_v2.md`. Thay đổi so với v1:
  - thêm Backbone, Input/Output (lấy từ đâu, lưu ở đâu), Đánh giá từng phần, log;
  - How-to-do ghi cụ thể từng bước;
  - mục 2 thêm so sánh LLM với mô hình khác;
  - Toàn code toàn bộ, Toàn duyệt.
  - Phần AI chi tiết hoá đánh `[đề xuất]`. v1 giữ nguyên, không dùng.

## Background
- Môn Học sâu (UTH). Đề: "Tìm hiểu mô hình ngôn ngữ lớn — Large Language Models (LLMs)".
- Mục tiêu của cô: giải thích được cách LLM học và sinh văn bản, phân biệt các giai đoạn huấn luyện, nhận biết khả năng và hạn chế của mô hình. Gồm 10 mục, mỗi mục có yêu cầu minh hoạ.
- **Góp ý của cô (11/11):** không cần fine-tune, nhưng mọi mục phải có demo trên model thật kèm phân tích output.
- **Yêu cầu thêm:**
  - slide nêu thư viện sử dụng và môi trường chạy;
  - mỗi mục có demo trên slide; slide phải có thực nghiệm;
  - giải thích được từng dòng notebook;
  - phần không có code trên slide thì mở notebook trình bày.

## Purpose
- **Làm:**
  - slide khoảng 21 trang, 15–20 phút;
  - 1 notebook chạy trên Kaggle (phần 0–10), có output, có log mỗi lần chạy;
  - bộ câu thử nghiệm + kết quả + phân tích lỗi;
  - tài liệu tham khảo;
  - mỗi người khoảng 5 câu Q&A;
  - **điểm cộng:** mục 2 có phần so sánh LLM với RNN/LSTM, encoder-only (BERT), VLM, **có demo** (BERT vs Qwen; Qwen2.5 vs Qwen2-VL), thêm 1 slide 6b (thiết kế demo: Toàn duyệt 16/11).
- **Không làm:**
  - fine-tune hay huấn luyện model (nên không có epoch, training loss/accuracy, không có trọng số tự train);
  - mã nguồn ngoài notebook;
  - model lớn hơn 3B;
  - đánh giá trên nhiều bộ dữ liệu.

## Backbone
| Model / tokenizer | Nguồn | Dùng ở phần | Model card (có sơ đồ kiến trúc) |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (chính) | `Qwen/Qwen2.5-1.5B-Instruct` | 1, 2, 4, 5, 6, 7, 8, 9, 10 | `models/qwen2.5-1.5b-instruct_v2.md` |
| Qwen2.5-1.5B (base) | `Qwen/Qwen2.5-1.5B` | 5 | `models/qwen2.5-1.5b-base_v2.md` |
| GPT-2 tokenizer | `openai-community/gpt2` | 3 | `models/gpt2-tokenizer_v2.md` |
| mBERT (so sánh, mục 2) | `google-bert/bert-base-multilingual-cased` | 2 | `models/bert-base-multilingual-cased_v1.md` |
| Qwen2-VL-2B-Instruct (so sánh, mục 2) | `Qwen/Qwen2-VL-2B-Instruct` | 2 | `models/qwen2-vl-2b-instruct_v1.md` |

## Input / Output
Bảng đầy đủ: `rules/FOLDER_STRUCTURE_v2.md` §3; tên file từng phần: `../project_description.md` §2.3.
- **Input:**
  - Model, tokenizer: tải từ Hugging Face Hub bằng `from_pretrained` vào bộ nhớ đệm của phiên Kaggle.
  - UIT-VSFC: đọc file Parquet trên Hugging Face bằng `pd.read_parquet`.
  - Bộ câu nhóm soạn và danh sách ID: `data/`.
  - Tham số: cell cấu hình đầu notebook.
- **Output:**
  - Kết quả: `outputs/results/` (CSV/JSON, `env.json`).
  - Hình: `outputs/figures/` (PNG).
  - Log: `outputs/logs/<YYYYMMDD_HHMMSS>_normal/` (smoke: `outputs/_smoke/`).
  - Còn lại: `slides/`, `docs/experiments/`.
- **Code:** `notebook/llm_presentation.ipynb`, phần 0–10, Toàn viết toàn bộ.

## Success criteria
- 10 mục, mỗi mục có minh hoạ đúng cột "Yêu cầu minh họa" của đề, với output thật từ notebook (bảng đối chiếu ở `../project_description.md` §1).
- Slide 2 có thư viện + phiên bản và môi trường, lấy từ `outputs/results/env.json`.
- Thực nghiệm có kiểm soát (mục 7, 10): 500 câu UIT-VSFC (250/250), zero-shot vs 3-shot × 3 bộ, greedy; có accuracy, confusion matrix, tỉ lệ không parse được, McNemar; bảng kết quả + ví dụ lỗi thật.
- Notebook chạy Save & Run All không lỗi; mỗi cell code có markdown trước, comment, phân tích sau output.
- Mỗi lần chạy có thư mục log theo ngày giờ, đủ phần đầu / mỗi mẫu / cuối (`templates/RUN_LOG_TEMPLATE_v2.md`).
- Mọi số trên slide có đường dẫn file kết quả và `run_id` của lần chạy chính thức.
- Tập thuyết trình trong 20 phút, gồm cả thời gian mở notebook.

## Đánh giá  (Toàn duyệt 16/11)
| Phần | Đánh giá cái gì | Đo bằng gì | Thế nào là đạt | Output |
|---|---|---|---|---|
| 0 Setup | Môi trường và model tải đúng | `env.json` đủ phiên bản; số tham số đếm bằng code | Bằng 1,543,714,304 (khớp model card) | `results/env.json`, log phần 0 |
| 1 Tác vụ mẫu | Model làm được 4 tác vụ | Chấm tay theo thang ghi sẵn (đúng yêu cầu / sai) | Có output thật cho cả 4 tác vụ, chấm và ghi lý do | `results/tasks_outputs.csv` |
| 2 Kiến trúc | Causal mask và xác suất token tiếp theo | `assert` mask tam giác dưới; tổng xác suất top-5 ≤ 1 | Assert đạt; heatmap đúng hình tam giác | `results/top5_next_tokens.csv`, `figures/causal_mask.png` |
| 2 So sánh (demo) | BERT nhìn hai chiều vs Qwen chỉ nhìn quá khứ; VLM vs LLM trên câu hỏi về ảnh | Top-5 + xác suất của 2 model; đúng/sai theo đáp án ghi sẵn | Có output thật cho cả 2 demo, chấm đủ | `results/compare_bert_qwen.csv`, `results/compare_vlm.csv` |
| 3 Tokenization | Số token Việt/Anh của 2 tokenizer | Đếm `len(input_ids)` cùng cách cho 2 tokenizer | Có bảng 2 câu × 2 tokenizer | `results/token_counts.csv` |
| 4 Nguyên lý học | Cross-entropy từng token | Loss trung bình tự tính so với `outputs.loss` | Lệch < 1e-3 (`torch.allclose`) | `results/per_token_loss.csv` |
| 5 Base vs instruct | Khác biệt do post-training | Chấm tay theo thang ghi sẵn (làm theo chỉ dẫn hay không) | Cùng prompt, cùng `do_sample=False`; chấm đủ mọi prompt | `results/base_vs_instruct.csv` |
| 6 Sinh văn bản | Ảnh hưởng của cấu hình sinh | Greedy chạy 2 lần cho cùng output; so độ đa dạng 5 cấu hình | Greedy lặp lại y hệt | `results/decoding_samples.csv` |
| 7, 10 Zero/few-shot | Few-shot có hơn zero-shot không | Accuracy, confusion matrix, tỉ lệ không parse được, McNemar | Kết luận theo p-value của McNemar, không theo cảm tính | `results/icl_*.csv`, `results/icl_metrics.json` |
| 8 RAG | RAG có giúp trả lời đúng hơn; có biết từ chối khi tài liệu không có đáp án không | Nhãn Đúng / Sai / "Không có trong tài liệu": tự động kiểu SQuAD, rồi 2 người chấm mù (% đồng ý, kappa); McNemar exact; context recall@3; faithfulness; số câu từ chối đúng (RGB) | Chấm đủ mọi câu, báo đủ các số | `results/rag_results.csv` |
| 9 Hạn chế | Hallucination, thiên lệch, ngữ cảnh dài | Hallucination theo SimpleQA: Correct / Incorrect / Not attempted, 2 người chấm (% đồng ý, kappa); báo % Incorrect, F-score; câu dài theo FActScore | Chấm đủ mọi câu; có ít nhất 1 câu Incorrect kèm cách kiểm chứng | `results/*_results.csv` |

## How-to-do  (Toàn duyệt 16/11)
Từng bước chi tiết: code phần 0–10 ở `plans/detail_plan_notebook_v1.md`; quy trình, smoke run, normal run, duyệt và commit ở `plans/huong_dan_tung_buoc_v1.md`.

1. **Toàn: phần 0 Setup.** Cell cấu hình (`SMOKE`, seed 42, tên model, đường dẫn); tải 2 model + tokenizer + UIT-VSFC; in `model.config`, đếm số tham số; ghi `env.json`, `config.json`, mở `run.log`. Smoke run riêng phần 0 trên Kaggle.
2. **Mỗi người phụ trách mục: viết dàn ý desc mục mình** trong `parts/` (khuôn `PART_TEMPLATE_v2.md`): Purpose, I/O, Backbone, Đánh giá, How to do. Toàn duyệt.
3. **Muội, Phúc: soạn file trong `data/`** cho mục mình (danh sách ở `rules/RULES_v2.md` §4, Toàn duyệt 16/11).
4. **Toàn: code phần 1–10** theo detail plan đã duyệt, mỗi bước một small prompt; xong bước nào chạy kiểm bước đó; mỗi phần ghi `partNN_<tên>.log` và `_records.csv`.
5. **Toàn: smoke run toàn notebook** (`SMOKE = True`), kiểm đủ file kết quả và log trong `outputs/_smoke/`.
6. **Toàn: bàn giao từng phần** cho người phụ trách mục (giải thích từng cell, chỉ log và file kết quả); ghi vào `../progress_status.md`.
7. **Toàn: mục 7, 10** chốt prompt trên split validation; **normal run**: Save & Run All trên Kaggle → `outputs/logs/<ts>_normal/`; xuất HTML dự phòng.
8. **Audit:** đối chiếu kết quả từng phần với bảng Đánh giá và Success criteria ở trên; ghi "Kết quả audit" trong desc mục.
9. Mỗi người điền số thật vào slide mục mình (ghi `run_id`); Muội ghép slide; Thư review.
10. Tập trình bày (mỗi người giải thích code phần mình trình bày), hỏi chéo Q&A, nộp.

## Quyết định (người ghi)
- 2026-11-11: không fine-tune; mọi mục có demo trên model thật (góp ý của cô) — Thư ghi.
- 2026-11-15: repo riêng `LLMs_DL`; code hết trong notebook, không có mã nguồn riêng; cấu trúc thư mục gọn; ưu tiên phần setup — góp ý của Thư, Toàn ghi.
- 2026-11-15: **Toàn viết toàn bộ code notebook (phần 0–10)**; Muội, Phúc làm lý thuyết, slide, thuyết trình mục mình và phải giải thích được code phần đó sau khi Toàn bàn giao — Toàn quyết định.
- 2026-11-16: Toàn duyệt 4 cổng và merge PR; Thư review — Toàn quyết định.
- 2026-11-16: không fine-tune nên log ghi từng lần chạy (từng mẫu) thay cho log huấn luyện theo epoch — Toàn quyết định.
- 2026-11-16: thêm phần so sánh LLM với RNN/LSTM, encoder-only (BERT), VLM, **gộp vào mục 2, phải có demo** — Toàn quyết định. Thiết kế demo (mBERT, Qwen2-VL-2B; +1 slide 6b) Toàn duyệt 16/11, xem `research/noi_dung_muc02_kien_truc_v2.md` §11.3.
- 2026-11-16: phần 0 Setup (lý thuyết + code) do Toàn — Toàn quyết định.
- 2026-11-16: duyệt toàn bộ bản v2 (rules, templates, description, plans, model card); Bonferroni ở mục 10; parse nhận cả nhãn Việt, thử nhiều mẫu prompt trên validation; repo `pptthu/LLMs_DL`, chủ repo Thư; Ngày 1 = 15/11 — Toàn quyết định.
- 2026-11-16: cách chấm mục 8 (RAG) và mục 9 (hallucination) theo các chuẩn SQuAD, RGB, RAGAS, SimpleQA, FActScore; 2 người chấm mù, báo % đồng ý + Cohen's kappa; không dùng LLM làm giám khảo chính — Toàn quyết định. Chi tiết: `research/noi_dung_muc08_*_v2.md` §3.4, `research/noi_dung_muc09_*_v2.md` §5.1b.

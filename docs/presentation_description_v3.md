# Presentation LLMs — Tìm hiểu mô hình ngôn ngữ lớn  (v3)
- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-21
- Trạng thái: **Duyệt: Toàn, 2026-11-16**; cập nhật theo hiện trạng 21/11 (Thư review) | Trạng thái công việc: notebook xong (run chính thức `20261121_010447_normal`), slide đang làm
- Khuôn: `templates/DESC_TEMPLATE_v2.md`. Thay đổi so với v2 (v2 giữ nguyên, không dùng):
  - Toàn trình bày cả 10 mục (đổi phân công 21/11);
  - bỏ phần so sánh mBERT / VLM / LSTM khỏi notebook nộp (chuyển sang `notebook/llm_extension_part02.ipynb`, không nộp, 19/11);
  - thực nghiệm mục 7, 10 chạy toàn bộ 2,999 câu test, 4 ví dụ few-shot, chọn prompt trên validation (19/11);
  - mục 8, 9 dùng UIT-ViQuAD 2.0 thay bộ câu tự soạn (19/11); mục 9 thêm thiên lệch, giới hạn ngữ cảnh, chất lượng theo dữ liệu (20–21/11);
  - bỏ chấm 2 người, kappa, Bonferroni (chỉ 1 phép so sánh, 1 người chấm, ghi là giới hạn).

## Background
- Môn Học sâu (UTH). Đề: "Tìm hiểu mô hình ngôn ngữ lớn — Large Language Models (LLMs)".
- Mục tiêu của cô: giải thích được cách LLM học và sinh văn bản, phân biệt các giai đoạn huấn luyện, nhận biết khả năng và hạn chế của mô hình. Gồm 10 mục, mỗi mục có yêu cầu minh hoạ.
- **Góp ý của cô (11/11):** không cần fine-tune, nhưng mọi mục phải có demo trên model thật kèm phân tích output.
- **Yêu cầu thêm:** slide nêu thư viện và môi trường chạy; mỗi mục có demo trên slide; slide phải có thực nghiệm; giải thích được từng dòng notebook; phần không có code trên slide thì mở notebook trình bày.

## Purpose
- **Làm:**
  - 16 slide, 15–20 phút (gồm cả mở notebook), Toàn trình bày cả bài;
  - 1 notebook chạy trên Kaggle (phần 0–10), có output, có log mỗi lần chạy, có khung ĐẦU VÀO trước mỗi demo;
  - bộ câu thử nghiệm + kết quả + phân tích lỗi;
  - tài liệu tham khảo.
- **Không làm:**
  - fine-tune hay huấn luyện model (không có epoch, training loss/accuracy, không có trọng số tự train);
  - mã nguồn ngoài notebook;
  - model lớn hơn 3B; chạy nhiều seed; dùng LLM khác để chấm.

## Backbone
| Model / tokenizer | Nguồn | Dùng ở phần | Model card |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (chính) | `Qwen/Qwen2.5-1.5B-Instruct` | 1, 2, 3, 4, 6, 7, 8, 9, 10 | `models/qwen2.5-1.5b-instruct_v2.md` |
| Qwen2.5-1.5B (base) | `Qwen/Qwen2.5-1.5B` | 5 | `models/qwen2.5-1.5b-base_v2.md` |
| GPT-2 tokenizer | `openai-community/gpt2` | 3 | `models/gpt2-tokenizer_v2.md` |

## Input / Output
Bảng đầy đủ: `rules/FOLDER_STRUCTURE_v3.md` §3; tên file từng phần: `../project_description.md` §2.3.
- **Input:**
  - Model, tokenizer: Hugging Face Hub (`from_pretrained`).
  - UIT-VSFC, UIT-ViQuAD 2.0, Nemotron-Personas-Vietnam: Parquet trên Hugging Face (`pd.read_parquet`); WinoGender: TSV trên GitHub (`pd.read_csv`).
  - Bộ câu nhóm soạn: `data/tasks_prompts.json`, `data/base_vs_instruct.json`; câu tiền đề sai và khuôn câu thiên lệch tiếng Việt viết trong cell 9.3, 9.4.
  - Tham số: cell 0.1.
- **Output:** `outputs/results/` (CSV/JSON), `outputs/figures/` (PNG), `outputs/logs/<YYYYMMDD_HHMMSS>_normal/` (smoke: `outputs/_smoke/`).
- **Code:** `notebook/llm_presentation.ipynb`, phần 0–10, Toàn viết toàn bộ.

## Success criteria (kết quả 21/11)
| Tiêu chí | Kết quả |
|---|---|
| 10 mục có minh hoạ đúng cột "Yêu cầu minh họa" của đề, output thật | Đạt (bảng đối chiếu: `../project_description.md` §1; kiểm 21/11) |
| Slide 2 có thư viện, phiên bản, môi trường từ `env.json` | Có slide 2; cần đổi run id và thời gian chạy 9.2 phút |
| Thực nghiệm có kiểm soát (mục 7, 10): 2,999 câu, zero-shot vs few-shot, greedy; accuracy, confusion matrix, tỉ lệ không parse được, McNemar; bảng kết quả + ví dụ lỗi thật | Đạt: 89.86% vs 92.30%, McNemar p = 1.5e-08, 0 câu không parse được |
| Notebook Save & Run All không lỗi; markdown, comment, phân tích | Đạt: 140 cell, 0 lỗi (`20261121_010447_normal`) |
| Mỗi lần chạy có log theo ngày giờ | Đạt (`outputs/logs/20261121_010447_normal/`) |
| Mọi số trên slide có `run_id` | Chờ slide hoàn chỉnh |
| Trình bày ≤ 20 phút | Chưa tập bấm giờ |

## Đánh giá (theo cách đã làm)
| Phần | Đánh giá cái gì | Đo bằng gì | Thế nào là đạt | Kết quả `20261121_010447_normal` |
|---|---|---|---|---|
| 0 Setup | Môi trường và model tải đúng | `env.json`; số tham số đếm bằng code | 1,543,714,304 | Đạt |
| 1 Tác vụ mẫu | Model làm được 4 tác vụ | Luật chấm ghi trước (từ khoá, chạy code, chấm tay tóm tắt) | Chấm đủ 4, ghi lý do | 4/4 đạt |
| 2 Kiến trúc | Causal mask, xác suất token tiếp theo | `assert` mask; attention thật phía trên đường chéo = 0; top-5 | Assert đạt | Đạt; P(" đẹp") = 0.788 |
| 3 Tokenization | Số token Việt / Anh, 2 tokenizer | Đếm `input_ids` cùng cách; ghép byte = câu gốc | Có bảng 2 × 2, decode đúng 4/4 | Qwen 8 / GPT-2 22; Anh 7 / 7 |
| 4 Nguyên lý học | Cross-entropy từng token | Loss tự tính so với `outputs.loss` | Lệch < 1e-3 | Lệch 6.3e-08; loss 2.985 |
| 5 Base vs Instruct | Khác biệt do post-training | Luật `follow_rule` ghi trước | Cùng prompt, cùng tham số, chấm đủ | Base 0/4, Instruct 4/4 |
| 6 Sinh văn bản | Ảnh hưởng của cấu hình sinh | Greedy lặp lại; số output khác nhau / 3 lần | Greedy y hệt | Greedy 1, lấy mẫu 3/3 |
| 7, 10 Zero / few-shot | Few-shot có hơn zero-shot | Accuracy, CI 95%, confusion matrix, McNemar exact | Kết luận theo p-value | 0.8986 vs 0.9230, p = 1.5e-08 |
| 8 RAG | RAG có giúp trả lời đúng; có từ chối khi tài liệu không có đáp án | Đúng / Sai / Từ chối (tự động, đọc lại tay); recall@3 | Chấm đủ mọi câu | 0/15 → 10/15; recall@3 0.722; bịa 2/3 câu không có đáp án |
| 9 Hạn chế | Hallucination; thiên lệch; giới hạn ngữ cảnh; chất lượng theo dữ liệu | SimpleQA (Đúng / Sai / Từ chối); Pearson r (luật ghi trước r > 0, p < 0.05); kim trong đống rơm; accuracy theo nhóm câu | Có câu sai kèm kiểm chứng; luật ghi trước | Sai 12/15; bịa 5/6; r = 0.69 (Anh), 0.44 (Việt); needle 6/6, cắt → bịa "1234"; góp ý 0.79 vs 0.96 |

## How-to-do (trạng thái 21/11)
1. ~~Phần 0–10: code, smoke run, normal run~~ — xong, run chính thức `20261121_010447_normal`; nhật ký ở `../progress_status.md`, tóm tắt ở `experiments/experiment_log.md`.
2. Slide ở `slides/infographic/`; số lấy từ `outputs/results/`, hình từ `outputs/figures/`, ghi `run_id`.
3. Lý thuyết, code, kết quả từng mục: `research/mucNN_*_v1.md`.
4. Tập trình bày, bấm giờ từng slide (≤ 20 phút); Thư hỏi Q&A thử.
5. Nộp: slide PDF/PPTX, notebook đã chạy, `outputs/`, `data/`, `docs/references.md`.

## Quyết định (người ghi)
- 2026-11-11: không fine-tune; mọi mục có demo trên model thật (góp ý của cô) — Thư ghi.
- 2026-11-15: repo riêng `LLMs_DL`; code hết trong notebook — góp ý của Thư, Toàn ghi.
- 2026-11-15: Toàn viết toàn bộ code notebook (phần 0–10) — Toàn quyết định.
- 2026-11-16: Toàn duyệt 4 cổng và merge PR; Thư review — Toàn quyết định.
- 2026-11-16: log ghi từng lần chạy (từng mẫu) thay cho log huấn luyện — Toàn quyết định.
- 2026-11-19: bỏ cell so sánh mục 2 khỏi notebook nộp; chạy đủ 2,999 câu test; 4 ví dụ few-shot; chọn prompt trên 200 câu validation; mục 8, 9 dùng UIT-ViQuAD 2.0; bỏ McNemar nhiều bộ, Bonferroni, kappa — Toàn quyết định (theo file chốt khung v4 của Thư).
- 2026-11-20: mở lại demo thiên lệch (WinoGender + Nemotron-Personas-Vietnam, luật kết luận ghi trước), thêm giới hạn ngữ cảnh và chất lượng theo dữ liệu ở mục 9; thêm bảng kẻ vạch, khung ĐẦU VÀO, các hình minh họa — Toàn quyết định.
- 2026-11-21: Toàn trình bày cả 10 mục; xoá file bàn giao / trao đổi giữa thành viên (còn trong lịch sử git) — Toàn quyết định.
- 2026-11-24: bổ sung bộ metric theo danh sách 18 metric của cô (10.3b–10.3d, 1.6, 4.6, 6.5, 6.6) — Toàn quyết định.
- 2026-11-27: giám khảo DeepEval / G-Eval đổi sang API `gpt-6.2-sol` (run `20261127_234527_normal`), giữ tuỳ chọn Qwen2.5-7B local để so sánh — Toàn quyết định.
- 2026-11-28: G-Eval chấm 20 lần / tiêu chí lấy trung bình (giám khảo không trả log-prob; chạy thử `20261128_040436_pilot`), giám khảo gọi song song 16 luồng; run chính thức `20261128_041850_normal`, VRAM 6.6 / 9.5 lấy run `20261127_234527_normal` (run mới đo lệch, Toàn chốt không chạy lại). Slide còn 26 trang: bỏ slide so khớp chữ + DeepEval, DeepEval 5W1H, tổng hợp 14 bộ; slide so sánh 2 giám khảo trình bày lại thành bảng.
- 2026-11-28: tài liệu trên GitHub chỉ giữ research từng mục (`research/mucNN_*_v1.md`), luật, khuôn, model card, sổ thí nghiệm, tài liệu tham khảo; tài liệu cũ, kế hoạch, tài liệu chuẩn bị trình bày giữ ở máy — Toàn quyết định.

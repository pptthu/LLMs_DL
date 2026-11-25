# LLMs_DL — Presentation: Tìm hiểu mô hình ngôn ngữ lớn (LLMs)

Bài thuyết trình nhóm môn Học sâu (UTH). Nhóm: Muội, Phúc, Toàn, Thư. Người trình bày cả 10 mục và viết code notebook: Toàn (đổi phân công 21/11).

> Trạng thái: đã chạy chính thức trên Kaggle (GPU T4), run_id **`20261125_214650_normal`** (bản có thêm cell 1.6 Pass@k, 4.6 top-1/top-5 accuracy, 6.5 distinct-n, 6.6 hiệu quả thực thi, 10.3b Exact Match / token F1 / BLEU / ROUGE, 10.3c DeepEval, 10.3d G-Eval). Mọi số liệu trong `outputs/results/` và trên slide lấy từ lần chạy này; phần 1–10 ra đúng số của run cũ `20261125_082211_normal`, `20261124_180414_normal` và `20261121_010447_normal` (riêng bộ nhớ ở 6.6, 9.5 thấp hơn run 082211 vì đã sửa lỗi chưa giải phóng model base). Người phụ trách README: Muội.

## Nội dung
10 mục theo đề của cô; mỗi mục có demo trên model thật, cùng một notebook. Giải thích lý thuyết, code và kết quả từng mục: `docs/parts/ban_giao_toan_v2.md`; kịch bản thuyết trình và thứ tự ôn: `docs/scripts/kich_ban_toan_v2.md`.

| # | Mục |
|---|---|
| 1 | Giới thiệu: LM, LLM, chatbot |
| 2 | Kiến trúc: Transformer, decoder-only, causal attention |
| 3 | Tokenization |
| 4 | Nguyên lý học: next-token prediction, cross-entropy |
| 5 | Các giai đoạn huấn luyện: pretraining, SFT, RLHF/DPO |
| 6 | Sinh văn bản: greedy, temperature, top-p |
| 7 | Khả năng: in-context learning, zero-shot vs few-shot |
| 8 | Prompting, fine-tuning, RAG |
| 9 | Hạn chế: hallucination, thiên lệch, giới hạn ngữ cảnh |
| 10 | Thực nghiệm có kiểm soát và kết luận |

## Setup

### 1. Môi trường
- **Kaggle Notebook, GPU T4.** Settings → Accelerator → GPU T4; bật **Internet** để tải model.
- Muốn bật GPU và Internet, tài khoản Kaggle phải **xác minh số điện thoại**.
- **Kiểu số:** `torch.float16`. T4 không hỗ trợ bf16 bằng phần cứng.
- Phiên bản thật của Python, thư viện, GPU được in ở **phần 0** của notebook và lưu `outputs/results/env.json`.

### 2. Thư viện
| Thư viện | Dùng làm gì |
|---|---|
| `torch` | Chạy model trên GPU |
| `transformers` (≥ 4.37.0) | Tải model, tokenizer; `generate()` |
| `accelerate` | `device_map="auto"` |
| `pandas`, `pyarrow` | Đọc UIT-VSFC, UIT-ViQuAD 2.0, Nemotron-Personas-Vietnam (Parquet), WinoGender (TSV); lưu CSV |
| `scikit-learn` | Accuracy, confusion matrix, TF-IDF |
| `statsmodels` | McNemar test |
| `scipy` | Tương quan Pearson (mục 9, thiên lệch) |
| `matplotlib` | Vẽ hình |

Kaggle thường cài sẵn. Cài trên máy khác: `pip install -r requirements.txt`.

### 3. Model, tokenizer, dữ liệu: lấy ở đâu, như thế nào
Cả ba model đều công khai, **không cần token Hugging Face**.

| Thứ | Lấy ở đâu | Giấy phép | Cách lấy |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (model chính) | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Apache-2.0 | `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto")` |
| Qwen2.5-1.5B (base, mục 5) | https://huggingface.co/Qwen/Qwen2.5-1.5B | Apache-2.0 | `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", ...)` |
| GPT-2 tokenizer (mục 3) | https://huggingface.co/openai-community/gpt2 | MIT | `AutoTokenizer.from_pretrained("openai-community/gpt2")` |
| UIT-VSFC (mục 7, 10) | https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback | Ghi "unknown" → trích bài báo gốc, không commit | `pd.read_parquet(".../resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet")` |
| UIT-ViQuAD 2.0 (mục 8, 9) | https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0 | Không ghi → trích bài báo gốc, không commit | `pd.read_parquet(".../resolve/main/data/validation-00000-of-00001.parquet")` |
| WinoGender (mục 9, thiên lệch tiếng Anh) | https://github.com/rudinger/winogender-schemas | MIT | `pd.read_csv(".../data/all_sentences.tsv", sep="\t")`, `occupations-stats.tsv` |
| Nemotron-Personas-Vietnam (mục 9, thiên lệch tiếng Việt) | https://huggingface.co/datasets/nvidia/Nemotron-Personas-Vietnam | CC BY 4.0 | `pd.read_parquet(".../data/train-00000-of-00011.parquet", columns=["occupation", "sex"])` (khoảng 200 MB) |

- `from_pretrained` tự tải file về bộ nhớ đệm của Hugging Face; lần sau không tải lại.
- Mỗi model Qwen khoảng 3.1 GB.
- Model card có sơ đồ kiến trúc: `docs/models/*_v2.md`.

## Cách chạy
1. Mở `notebook/llm_presentation.ipynb` trên Kaggle (Upload notebook hoặc import từ GitHub).
2. Bật GPU T4 và Internet.
3. **Chạy thử:** đặt `SMOKE = True` ở cell đầu → Run All. Mỗi phần chạy vài câu, kết quả ghi vào `outputs/_smoke/`.
4. **Chạy chính thức:** `SMOKE = False` → **Save & Run All**. Kết quả ghi vào `outputs/results/` và `outputs/figures/`; log ghi vào `outputs/logs/<YYYYMMDD_HHMMSS>_normal/`.
5. Thời gian chạy: chính thức **41.2 phút** trên Kaggle T4 x2, trong đó DeepEval (10.3c) 15.2 phút và G-Eval (10.3d) 13.7 phút (`outputs/logs/20261125_214650_normal/run.log`, chưa tính thời gian tải model).
6. Notebook cần Internet suốt lần chạy: ngoài model còn đọc dữ liệu từ Hugging Face và GitHub (mục 7–9).

## Cấu trúc thư mục
```text
LLMs_DL/
├── README.md, project_description.md, progress_status.md, AI_usage.md, AGENTS.md, CONTRIBUTING.md
├── requirements.txt, .gitignore, .github/
├── notebook/      # llm_presentation.ipynb (bản nộp); llm_extension_part02.ipynb (mở rộng mục 2, không nộp)
├── data/          # bộ câu nhóm tự soạn (mục 1, 5); câu test, ví dụ few-shot, câu hỏi RAG đã chọn lưu ở outputs/results/
├── outputs/       # results/ (CSV, JSON, env.json), figures/ (PNG), logs/<YYYYMMDD_HHMMSS>_normal/ (log mỗi lần chạy), _smoke/ (không commit)
├── slides/        # PPTX, PDF
└── docs/          # luật, khuôn, kế hoạch, research, model card, sổ thí nghiệm, tài liệu tham khảo
```
Chi tiết: `docs/rules/FOLDER_STRUCTURE_v3.md`. Đọc tài liệu bắt đầu từ `docs/00_README_v3.md`.

## Sản phẩm nộp
| Sản phẩm | Đường dẫn |
|---|---|
| Slide PDF / PPTX | `slides/` |
| Notebook + hướng dẫn chạy | `notebook/llm_presentation.ipynb`, file này |
| Bộ câu thử nghiệm, kết quả, phân tích lỗi | `data/`, `outputs/results/`, `docs/experiments/error_analysis_v2.md` |
| Tài liệu tham khảo | `docs/references.md` |

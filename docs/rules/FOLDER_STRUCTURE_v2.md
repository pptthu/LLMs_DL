# Cấu trúc thư mục bài Presentation LLMs (v2)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-16T01:40:17+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-16T01:40:17+07:00
- Trạng thái: **Duyệt: Toàn, 2026-11-16** (Thư review)
- Thay đổi so với v1:
  - §1 ghi đúng cây thư mục **đang có** trong repo ngày 16/11 (từng file), tách riêng phần **sẽ tạo**.
  - Thêm `outputs/logs/` (log từng lần chạy, đặt tên theo ngày giờ) và quy tắc đặt tên ở §4.
  - Thêm §3 "Input lấy từ đâu, output lưu ở đâu" và §5 "Model lưu ở đâu".
  - v1 giữ nguyên, không dùng.

Ký hiệu: **(có)** = đã có trong repo; **(sẽ tạo)** = tạo khi có file đầu tiên; **(không commit)** = `.gitignore` chặn.

## 1. Cây thư mục hiện tại (2026-11-16)

    LLMs_DL/
    ├── README.md                     (có) Setup + cách chạy: môi trường, thư viện, lấy model/dữ liệu ở đâu
    ├── project_description.md        (có) Define + Design: đề, 10 mục, phạm vi, file vào/ra từng phần
    ├── progress_status.md            (có) Bảng tiến độ, bàn giao code, kiểm chứng, nhật ký
    ├── AI_usage.md                   (có) Khai báo dùng AI, mỗi người một cột
    ├── AGENTS.md                     (có) Luật cho AI agent
    ├── CONTRIBUTING.md               (có) Quy tắc Git: nhánh, commit, PR
    ├── requirements.txt              (có) Thư viện; ghim phiên bản sau lần chạy chính thức
    ├── .gitignore                    (có) Chặn dữ liệu tải về, trọng số, outputs/_smoke/, token
    ├── .github/
    │   ├── PULL_REQUEST_TEMPLATE.md  (có) Mẫu PR
    │   └── ISSUE_TEMPLATE/
    │       └── part_task.md          (có) Mẫu issue cho một mục / một phần notebook
    ├── notebook/
    │   ├── llm_presentation.ipynb    (có) Notebook nộp, phần 0–10; hiện chỉ có 12 ô markdown khung, chưa có code
    │   └── parts/                    (có, trống) Không bắt buộc: Toàn thử riêng một phần
    ├── data/                         (có, trống) Bộ câu nhóm soạn + danh sách ID (§3)
    ├── outputs/
    │   ├── results/                  (có, trống) CSV/JSON kết quả của lần chạy chính thức
    │   ├── figures/                  (có, trống) PNG chèn vào slide
    │   ├── logs/                     (sẽ tạo) Log từng lần chạy normal, mỗi lần một thư mục theo ngày giờ (§4)
    │   └── _smoke/                   (có, không commit) Mọi đầu ra của smoke run, cùng cấu trúc results/ figures/ logs/
    ├── slides/                       (có, trống) llm_presentation_vN.pptx, bản final .pdf
    ├── _to_delete/                   (có, chưa commit) Bản _v2 cũ AI tạo rồi huỷ ngày 16/11; Toàn xoá thư mục này
    └── docs/                         (có) Xem §2

## 2. Cây thư mục `docs/` hiện tại

    docs/
    ├── 00_README_v1.md, 00_README_v2.md           Mục lục; đọc bản số lớn nhất
    ├── presentation_description_v1.md, _v2.md     Desc cả bài
    ├── references.md                              Tài liệu tham khảo (sản phẩm nộp số 4)
    ├── rules/
    │   ├── RULES_v1.md, RULES_v2.md               Quy trình, luật chung, phân vai
    │   └── FOLDER_STRUCTURE_v1.md, _v2.md         File này
    ├── templates/
    │   ├── DESC_TEMPLATE_v1.md, _v2.md            Khuôn desc cả bài
    │   ├── PART_TEMPLATE_v1.md, _v2.md            Khuôn desc từng mục → docs/parts/
    │   ├── RUN_LOG_TEMPLATE_v1.md, _v2.md         Khuôn log mỗi lần chạy (§4)
    │   ├── MODEL_CARD_TEMPLATE_v1.md, _v2.md      Khuôn model card (bắt buộc có sơ đồ kiến trúc)
    │   ├── EXPERIMENT_NOTE_TEMPLATE_v1.md         Khuôn ghi chú thí nghiệm có so sánh
    │   ├── BUG_REPORT_TEMPLATE_v1.md              Khuôn báo lỗi
    │   └── SCRIPT_TEMPLATE_v1.md                  Khuôn kịch bản thuyết trình và Q&A
    ├── plans/
    │   ├── ke_hoach_presentation_llms_v1.md, _v2.md   Kế hoạch nhóm (nội dung, demo, hạn từng người)
    │   ├── khung_presentation_llms_v1.md, _v2.md      Khung slide, notebook, danh sách việc
    │   ├── plan_presentation_llm_v1.md, _v2.md, _v3.md   Plan cấu trúc + plan code (v3 hiện hành)
    │   ├── detail_plan_notebook_v1.md                 Từng bước code phần 0–10
    │   └── huong_dan_tung_buoc_v1.md                  Từng bước: quy trình, chạy Kaggle, duyệt, commit
    ├── research/                                   Nội dung chi tiết từng mục, có nguồn
    │   ├── research_presentation_llm_v1.md         Kiểm chứng kỹ thuật (Qwen, UIT-VSFC, Kaggle, API)
    │   ├── nam_kien_thuc_5w1h_v1.md                Tóm tắt 10 mục theo 5W1H
    │   ├── huong_dan_research_presentation_v1.md   Hướng dẫn tự research
    │   ├── noi_dung_muc01_gioi_thieu_v1.md
    │   ├── noi_dung_muc02_kien_truc_v1.md, _v2.md  (v2 thêm phần so sánh LLM với mô hình khác)
    │   ├── noi_dung_muc03_tokenization_v1.md, _v2.md
    │   ├── noi_dung_muc04_nguyen_ly_hoc_v1.md, _v2.md
    │   ├── noi_dung_muc05_giai_doan_huan_luyen_v1.md
    │   ├── noi_dung_muc06_sinh_van_ban_v1.md, _v2.md
    │   ├── noi_dung_muc08_prompting_finetuning_rag_v1.md, _v2.md
    │   ├── noi_dung_muc09_han_che_danh_gia_v1.md, _v2.md
    │   └── noi_dung_toan_muc07_muc10_v1.md, _v2.md (mục 7, 10)
    ├── models/
    │   ├── qwen2.5-1.5b-instruct_v1.md, _v2.md     Model card (v2 có sơ đồ kiến trúc)
    │   ├── qwen2.5-1.5b-base_v1.md, _v2.md
    │   ├── gpt2-tokenizer_v1.md, _v2.md
    │   ├── bert-base-multilingual-cased_v1.md      Model so sánh mục 2
    │   └── qwen2-vl-2b-instruct_v1.md              Model so sánh mục 2
    ├── experiments/
    │   ├── experiment_log.md                       Sổ ghi mọi lần chạy (tóm tắt, trỏ tới outputs/logs/)
    │   └── error_analysis_v1.md                    (sẽ tạo) Phân tích lỗi mục 10
    ├── parts/                                      (có, trống) Desc từng mục mucNN_<tên>_v1.md, người phụ trách mục viết dàn ý
    ├── scripts/                                    (sẽ tạo) Kịch bản thuyết trình từng người
    ├── qa/                                         (sẽ tạo) Q&A từng người
    └── bugs/                                       (sẽ tạo) Ghi chép lỗi khó

## 3. Input lấy từ đâu, output lưu ở đâu

### 3.1 Input
| Input | Lấy từ đâu | Lấy bằng cách nào | Lưu ở đâu khi chạy | Commit? |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Hugging Face Hub `Qwen/Qwen2.5-1.5B-Instruct` | `AutoModelForCausalLM.from_pretrained(...)` ở phần 0 | Bộ nhớ đệm Hugging Face `~/.cache/huggingface/hub/` của phiên Kaggle | Không |
| Qwen2.5-1.5B (base) | `Qwen/Qwen2.5-1.5B` | `from_pretrained(...)` ở phần 5 | Như trên | Không |
| GPT-2 tokenizer | `openai-community/gpt2` | `AutoTokenizer.from_pretrained(...)` ở phần 3 | Như trên | Không |
| mBERT (so sánh) | `google-bert/bert-base-multilingual-cased` | `pipeline("fill-mask", ...)` ở phần 2 | Như trên | Không |
| Qwen2-VL-2B-Instruct (so sánh) | `Qwen/Qwen2-VL-2B-Instruct` | `Qwen2VLForConditionalGeneration.from_pretrained(...)` ở phần 2 | Như trên | Không |
| UIT-VSFC | Hugging Face `uitnlp/vietnamese_students_feedback`, file Parquet đã chuyển đổi | `pd.read_parquet(<URL>)` ở phần 0 | Trong RAM (DataFrame) | Không |
| ID 500 câu test, 3 bộ ví dụ few-shot | `data/test_500_ids.json`, `data/fewshot_pools.json` | Notebook tạo một lần (seed 42) rồi đọc lại | `data/` | Có |
| Bộ câu nhóm soạn | `data/compare_prompts.json`, `data/vlm_image.png` (ảnh nhóm tự tạo), `data/tasks_prompts.json`, `base_vs_instruct.json`, `rag_questions.csv`, `hallucination.csv`, `bias_prompts.json`, `long_context.json` | Đọc bằng `json` / `pandas` | `data/` | Có |
| Slide môn học (nguồn cho RAG, phần 8) | Máy của nhóm | Upload làm Kaggle Dataset riêng, không public | `tailieu/` (đã chặn trong `.gitignore`) | Không |
| Tham số chạy | Cell cấu hình đầu notebook (`SMOKE`, seed, tên model, số câu, đường dẫn) | Các phần đọc từ đó | Ghi lại vào `config.json` của lần chạy (§4) | — |

### 3.2 Output
| Output | Lưu ở đâu (normal run) | Smoke run | Commit? |
|---|---|---|---|
| Môi trường chạy | `outputs/results/env.json` | `outputs/_smoke/results/env.json` | Có |
| Kết quả từng phần (CSV/JSON) | `outputs/results/<tên>.csv` / `.json` (tên ở `project_description.md` §2.3) | `outputs/_smoke/results/` | Có |
| Hình | `outputs/figures/<tên>.png` | `outputs/_smoke/figures/` | Có |
| Log từng lần chạy | `outputs/logs/<YYYYMMDD_HHMMSS>_normal/` (§4) | `outputs/_smoke/logs/<YYYYMMDD_HHMMSS>_smoke/` | Có (log normal); không (smoke) |
| Notebook đã chạy | `notebook/llm_presentation.ipynb` (bản Save & Run All, có output) | — | Có |
| Bản HTML dự phòng | `slides/llm_presentation_run.html` | — | Có |
| Tóm tắt lần chạy | 1 mục trong `docs/experiments/experiment_log.md` | Có, ghi rõ "smoke" | Có |
| Phân tích lỗi mục 10 | `docs/experiments/error_analysis_v1.md` | — | Có |
| Slide | `slides/` | — | Có |

## 4. Log từng lần chạy: đặt tên và nội dung

- Mỗi lần chạy (một lần Run All, hoặc chạy riêng một phần) tạo **một thư mục** đặt tên theo thời điểm bắt đầu, giờ Việt Nam: `<YYYYMMDD_HHMMSS>_<smoke|normal>`. Ví dụ: `outputs/logs/20261118_093000_normal/`.
- Trong thư mục đó:

| File | Nội dung |
|---|---|
| `run.log` | Log đọc bằng mắt của cả lần chạy: đầu (cấu hình, môi trường, model, kiến trúc), mỗi phần một đoạn, cuối (thời gian, file kết quả, lỗi) |
| `config.json` | Toàn bộ tham số của lần chạy: SMOKE, seed, tên model + revision, dtype, tham số sinh từng phần, số câu, đường dẫn dữ liệu |
| `env.json` | Python, `torch`, `transformers`, `pandas`, `scikit-learn`, `statsmodels`; GPU, CUDA, VRAM |
| `partNN_<tên>.log` | Log riêng của phần NN: tham số, mỗi mẫu một dòng, tổng kết |
| `partNN_<tên>_records.csv` | Cùng các dòng "mỗi mẫu" trong `.log`, dạng bảng để tính lại |

- Khuôn chi tiết: `templates/RUN_LOG_TEMPLATE_v2.md`.
- **Không có dòng epoch, training loss/accuracy** vì bài không fine-tune (góp ý của cô ngày 11/11). Log ghi **từng mẫu** thay cho từng epoch; loss có ở phần 4 (cross-entropy từng token), accuracy có ở phần 7, 10.

## 5. Model lưu ở đâu

- Bài **không train**: không có trọng số do nhóm tạo ra, nên không có `outputs/models/`.
- Model dùng là bản có sẵn trên Hugging Face, tải về bộ nhớ đệm khi chạy (§3.1), không commit.
- Thông tin model (nguồn, kiến trúc có sơ đồ, số tham số, cách lấy, cách dùng) ghi ở `docs/models/<tên>_v2.md`.
- Nếu sau này cô yêu cầu fine-tune: trọng số lưu `outputs/models/<YYYYMMDD_HHMMSS>_<tên>_best.*` (không commit), log thêm dòng mỗi epoch (`epoch, train_loss, train_acc, val_loss, val_acc, lr, time_s`) theo `LOG_TEMPLATE_v1.md` của Lab 1. **[cần chốt nếu xảy ra]**

## 6. Sản phẩm nộp (theo đề) → đường dẫn

| Sản phẩm | Đường dẫn |
|---|---|
| (1) Slide PDF hoặc PPTX | `slides/` |
| (2) Notebook + hướng dẫn chạy | `notebook/llm_presentation.ipynb`, `README.md`, `requirements.txt` |
| (3) Bộ câu thử nghiệm, kết quả, phân tích lỗi | `data/`, `outputs/results/`, `outputs/logs/`, `docs/experiments/error_analysis_v1.md` |
| (4) Tài liệu tham khảo | `docs/references.md` (và slide cuối) |

## 7. How to do: từng bước

**Thêm một file tài liệu mới**
1. Xác định loại file theo §2 (luật → `rules/`, khuôn → `templates/`, kế hoạch → `plans/`, desc mục → `parts/`, nội dung → `research/`, model card → `models/`).
2. Đặt tên tiếng Việt không dấu hoặc tiếng Anh, chữ thường, nối bằng `_`, có hậu tố `_v1` (ví dụ `docs/parts/muc03_tokenization_v1.md`).
3. Header có: tác giả, ngày tạo đủ giờ phút giây (+07:00), trạng thái.
4. Thư mục chưa có (`scripts/`, `qa/`, `bugs/`) thì tạo khi có file đầu tiên.
5. Thêm một dòng vào mục lục `00_README_v2.md` và cây §2 của file này.

**Sửa một file đã có version**
1. Copy `<tên>_vN.md` thành `<tên>_vN+1.md`; không sửa bản cũ.
2. Ở header bản mới ghi "Thay đổi so với vN" và "Cập nhật: <tên>, <thời điểm>".
3. Sửa mọi chỗ trỏ tới bản cũ (`grep -rn "<tên>_vN" .`) sang bản mới, trừ dòng nhật ký và dòng "Dựa trên".
4. Cập nhật `00_README_v2.md` (hoặc bản mục lục mới nhất).

**Thêm một file dữ liệu vào `data/`**
1. Người phụ trách mục soạn file (CSV/JSON), tên theo `project_description.md` §2.3.
2. Mỗi câu có cột nguồn / đáp án nếu dùng để chấm.
3. Không chép dữ liệu có bản quyền hoặc dữ liệu tải về (UIT-VSFC chỉ lưu ID).
4. Commit với type `data`, ví dụ `data(part09): add 20 hallucination questions with sources`.

**Lưu kết quả và log khi chạy** (code làm tự động, người kiểm)
1. Cell 0.1 tạo `RUN_ID = <YYYYMMDD_HHMMSS>_<smoke|normal>` và `OUT_DIR` (`outputs/` hoặc `outputs/_smoke/`).
2. Mỗi phần ghi kết quả vào `OUT_DIR/results/`, hình vào `OUT_DIR/figures/`, log vào `OUT_DIR/logs/RUN_ID/`.
3. Chạy xong, mở thư mục log kiểm đủ file theo §4.
4. Normal run trên Kaggle: tải về và chép vào đúng thư mục trong repo (`plans/huong_dan_tung_buoc_v1.md` §B2).

**Model**
1. Không lưu trọng số vào repo; `from_pretrained` tự tải vào bộ nhớ đệm.
2. Thêm model mới: viết model card `docs/models/<tên>_v1.md` theo `templates/MODEL_CARD_TEMPLATE_v2.md` (có sơ đồ), thêm vào bảng §3.1.

## 8. Khác với Lab 1 và vì sao

| Lab 1 | Bài presentation | Vì sao |
|---|---|---|
| 8 folder module, mỗi folder 1 script | Không có mã nguồn riêng; code trong notebook | Cô hỏi setup hay code phần nào thì chỉ ngay cell đó |
| `tests/` với pytest | `assert` trong cell ngay sau hàm | Không có file `.py` để test |
| `outputs/models/` + log mỗi epoch | Không có `models/`; log mỗi mẫu trong `outputs/logs/` | Không train |
| `data/raw/`, `data/splits/` | `data/` phẳng | Dữ liệu gốc tải lúc chạy; chỉ lưu file nhỏ nhóm soạn |
| Nằm trong repo Lab-DL | Repo riêng `LLMs_DL` | Bài presentation có GitHub riêng |

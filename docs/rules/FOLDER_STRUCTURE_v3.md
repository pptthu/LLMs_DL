# Cấu trúc thư mục bài Presentation LLMs (v3)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-16T01:40:17+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-21
- Trạng thái: **Duyệt: Toàn, 2026-11-16** (Thư review)
- Thay đổi so với v2 (v2 giữ nguyên, không dùng): §1, §2 ghi đúng cây thư mục ngày 21/11; §3 bỏ dữ liệu không còn dùng (ID 500 câu, mBERT, VLM, bộ câu mục 8–9 tự soạn), thêm UIT-ViQuAD 2.0, WinoGender, Nemotron-Personas-Vietnam; đánh dấu file lịch sử.

Ký hiệu: **(có)** = đã có trong repo; **(sẽ tạo)** = tạo khi có file đầu tiên; **(không commit)** = `.gitignore` chặn.

## 1. Cây thư mục hiện tại (2026-11-21)

    LLMs_DL/
    ├── README.md                     (có) Setup + cách chạy: môi trường, thư viện, lấy model/dữ liệu ở đâu
    ├── project_description.md        (có) Define + Design: đề, 10 mục → phần notebook → slide, file vào/ra từng phần
    ├── progress_status.md            (có) Bảng tiến độ, kiểm chứng, nhật ký
    ├── AI_usage.md                   (có) Khai báo dùng AI, mỗi người một cột
    ├── AGENTS.md                     (có) Luật cho AI agent
    ├── CONTRIBUTING.md               (có) Quy tắc Git: nhánh, commit, PR
    ├── requirements.txt              (có) Thư viện (phiên bản thật ở outputs/results/env.json)
    ├── .gitignore                    (có) Chặn dữ liệu tải về, trọng số, outputs/_smoke/, *.zip, token
    ├── .github/                      (có) Mẫu PR, mẫu issue
    ├── notebook/
    │   ├── llm_presentation.ipynb    (có) Notebook nộp, phần 0–10, 140 cell, đã chạy (run 20261121_010447_normal)
    │   ├── llm_extension_part02.ipynb (có) Phần so sánh mBERT / VLM / LSTM của mục 2, KHÔNG nộp
    │   └── parts/                    (có, trống)
    ├── data/                         (có) tasks_prompts.json (mục 1), base_vs_instruct.json (mục 5)
    ├── outputs/
    │   ├── results/                  (có) 30 file CSV/JSON của run chính thức
    │   ├── figures/                  (có) 13 hình PNG
    │   ├── logs/                     (có) Log mỗi lần chạy normal, một thư mục theo ngày giờ (§4)
    │   └── _smoke/                   (có, không commit) Đầu ra của smoke run
    ├── slides/                       (có) Slide 1 (jpg), slide 2 (pdf), slide 9, 12, 13 (html, pdf) — bản cũ; `infographic/` bộ đủ 16 slide: `llm_slides.html` (nguồn), `llm_slides.pdf`, `png/slide_01..16.png`
    └── docs/                         (có) Xem §2

## 2. Cây thư mục `docs/` hiện tại

Bản **hiện hành** in đậm; còn lại là **lịch sử** (giữ nguyên, không sửa, không dùng làm nguồn số liệu).

    docs/
    ├── 00_README_v3.md                             **Mục lục hiện hành** (v1, v2: lịch sử)
    ├── presentation_description_v3.md              **Desc cả bài** (v1, v2: lịch sử)
    ├── giai_thich_du_an_v4.md                      **Giải thích lý thuyết, code, output từng cell, cách trình bày** (v1–v3: lịch sử)
    ├── references.md                               **Tài liệu tham khảo** (sản phẩm nộp số 4)
    ├── rules/
    │   ├── RULES_v3.md                             **Quy trình, luật chung, phân vai** (v1, v2: lịch sử)
    │   └── FOLDER_STRUCTURE_v3.md                  **File này** (v1, v2: lịch sử)
    ├── templates/                                  Khuôn; dùng bản số lớn nhất (RUN_LOG_TEMPLATE_v3)
    ├── parts/
    │   ├── ban_giao_toan_v2.md                     **Tóm gọn lý thuyết + code + kết quả 10 mục** (v1: lịch sử)
    │   └── muc01–muc05_*.md                        Desc từng mục viết 16–17/11: lịch sử (cell và số đã đổi; xem giai_thich_du_an_v4.md)
    ├── scripts/
    │   ├── kich_ban_toan_v2.md                     **Kịch bản 16 slide, giải thích hình, thứ tự ôn** (v1: lịch sử)
    ├── plans/
    │   ├── yeu_cau_cua_co_v1.md                    **Chỉ yêu cầu của cô**
    │   ├── khung_presentation_llms_v5.md           **Khung 16 slide hiện hành** (v1–v4: lịch sử)
    │   └── (còn lại)                               Kế hoạch, plan, hướng dẫn, prompt làm slide: lịch sử
    ├── research/                                   Nội dung research từng mục 14–17/11: lịch sử, tham khảo lý thuyết
    ├── models/                                     **Model card *_v2** (Qwen Instruct, Qwen base, GPT-2); mBERT, Qwen2-VL: chỉ dùng ở notebook mở rộng
    ├── experiments/
    │   ├── experiment_log.md                       **Sổ ghi mọi lần chạy**
    │   └── error_analysis_v2.md                    **Phân tích lỗi mục 10** (v1: lịch sử, cùng số)
    └── drafts/                                     (không commit) Bản nháp cá nhân

## 3. Input lấy từ đâu, output lưu ở đâu

### 3.1 Input
| Input | Lấy từ đâu | Lấy bằng cách nào | Lưu ở đâu khi chạy | Commit? |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Hugging Face Hub `Qwen/Qwen2.5-1.5B-Instruct` | `AutoModelForCausalLM.from_pretrained(...)` ở phần 0 | Bộ nhớ đệm Hugging Face `~/.cache/huggingface/hub/` của phiên Kaggle | Không |
| Qwen2.5-1.5B (base) | `Qwen/Qwen2.5-1.5B` | `from_pretrained(...)` ở cell 5.1 | Như trên | Không |
| GPT-2 tokenizer | `openai-community/gpt2` | `AutoTokenizer.from_pretrained(...)` ở phần 3 | Như trên | Không |
| UIT-VSFC | Hugging Face `uitnlp/vietnamese_students_feedback`, Parquet đã chuyển đổi | `pd.read_parquet(<URL>)` ở cell 0.7 | Trong RAM | Không |
| UIT-ViQuAD 2.0 (validation) | Hugging Face `taidng/UIT-ViQuAD2.0` | `pd.read_parquet(<URL>)` ở cell 8.1 | Trong RAM | Không |
| WinoGender | GitHub `rudinger/winogender-schemas`: `all_sentences.tsv`, `occupations-stats.tsv` | `pd.read_csv(<URL>, sep="\t")` ở cell 9.4 | Trong RAM | Không |
| Nemotron-Personas-Vietnam (2 cột, shard 0) | Hugging Face `nvidia/Nemotron-Personas-Vietnam` | `pd.read_parquet(<URL>, columns=["occupation", "sex"])` ở cell 9.4 | Trong RAM (~200 MB tải về) | Không |
| Bộ câu nhóm soạn | `data/tasks_prompts.json` (mục 1), `data/base_vs_instruct.json` (mục 5); câu tiền đề sai, khuôn câu thiên lệch tiếng Việt viết trong cell 9.3, 9.4 | `json` / trong cell | `data/`, notebook | Có |
| Câu test, ví dụ few-shot, câu hỏi RAG đã chọn | Notebook chọn bằng seed 42 rồi lưu | Ghi `outputs/results/vsfc_test_ids.csv`, `fewshot_examples.json`, `rag_questions_used.csv` | `outputs/results/` | Có |
| Tham số chạy | Cell cấu hình đầu notebook (`SMOKE`, seed, tên model, số câu, đường dẫn) | Các phần đọc từ đó | Ghi lại vào `config.json` của lần chạy (§4) | — |

### 3.2 Output
| Output | Lưu ở đâu (normal run) | Smoke run | Commit? |
|---|---|---|---|
| Môi trường chạy | `outputs/results/env.json` | `outputs/_smoke/results/env.json` | Có |
| Kết quả từng phần (CSV/JSON) | `outputs/results/<tên>.csv` / `.json` (tên ở `project_description.md` §2.3) | `outputs/_smoke/results/` | Có |
| Hình | `outputs/figures/<tên>.png` | `outputs/_smoke/figures/` | Có |
| Log từng lần chạy | `outputs/logs/<YYYYMMDD_HHMMSS>_normal/` (§4) | `outputs/_smoke/logs/<YYYYMMDD_HHMMSS>_smoke/` | Có (log normal); không (smoke) |
| Notebook đã chạy | `notebook/llm_presentation.ipynb` (bản Save & Run All, có output) | — | Có |
| Tóm tắt lần chạy | 1 mục trong `docs/experiments/experiment_log.md` | Có, ghi rõ "smoke" | Có |
| Phân tích lỗi mục 10 | `docs/experiments/error_analysis_v2.md` | — | Có |
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

- Khuôn chi tiết: `templates/RUN_LOG_TEMPLATE_v3.md`.
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
| (3) Bộ câu thử nghiệm, kết quả, phân tích lỗi | `data/`, `outputs/results/`, `outputs/logs/`, `docs/experiments/error_analysis_v2.md` |
| (4) Tài liệu tham khảo | `docs/references.md` (và slide cuối) |

## 7. How to do: từng bước

**Thêm một file tài liệu mới**
1. Xác định loại file theo §2 (luật → `rules/`, khuôn → `templates/`, kế hoạch → `plans/`, desc mục → `parts/`, nội dung → `research/`, model card → `models/`).
2. Đặt tên tiếng Việt không dấu hoặc tiếng Anh, chữ thường, nối bằng `_`, có hậu tố `_v1` (ví dụ `docs/parts/muc03_tokenization_v1.md`).
3. Header có: tác giả, ngày tạo đủ giờ phút giây (+07:00), trạng thái.
4. Thư mục chưa có (`scripts/`, `qa/`, `bugs/`) thì tạo khi có file đầu tiên.
5. Thêm một dòng vào mục lục `00_README_v3.md` và cây §2 của file này.

**Sửa một file đã có version**
1. Copy `<tên>_vN.md` thành `<tên>_vN+1.md`; không sửa bản cũ.
2. Ở header bản mới ghi "Thay đổi so với vN" và "Cập nhật: <tên>, <thời điểm>".
3. Sửa mọi chỗ trỏ tới bản cũ (`grep -rn "<tên>_vN" .`) sang bản mới, trừ dòng nhật ký và dòng "Dựa trên".
4. Cập nhật `00_README_v3.md` (hoặc bản mục lục mới nhất).

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

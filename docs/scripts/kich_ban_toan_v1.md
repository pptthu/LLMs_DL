# Kịch bản thuyết trình — Toàn (slide 2, 9, 12, 13; notebook phần 0, 7, 10) (v1)

- **Created**: 2026-11-20 (Toàn, AI hỗ trợ)
- Khuôn: `docs/templates/SCRIPT_TEMPLATE_v1.md`. Nguồn nội dung: `slides/slide_09_12_13_Toan.html`, `slides/Mô hình · Môi trường · Thư viện.pdf`, `docs/parts/ban_giao_toan_v1.md`, notebook `notebook/llm_presentation.ipynb`.
- Mọi số lấy từ lần chạy chính thức **`20261119_150356_normal`** (Kaggle, GPU T4, 7.4 phút). Đã đối chiếu lại với `outputs/results/` ngày 20/11: accuracy zero-shot 0.8986, few-shot 0.9230; 4 câu ví dụ slide 9 và bảng `prompt_selection.csv` khớp slide.
- Chữ trong ngoặc vuông [ ] là việc làm trên màn hình, không đọc. Chữ trong ngoặc kép là lời nói.
- Thời gian theo `khung_presentation_llms_v4.md`: slide 2 = 1.0 phút, slide 9 = 1.5, slide 12 = 2.0, slide 13 = 1.5 → **tổng 6.0 phút, đã gồm lúc mở notebook**.

---

## Mở sẵn trước khi thuyết trình

| Slide | Mục | Notebook (phần, cell) | File / hình cần mở | Thời gian |
|---|---|---|---|---|
| 2 | Thư viện, môi trường | Phần 0: cell 0.3 (môi trường), 0.4 (tải model), 0.5 (đếm tham số) | `outputs/results/env.json` | 1.0 |
| 9 | 7. Khả năng của LLM | Phần 7: cell 7.2 (`build_prompt`, `parse_label`) | `outputs/results/fewshot_examples.json` | 1.5 |
| 12 | 10. Thiết lập & kết quả | Phần 7: cell 7.3 (`classify_batch`); phần 10: cell 10.2 (McNemar) | `outputs/results/icl_metrics.json` | 2.0 |
| 13 | 10. Phân tích lỗi & tổng hợp | Phần 10: cell 10.3 (nhóm lỗi), 10.4 (bảng tổng hợp) | `outputs/figures/confusion_matrix_icl.png`, `outputs/results/summary_table.csv` | 1.5 |

Chuẩn bị:
- Notebook mở bản **đã chạy** (có output của run `20261119_150356_normal`), **không chạy lại** trong lúc thuyết trình (tải model mất vài phút).
- Đặt sẵn bookmark/tab ở 4 vị trí: cell 0.4, cell 7.2, cell 7.3, cell 10.2. Thu nhỏ notebook về cỡ chữ đọc được từ cuối phòng (zoom 125–150%).
- Thứ tự mỗi slide: nói ý chính trên slide → [mở notebook đúng cell] → chỉ code và output → nói 1 câu phân tích → quay lại slide.

---

## Slide 2 — Mô hình · Môi trường · Thư viện (1.0 phút)

[Chiếu slide 2. Chỉ tay lần lượt 3 cột: MODEL → MÔI TRƯỜNG → THƯ VIỆN.]

"Trước khi vào nội dung, em giới thiệu nhanh model, môi trường và thư viện mà cả nhóm dùng cho mọi demo.

Model chính là **Qwen2.5-1.5B-Instruct** trên Hugging Face, giấy phép Apache 2.0. Đây là Transformer **decoder-only**, 28 block, hidden size 1,536, dùng GQA với 12 head Q và 2 head K-V. Tổng số tham số là **1,543,714,304** — con số này nhóm **đếm bằng code**, không chép từ model card. Thanh màu bên dưới cho thấy khoảng **75% tham số nằm ở FFN**, embedding khoảng 233 triệu.

Môi trường: Kaggle Notebook, GPU Tesla T4, float16, seed 42. Nhóm **chỉ suy luận, không fine-tune**, và khi sinh để đánh giá thì luôn dùng **greedy**.

Thư viện chính: torch để tính trên GPU, transformers để tải model và gọi `generate`, statsmodels cho kiểm định McNemar ở phần 10. Chạy toàn bộ notebook mất **7.4 phút**."

[Mở notebook → cell 0.4, chỉ 3 dòng `from_pretrained` / `torch_dtype` / `device_map`.]

"Đây là dòng tải model: `torch_dtype=float16` vì T4 không hỗ trợ bf16 bằng phần cứng; `device_map={"": 0}` đưa cả model lên GPU 0. Bên dưới em in `generation_config`: bản Instruct **mặc định `do_sample=True`** — tức là lấy mẫu ngẫu nhiên — nên mọi lần gọi `generate` trong notebook đều truyền tham số sinh tường minh."

[Cuộn xuống cell 0.5, chỉ dòng `assert ARCH["n_params"] == 1_543_714_304`.]

"Còn đây là dòng đếm tham số, có `assert` để chắc chắn khớp số trên slide. Phiên bản thư viện trên slide cũng được đọc bằng code ở cell 0.3 và lưu vào `env.json`, không ghi theo trí nhớ."

[Quay về slide → chuyển cho người trình bày mục 1.]

> Nếu bị dồn thời gian: bỏ đoạn mở cell 0.5, chỉ giữ cell 0.4.

---

## Slide 9 — Mục 7: Khả năng của LLM — zero-shot vs few-shot (1.5 phút)

[Chiếu slide 9. Chỉ 3 thẻ trên cùng.]

"Mục 7 nói về khả năng của LLM. Em nêu ba khả năng.

Thứ nhất, **in-context learning**: model học từ ví dụ nằm ngay trong prompt, **không đổi trọng số nào**.
Thứ hai, **làm theo chỉ dẫn**: ta viết 'Chỉ trả lời: tích cực hoặc tiêu cực' thì model làm đúng như vậy — có được là nhờ giai đoạn SFT ở mục 5.
Thứ ba, **một model làm nhiều tác vụ**: ở đây phân loại cảm xúc cũng chỉ là bài toán sinh ra chữ 'tích cực' hay 'tiêu cực'.

Để minh hoạ, nhóm so sánh zero-shot và few-shot trên **cùng một tác vụ**: phân loại cảm xúc câu phản hồi sinh viên trong bộ UIT-VSFC."

[Chỉ 2 khung prompt cạnh nhau.]

"Bên trái là zero-shot: chỉ có chỉ dẫn và câu cần phân loại. Bên phải là few-shot: cùng chỉ dẫn đó, thêm **4 ví dụ có nhãn lấy từ tập train**, 2 tích cực, 2 tiêu cực, xếp xen kẽ. Nhóm **chỉ đổi đúng một biến** là có ví dụ hay không: cùng model, cùng mẫu prompt, cùng greedy, cùng 2,999 câu test."

[Mở notebook → cell 7.2, chỉ hàm `build_prompt`.]

"Trong code, zero-shot và few-shot dùng **chung một hàm `build_prompt`**. Truyền `examples=[]` là zero-shot, truyền 4 ví dụ là few-shot — nên chắc chắn hai prompt chỉ khác phần ví dụ. Hàm `parse_label` bên dưới đọc nhãn từ output: tìm chữ 'tích cực' hoặc 'tiêu cực'; có cả hai hoặc không có thì tính là **không parse được, và tính là sai**."

[Quay về slide, chỉ bảng output thật.]

"Đây là output thật. Hai câu đầu là **câu góp ý** — 'cần giảng chậm và kỹ', 'thầy giảng nhanh' — bộ dữ liệu gán tiêu cực; zero-shot đoán tích cực, few-shot đoán đúng. Nhưng few-shot không phải lúc nào cũng tốt hơn: câu 'giảng kỹ' là khen, few-shot lại đoán tiêu cực. Tổng trên 2,999 câu, accuracy tăng từ **89.86% lên 92.30%**. Có ý nghĩa thống kê hay không, em trình bày ở slide 12."

[Chuyển cho người trình bày mục 8.]

> Lưu ý khi nói: few-shot output dạng "Cảm xúc: tiêu cực" (model chép lại tiền tố trong ví dụ). `parse_label` vẫn đọc được vì nó tìm từ nhãn trong chuỗi.

---

## Slide 12 — Mục 10: Thử nghiệm có kiểm soát — thiết lập & kết quả (2.0 phút)

[Chiếu slide 12. Đọc câu hỏi ở góc phải.]

"Mục 10 là thử nghiệm có kiểm soát. Câu hỏi: **thêm 4 ví dụ vào prompt có làm accuracy khác zero-shot một cách có ý nghĩa thống kê không?**"

[Chỉ dải ① THIẾT LẬP, đi từ trái sang phải.]

"Thiết lập gồm 5 bước.
Model là Qwen2.5-1.5B-Instruct, float16, trên T4.
Dữ liệu là toàn bộ tập test UIT-VSFC sau khi bỏ nhãn neutral: **2,999 câu**, 1,590 tích cực và 1,409 tiêu cực — gần cân bằng.
Mẫu prompt được **chọn trên 200 câu validation**, trong 3 mẫu, **trước khi nhìn tập test**, để kết quả test không bị chọn theo đáp án. Mẫu thắng là `vi_nhan_vi`, accuracy trung bình 0.875.
Biến thay đổi duy nhất: 0 ví dụ hay 4 ví dụ, ví dụ lấy từ train và có `assert` kiểm không trùng câu test.
Cuối cùng là sinh greedy, tối đa 10 token, seed 42; câu không parse được tính là sai."

[Mở notebook → cell 7.3, chỉ 3 chỗ: `padding_side = "left"`, dòng `model.generate(... do_sample=False ...)`, dòng `assert classify_batch(probe) == classify_batch(probe)`.]

"Để chạy hết 2,999 câu trong khoảng 5 phút, em sinh **16 câu một lô**. Prompt dài ngắn khác nhau nên phải đệm cho bằng nhau, và đệm **bên trái** để model sinh tiếp ngay sau token cuối của mỗi prompt. `do_sample=False` là greedy; dòng `assert` cuối chạy cùng một lô hai lần và kiểm ra **y hệt nhau**."

[Quay về slide, chỉ ② KẾT QUẢ.]

"Kết quả: zero-shot **89.86%**, khoảng tin cậy 95% là ±1.08 điểm; few-shot **92.30%**, ±0.95. Tăng 2.44 điểm phần trăm. Cả hai cách đều **không có câu nào không parse được**. Lưu ý trục bắt đầu từ 85%."

[Chỉ ③ McNEMAR.]

"Nhưng hai khoảng tin cậy này chưa đủ để kết luận, vì hai cách chạy trên **cùng một bộ câu**. Nên em dùng **kiểm định McNemar**: chỉ những câu mà hai cách cho kết quả **khác nhau** mới nói lên cách nào tốt hơn.
Ô b = 47 câu chỉ zero-shot đúng; ô c = 120 câu chỉ few-shot đúng. Giả thuyết H0 là hai cách có tỉ lệ sai như nhau, tức kỳ vọng b xấp xỉ c. Thực tế c gấp hơn 2.5 lần b. Kiểm định nhị thức chính xác trên 167 câu này cho **p = 1.5 nhân 10 mũ trừ 8**, nhỏ hơn rất nhiều so với 0.05 → few-shot tốt hơn **có ý nghĩa thống kê**."

[Mở notebook → cell 10.2, chỉ dòng `mcnemar(table, exact=True)`.]

"Trong code chỉ là một lệnh của statsmodels, `exact=True` để dùng phân phối nhị thức chính xác thay vì xấp xỉ chi bình phương."

[Quay về slide, chỉ dòng Giới hạn.]

"Giới hạn: chỉ 1 model 1.5B, 1 bộ dữ liệu, 1 tác vụ, và **1 bộ ví dụ few-shot**. Theo Zhao và cộng sự năm 2021, chỉ đổi ví dụ hoặc thứ tự ví dụ đã có thể làm accuracy thay đổi lớn. Mỗi cấu hình chạy 1 lần. Vì vậy nhóm **không suy rộng** kết luận này cho 'LLM nói chung'."

> Nếu bị dồn thời gian: bỏ đoạn mở cell 10.2 (công thức đã nằm trên slide).

---

## Slide 13 — Mục 10: Phân tích lỗi & tổng hợp kết quả (1.5 phút)

[Chiếu slide 13. Chỉ 2 confusion matrix.]

"Few-shot cải thiện ở đâu? Nhìn confusion matrix: lỗi **tiêu cực bị đoán tích cực** giảm từ **248 xuống 145**. Đổi lại, lỗi ngược — tích cực bị đoán tiêu cực — tăng nhẹ từ 56 lên 86. Đây chính là câu 'giảng kỹ' ở slide 9."

[Chỉ bảng nhóm lỗi.]

"Để hiểu vì sao, em lấy ngẫu nhiên **30 câu few-shot làm sai**, seed 42, rồi gán nhóm lỗi. 5 nhóm lỗi được **ghi sẵn trước khi đọc output**, để tránh chấm theo cảm tính.
Nhiều nhất là nhóm **(d) ý ngầm — 20 trên 30 câu**, ví dụ 'giảm thời lượng lý thuyết, tăng thời lượng thực hành' — nhãn gốc tiêu cực, model đoán tích cực.
Nhóm (c) vừa khen vừa chê 4 câu, như 'thầy dạy hơi khó hiểu nhưng rất kỹ'. Nhóm (e) model sai rõ ràng 4 câu, như 'thầy rất dễ tính' mà đoán tiêu cực. Nhóm (b) nhãn gốc có vẻ sai 2 câu — đây là bằng chứng cho ý 'phụ thuộc dữ liệu' ở mục 9. Không câu nào thuộc nhóm không parse được."

[Chỉ khung 💡 Vì sao few-shot giúp?]

"Đếm trên **toàn bộ** câu sai, không chỉ 30 câu: trong các câu tiêu cực bị đoán tích cực, câu góp ý — có chữ 'nên', 'cần', 'mong', 'thêm', 'hơn' — chiếm **76% ở zero-shot và 81% ở few-shot**.
Giải thích: UIT-VSFC gán **tiêu cực cho câu góp ý**, vì góp ý ngầm chỉ ra điều chưa tốt; còn model đọc giọng lịch sự nên đoán tích cực. Hai ví dụ tiêu cực trong prompt few-shot đều là câu góp ý, nên model **thấy được quy ước gán nhãn của bộ dữ liệu**. Few-shot không dạy kiến thức mới — trọng số không đổi — mà chỉ cho model thấy quy ước."

[Mở notebook → cell 10.3, chỉ `MANUAL_ERROR_GROUPS` và dòng `SUGGEST_RE`.]

"Đây là nhóm lỗi em gán tay cho từng câu theo `row_id`, kèm lý do ngắn; và đây là biểu thức đếm câu góp ý trên toàn bộ câu sai."

[Quay về slide, chỉ bảng Tổng hợp.]

"Cuối cùng, bảng tổng hợp kết quả mọi phần. Bảng này được cell 10.4 **đọc thẳng từ file kết quả** của lần chạy chính thức, không chép số tay — các bạn đã trình bày chi tiết từng dòng ở các mục trước.

**Kết luận** của thí nghiệm: 4 ví dụ giúp model nắm quy ước nhãn của bộ dữ liệu, cải thiện có ý nghĩa thống kê — trong phạm vi 1 model, 1 bộ dữ liệu, 1 bộ ví dụ. Em xin hết phần của mình."

[Chuyển sang slide 14 — tài liệu tham khảo.]

> Nếu bị dồn thời gian: bỏ đoạn mở cell 10.3; bảng tổng hợp chỉ nói 1 câu.

---

## Phụ lục A — Giải thích code từng cell (để trả lời "giải thích dòng này")

Yêu cầu thêm 5: người trình bày giải thích được từng dòng. Phần này không đọc khi thuyết trình; dùng khi cô chỉ vào một dòng.

### Phần 0 — Setup

**Cell 0.1 — Cấu hình chung**
- `SMOKE: bool = False`: `True` chạy thử ít câu để bắt lỗi (~2 phút); `False` chạy chính thức. Smoke ghi vào `outputs/_smoke/` để **không đè** kết quả thật.
- `SEED = 42`; `MODEL_INSTRUCT`, `MODEL_BASE`, `TOKENIZER_GPT2`: tên model trên Hugging Face. Base chỉ tải ở phần 5, GPT-2 chỉ lấy tokenizer cho phần 3.
- `DTYPE = torch.float16`: mỗi số 2 byte → 1.54 tỉ × 2 ≈ 3.1 GB. T4 không hỗ trợ bf16 bằng phần cứng.
- `N_TEST = 20 if SMOKE else None`: `None` = toàn bộ 2,999 câu. `N_VAL_PROMPT = 200`: số câu validation chọn prompt. `N_FEWSHOT = 4`. `BATCH_SIZE = 16`. `N_ERROR_SAMPLE = 30`.
- `UIT_VSFC_BASE`: nhánh Parquet do Hugging Face tự chuyển đổi, vì script tải cũ của bộ dữ liệu không chạy với `datasets >= 4.0`.
- `ROOT`, `DATA_DIR`: chạy trên máy thì gốc repo là thư mục cha của `notebook/`; trên Kaggle thì dò file `tasks_prompts.json` trong `/kaggle/input` để không phụ thuộc tên Dataset.
- `RUN_ID`: giờ bắt đầu (giờ Việt Nam) + `_normal`/`_smoke`, ví dụ `20261119_150356_normal`. Mọi log nằm trong `outputs/logs/<RUN_ID>/`.
- `random.seed`, `np.random.seed`, `torch.manual_seed`: cố định 3 nguồn ngẫu nhiên (torch đặt cả seed GPU).

**Cell 0.2 — Hàm ghi log**
- `now_iso()`: thời điểm hiện tại theo giờ Việt Nam, dạng ISO 8601.
- `log(text, part, to_run, echo)`: mở file chế độ `"a"` (append, ghi thêm, không ghi đè). Không có `part` → `run.log`; có `part` → `<part>.log`; `to_run=True` → ghi cả hai. `echo` → in ra màn hình.
- 3 dòng `assert` cuối: file log đã tạo và có dòng mở đầu.

**Cell 0.3 — Môi trường**
- `pkg_version(name)`: dùng `importlib.metadata.version` đọc phiên bản đã cài **mà không cần import thư viện**; chưa cài thì trả "chưa cài".
- `assert torch.cuda.is_available()`: quên bật GPU thì dừng ngay với hướng dẫn.
- `ENV`: Python, torch, CUDA, tên GPU, số GPU (Kaggle cấp 2), VRAM (byte / 1024³ = GB), dtype; vòng `for` thêm 7 thư viện. Đây là nguồn số trên slide 2.

**Cell 0.4 — Tải tokenizer và model**
- `AutoTokenizer.from_pretrained` / `AutoModelForCausalLM.from_pretrained`: tải file từ Hub về bộ nhớ đệm; `Auto…` tự chọn đúng lớp theo `config.json` (ở đây là Qwen2). `CausalLM` = model sinh token tiếp theo.
- `device_map={"": 0}`: cả model lên GPU 0. Không dùng `"auto"` vì có thể bị chia sang GPU 1 của Kaggle T4 x2.
- `model.eval()`: chế độ suy luận, tắt dropout.
- `print(model.generation_config)`: thấy mặc định `do_sample=True`, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.1.
- `MODEL_REVISION`: mã commit trên Hub (`989aa79…`) để người khác tải đúng bản. transformers 5.x trên Kaggle trả `_commit_hash = None` nên hỏi Hub bằng `model_info(...).sha`.

**Cell 0.5 — Kiến trúc**
- `ARCH`: đọc từ `model.config` — số block, hidden, head Q, head KV (GQA), FFN, vocab, context tối đa 32,768, `tie_word_embeddings`.
- `sum(p.numel() for p in model.parameters())`: cộng số phần tử mọi tensor trọng số. `parameters()` trả mỗi tensor một lần, nên embedding dùng chung với LM head chỉ được đếm một lần.
- `assert ... == 1_543_714_304`: khớp số tính tay từ `config.json`.

**Cell 0.6 — Tokenizer GPT-2**: chỉ tải tokenizer (vài MB), `assert vocab_size == 50_257`.

**Cell 0.7 — UIT-VSFC**
- `pd.read_parquet(URL)`: đọc thẳng file Parquet (bảng nén theo cột) từ Hugging Face.
- `assert len(df) == n_expected`: số dòng khớp trang dataset (11,426 / 1,583 / 3,166) → tải đủ, đúng bản.
- `df[df["sentiment"].isin(LABEL_NAMES)]`: giữ nhãn 0 (negative), 2 (positive), **bỏ 1 (neutral)** → còn 10,968 / 1,510 / 2,999.

**Cell 0.8 — Lưu JSON**
- `save_json`: `ensure_ascii=False` để giữ chữ tiếng Việt, `indent=2` cho dễ đọc.
- Ghi `env.json` (môi trường) và `config.json` (toàn bộ tham số + kiến trúc); `assert` đọc lại được.

### Phần 7 — Zero-shot và few-shot

**Cell 7.1 — Tập test, validation, ví dụ few-shot**
- `balanced_sample(df, n, seed)`:
  - `rename_axis("row_id").reset_index()`: giữ số thứ tự gốc làm `row_id` để truy lại câu.
  - `n` khác `None`: `groupby("sentiment")` rồi mỗi nhãn `sample(n // 2)` → cân bằng.
  - `sample(frac=1, random_state=seed)`: xáo toàn bộ thứ tự; cùng seed → cùng thứ tự.
  - Với `N_TEST = None`: lấy hết 2,999 câu, chỉ xáo thứ tự.
- `norm_text`: chữ thường + gộp khoảng trắng, để so trùng nội dung.
- `seen`: tập câu của test ∪ validation. `pool` = câu train **không nằm trong** `seen`, dài 5–20 từ (`str.split().str.len().between(5, 20)`).
- `pos`, `neg`: mỗi nhãn lấy 2 câu theo seed → 2:2 cân bằng (Zhao et al., 2021: few-shot thiên về nhãn chiếm đa số trong ví dụ). Vòng `zip` xếp **xen kẽ** positive, negative để không có 2 câu cùng nhãn liền nhau.
- Lưu `vsfc_test_ids.csv` và `fewshot_examples.json` (sản phẩm nộp "bộ câu thử nghiệm").
- `assert`: cân bằng khi lấy mẫu; ví dụ few-shot **không trùng** câu test; nhãn đúng thứ tự `[2, 0, 2, 0]`; gọi lại cùng seed ra cùng `row_id`.

**Cell 7.2 — Mẫu prompt, `build_prompt`, `parse_label`**
- `PROMPT_TEMPLATES`: 3 mẫu, mỗi mẫu có `instruction`, `item` (khung một câu) và `labels` (chữ của nhãn).
- `build_prompt(text, examples, tpl)`: danh sách khối = [chỉ dẫn] + [mỗi ví dụ: khung câu + nhãn] + [câu cần phân loại, để trống nhãn]; nối bằng dòng trống `"\n\n"`. `examples=[]` → zero-shot.
- `parse_label(output)`:
  - `.lower()` + `translate(... string.punctuation)`: chữ thường, bỏ dấu câu; `" ".join(...split())`: gộp khoảng trắng.
  - `has_pos`, `has_neg`: có từ nhãn tiếng Anh hoặc tiếng Việt.
  - `has_pos == has_neg` (cả hai True hoặc cả hai False) → `None` = không parse được.
- `assert`: parse đúng các dạng "Positive.", "Tích cực", "tiêu cực."; trả `None` khi có cả hai hoặc không có; prompt few-shot bắt đầu bằng chỉ dẫn, kết thúc giống hệt zero-shot, và có đúng 4 khối thêm.

**Cell 7.3 — Phân loại theo lô**
- `ICL_MAX_NEW_TOKENS = 10`: smoke run với 5 token cắt mất "tiêu cực" (nhiều token), mẫu `vi_nhan_vi` không parse được 60% → tăng lên 10.
- `apply_chat_template([...], add_generation_prompt=True, tokenize=False)`: bọc prompt thành lượt `user` theo định dạng chat của Qwen và thêm phần mở đầu lượt `assistant`, để model trả lời.
- `tok.padding_side = "left"` trong `try/finally`: đệm bên trái cho lô này rồi **trả lại như cũ** cho các phần khác.
- `tok(texts, padding=True, add_special_tokens=False)`: mã hoá cả lô, đệm cho bằng độ dài; `add_special_tokens=False` vì chat template đã có token đặc biệt. Kết quả gồm `input_ids` và `attention_mask` (1 = token thật, 0 = đệm).
- `torch.no_grad()`: không lưu gradient → nhanh, ít bộ nhớ.
- `generate(... do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=...)`: greedy; đặt `None` để **ghi đè** các giá trị lấy mẫu mặc định trong `generation_config`; `repetition_penalty=1.0` = không phạt lặp.
- `n_in = enc["input_ids"].shape[1]`; `o[n_in:]`: cắt bỏ phần prompt, chỉ decode token mới sinh; `skip_special_tokens=True` bỏ token kết thúc.
- `run_icl(df, examples, tpl)`: chia df thành lô 16 câu, dựng prompt, gọi `classify_batch`; `cuda_sync()` + `time.perf_counter()` đo thời gian (định nghĩa ở cell 1.1); cột `correct = pred == label` — `None` không bằng nhãn nào nên tự tính là sai.
- `assert classify_batch(probe) == classify_batch(probe)`: greedy chạy 2 lần ra y hệt.

**Cell 7.4 — Chọn mẫu prompt**
- 3 mẫu × 2 cách trên 200 câu validation → `prompt_selection.csv`.
- `groupby("template")["accuracy"].mean()` → accuracy trung bình zero + few; `idxmax()` chọn mẫu cao nhất (bằng nhau thì lấy mẫu đầu). Kết quả: `vi_nhan_vi` 0.875.

**Cell 7.5 — Chạy trên test**
- `tpl = PROMPT_TEMPLATES[CHOSEN_TEMPLATE]`: dùng mẫu đã chốt, không sửa sau khi xem test.
- `run_icl(test_df, [], tpl)` và `run_icl(test_df, FEWSHOT, tpl)` → `icl_zero_shot.csv`, `icl_few_shot.csv` (mỗi câu: `row_id`, câu, nhãn, output, dự đoán, đúng/sai).
- `records7` gộp hai bảng vào log từng mẫu `part07_icl_records.csv`.
- `assert icl_zero["row_id"].equals(icl_few["row_id"])`: cùng câu, cùng thứ tự — điều kiện để chạy McNemar theo cặp.

### Phần 10 — Thử nghiệm có kiểm soát

**Cell 10.1 — Accuracy, confusion matrix**
- `compute_metrics`: `accuracy = mean(correct)`; nửa khoảng tin cậy 95% = `1.96 * sqrt(acc * (1 - acc) / n)` (xấp xỉ chuẩn cho tỉ lệ); đếm `n_unparsed`.
- `confusion`: `pd.crosstab(nhãn thật, dự đoán)`, thêm cột "không parse được"; `reindex(..., fill_value=0)` để luôn đủ hàng/cột kể cả khi bằng 0.
- `toy`: ví dụ 4 câu tính tay (đúng 2, 1 câu không parse) → `assert` accuracy 0.5.
- Vẽ 2 heatmap bằng `imshow`, ghi số từng ô bằng `ax.text`, lưu `confusion_matrix_icl.png`.
- `assert`: tổng mỗi hàng = số câu của nhãn đó.

**Cell 10.2 — McNemar**
- `mcnemar_counts`: chuyển cột đúng/sai thành mảng bool; `a & b` cả hai đúng, `a & ~b` chỉ A (zero-shot) đúng, `~a & b` chỉ B (few-shot) đúng, `~a & ~b` cả hai sai.
- `assert mcnemar([[5, 3], [3, 5]], exact=True).pvalue == 1.0`: b = c thì p = 1; tổng 4 ô = 2,999.
- `mcnemar(table, exact=True)`: kiểm định nhị thức hai phía trên b + c = 167 câu với xác suất 0.5 → p = 1.5e-08.
- Lưu `icl_metrics.json` gồm accuracy, CI, a/b/c/d, p và toàn bộ thiết lập.

**Cell 10.3 — Phân tích lỗi**
- `ERROR_GROUPS`: 5 nhóm ghi sẵn trước khi đọc output.
- `MANUAL_ERROR_GROUPS`: `row_id → (nhóm, lý do)`, Toàn gán tay ngày 19/11.
- `wrong.sample(30, random_state=SEED)`: lấy ngẫu nhiên 30 câu few-shot sai. Câu `pred` là `None` tự vào nhóm (a); còn lại gán theo bảng tay. Lưu `error_analysis_sample.csv`.
- `SUGGEST_RE = r"\b(?:nên|cần|mong|thêm|hơn)\b"`: `\b` là ranh giới từ (không khớp "hơn" trong chữ khác). Đếm trên **toàn bộ** câu negative bị đoán positive: 189/248 (zero-shot), 117/145 (few-shot).
- `assert`: mọi `row_id` có trong kết quả few-shot và không trùng.

**Cell 10.4 — Bảng tổng hợp**
- Đọc lại 8 file kết quả của các phần 3–9; `tok_count`, `count_judged` là hàm đếm nhỏ. Ghi `summary_table.csv` — nguồn của bảng cuối slide 13.
- Ghi `RUN END` và tổng thời gian (7.4 phút).

**Cell cuối** — `shutil.make_archive`: nén `outputs/` trên Kaggle để tải về.

---

## Phụ lục B — Q&A dự kiến

| # | Câu hỏi | Ý trả lời (ngắn) | Nguồn / cell |
|---|---|---|---|
| 1 | Zero-shot khác few-shot ở đâu trong code? | Cùng hàm `build_prompt`; `examples=[]` là zero-shot; `assert` kiểm hai prompt chỉ khác 4 khối ví dụ | 7.2 |
| 2 | Làm sao chắc ví dụ few-shot không trùng test? | Lấy từ train, loại câu trùng nội dung (chữ thường, gộp khoảng trắng) với test/validation; có `assert` | 7.1 |
| 3 | Sao chắc là greedy, chạy lại ra y hệt? | `do_sample=False`; chạy cùng lô 2 lần, `assert` bằng nhau | 7.3 |
| 4 | Sao đệm bên trái? | Model sinh tiếp sau token cuối; đệm phải thì token cuối là token đệm, model sinh sai chỗ | 7.3 |
| 5 | Sao chọn prompt trên validation? | Chọn trên test là "nhìn đáp án" → kết quả test lạc quan quá. Chốt mẫu xong mới chạy test | 7.4 |
| 6 | `vi_nhan_vi` 0.875 vs `vi_nhan_en` 0.8725 — chênh nhỏ vậy chọn có ý nghĩa không? | Chênh nằm trong sai số của 200 câu; mục đích chỉ là **chốt một mẫu theo luật ghi trước**, không khẳng định mẫu nào tốt hơn | 7.4 |
| 7 | Output "Cảm xúc: tiêu cực" xử lý thế nào? | `parse_label` chữ thường, bỏ dấu câu, tìm từ nhãn → negative | 7.2 |
| 8 | Câu không parse được tính thế nào? | Tính là sai, báo riêng tỉ lệ (ở đây 0 / 0) | 7.3, 10.1 |
| 9 | Sao dùng McNemar mà không so 2 khoảng tin cậy? | Hai cách chạy trên cùng câu (dữ liệu theo cặp); McNemar chỉ dùng câu bất đồng b, c. Dietterich (1998): với thuật toán chỉ chạy được 1 lần, McNemar có sai lầm loại I chấp nhận được | 10.2 |
| 10 | b, c là gì? Sao `exact=True`? | b: chỉ zero-shot đúng (47); c: chỉ few-shot đúng (120). `exact` dùng nhị thức chính xác, không cần xấp xỉ chi bình phương | 10.2 |
| 11 | Có cần hiệu chỉnh Bonferroni không? | Không, chỉ 1 phép so sánh | 10.2 |
| 12 | Sao chạy hết 2,999 câu thay vì 500 như kế hoạch? | Sinh theo lô đủ nhanh (~5 phút) → khỏi giải thích cách lấy mẫu; sai số ±1 điểm thay vì ±3.5 | 0.1, 7.5 |
| 13 | Sao 4 ví dụ, xen kẽ? | 2+2 cân bằng nhãn; 3 ví dụ lệch 2:1, mà few-shot dễ thiên về nhãn chiếm đa số / nhãn cuối (Zhao et al., 2021) | 7.1 |
| 14 | Few-shot có làm model "học" không? | Không đổi trọng số; chỉ cho model thấy tác vụ và quy ước nhãn (câu góp ý = negative) ngay trong prompt | slide 9, 13 |
| 15 | Nhóm lỗi do ai gán? Có khách quan không? | Toàn gán tay, nhóm ghi sẵn trước khi đọc output; giới hạn: 1 người gán, không đo độ đồng thuận | 10.3 |
| 16 | Sao bỏ nhãn neutral? | Đề chỉ cần minh hoạ zero vs few-shot; 2 nhãn đơn giản, parse rõ; neutral chỉ 167 câu test | 0.7 |
| 17 | Sao float16 mà không bf16? | T4 không hỗ trợ bf16 bằng phần cứng; fp16 2 byte/tham số → ~3.1 GB | 0.1, 0.4 |
| 18 | Sao số tham số ra 1,543,714,304 mà model card ghi 1.54B? | Đếm `numel()` mọi tensor; embedding dùng chung với LM head nên chỉ đếm 1 lần; `assert` khớp số tính tay | 0.5 |

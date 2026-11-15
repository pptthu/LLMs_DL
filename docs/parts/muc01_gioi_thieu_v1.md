# Mục 01 — Giới thiệu: LLM là gì? (v1)

- **Created**: 2026-11-16T03:15:00+07:00
- **Last Updated**: 2026-11-16T03:20:00+07:00 (Toàn duyệt)
- **Người phụ trách mục** (lý thuyết, slide 3): Muội | **Demo (slide 4, notebook phần 1) và Code**: Toàn
- **Slide**: 3–4 | **Notebook**: phần 1 | **Thời lượng**: khoảng 2 phút
- **Status**: In Progress | **Duyệt**: Toàn, 2026-11-16
- Ghi chú quy trình: theo `rules/RULES_v2.md` §1, dàn ý do người viết. Bản này **AI viết theo yêu cầu của Toàn (16/11)**; Toàn đã duyệt toàn bộ ngày 16/11, giữ mọi ý (đã bỏ nhãn).

---

## Đề của cô  [Người viết]
- Nội dung: LLM là gì? Phân biệt language model, LLM và ứng dụng chatbot.
- Minh hoạ: ví dụ các tác vụ hỏi đáp, tóm tắt, dịch, sinh mã.

## Goals / Purpose  [Người viết]
- Notebook phần 1 ghi `outputs/results/tasks_outputs.csv`: 4 dòng (hỏi đáp, tóm tắt, dịch, sinh mã), mỗi dòng có prompt, output thật của Qwen2.5-1.5B-Instruct (greedy), số token vào/ra, thời gian, kết quả chấm và lý do.
- Hàm `is_even` do model sinh ra được chạy với n = 0, 1, 2, −3 trong cell riêng; kết quả ghi vào cột chấm của dòng sinh mã.
- Slide 4: lưới 2×2, mỗi ô = tác vụ + prompt rút gọn + output thật rút gọn, chân slide ghi "Notebook phần 1, cell …".
- Slide 3: bảng phân biệt language model / LLM / chatbot (`research/noi_dung_muc01_gioi_thieu_v1.md` §4).
- Không làm:
  - So sánh base với instruct trên cùng prompt — thuộc mục 5.
  - Thử nhiều cách sinh (temperature, top-p) — thuộc mục 6; phần 1 chỉ dùng greedy.
  - Đo độ chính xác trên bộ dữ liệu lớn — thuộc mục 7, 10; phần 1 chỉ minh hoạ 4 ví dụ, chấm tay.
  - Hallucination, kiểm chứng sự thật — thuộc mục 9 (chỉ nhắc một câu liên hệ).

## Backbone  [Người viết]
- Model / tokenizer: `Qwen/Qwen2.5-1.5B-Instruct`, fp16, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (model card: `docs/models/qwen2.5-1.5b-instruct_v2.md`, có sơ đồ kiến trúc). Dùng lại `model`, `tok` đã tải ở cell 0.4, không tải lại.

## Input / Output  [Người viết dàn ý, AI chi tiết hoá]
- **Input:** `tasks_prompts.json` — 4 tác vụ (`id, task, prompt, max_new_tokens, check_method, expected`) — trên máy: `data/tasks_prompts.json`; trên Kaggle: Kaggle Dataset chứa thư mục `data/` của repo, gắn vào notebook (cách (a), Toàn chốt 16/11) — đọc bằng `json.loads(path.read_text(encoding="utf-8"))`.
- **Output:** `tasks_outputs.csv` — lưu ở `outputs/results/` (smoke: `outputs/_smoke/results/`) — cột `id, task, prompt, output, n_input_tokens, n_output_tokens, time_s, check_method, judged, judge_note`.
- **Log:** `outputs/logs/<ts>_<mode>/part01_tasks.log` và `part01_tasks_records.csv`, cột `idx, id, task, prompt, n_input_tokens, n_output_tokens, output, time_s` (RUN_LOG_TEMPLATE_v2 §3).

## Lý thuyết cần trình bày (5W1H)  [Người viết dàn ý, AI chi tiết hoá]
| | |
|---|---|
| What | Language model: model gán xác suất cho chuỗi từ, dự đoán từ tiếp theo. LLM: language model dựa trên Transformer, hàng tỉ tham số, học trên lượng văn bản rất lớn. Chatbot: ứng dụng xây trên LLM (LLM + tinh chỉnh hội thoại + giao diện + lớp an toàn). |
| Why | Mọi tác vụ đều đưa được về dạng "văn bản vào → văn bản ra", nên một LLM làm nhiều tác vụ chỉ bằng cách đổi prompt, không train lại. |
| When | n-gram → neural LM (Bengio 2003) → Transformer → LLM (GPT-3, 2020) → ChatGPT (11/2022). |
| Where | Trợ lý ảo, dịch máy, tóm tắt, sinh code; trong bài: Qwen2.5-1.5B-Instruct chạy trên Kaggle T4. |
| Who | OpenAI (GPT), Alibaba Qwen Team (Qwen2.5), Meta (Llama)… |
| How | Prompt được định dạng bằng chat template → model sinh từng token, mỗi bước chọn token có xác suất cao nhất (greedy) → dừng ở token kết thúc hoặc `max_new_tokens`. |
Nguồn: chi tiết và trích dẫn ở `research/noi_dung_muc01_gioi_thieu_v1.md` §1–5, §9 [Bổ sung: Jurafsky & Martin; Bengio et al. 2003; Zhao et al. 2023; Brown et al. 2020; OpenAI 2022]; Slide Ch1 s7, s10, s12, s13, s15.

## How to do it (general plan)  [Người viết]
1. Soạn `data/tasks_prompts.json` (4 tác vụ, đoạn văn tóm tắt nhóm tự viết) → Toàn duyệt nội dung.
2. Cell 1.1: hàm `chat_generate()` chạy thử 1 prompt ngắn.
3. Cell 1.2: đọc `tasks_prompts.json`, chạy 4 tác vụ, mỗi tác vụ 1 dòng log và 1 dòng `_records.csv`.
4. Cell 1.3: tách hàm `is_even` từ output sinh mã, chạy với n = 0, 1, 2, −3 (ghi rõ đây là code của model).
5. Cell 1.4: chấm 4 dòng theo thang ở mục Đánh giá, ghi `tasks_outputs.csv`.
6. Cell Phân tích: 2–3 câu theo khung `research/noi_dung_muc01_gioi_thieu_v1.md` §5.3, chỉ điền sau khi có output.
7. Smoke run (`SMOKE = True`) → kiểm file, log.
8. Normal run (Save & Run All) → điền slide 4 từ `tasks_outputs.csv`.

## Detailed plan  [AI chi tiết hoá]
- **Cách sinh:** `do_sample=False`, truyền tường minh `temperature=None, top_p=None, top_k=None` để tắt cảnh báo vì `generation_config` mặc định có `do_sample=True` (cell 0.4); `repetition_penalty` để mặc định 1.1 của model, ghi vào log — lý do: greedy cho kết quả lặp lại được giữa các lần chạy.
- **`max_new_tokens`:** hỏi đáp 32, tóm tắt 96, dịch 48, sinh mã 192 — ghi trong `tasks_prompts.json`, không đặt cứng trong code.
- **Smoke và normal:** chạy cả 4 tác vụ ở cả hai chế độ (chỉ 4 lần `generate`, vài giây); khác nhau chỉ ở thư mục ghi.
- **Đọc dữ liệu trên Kaggle:** thêm `DATA_DIR` tự dò ở cell 0.1: nếu có `/kaggle/input`, tìm thư mục chứa `tasks_prompts.json` trong đó; không có thì dùng `ROOT / "data"`. In `DATA_DIR` ra log. Lý do: đường dẫn gắn Dataset phụ thuộc tên Dataset, dò file tránh đặt cứng.
- **Cell / hàm:**
  - `chat_generate(model, tok, prompt: str, max_new_tokens: int) -> tuple[str, int, int, float]` (cell 1.1): `tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")` → `model.generate(...)` → cắt phần prompt khỏi output, decode với `skip_special_tokens=True`; trả (output, số token vào, số token ra, thời gian giây).
  - Cell 1.2: vòng lặp 4 tác vụ, gọi `log(..., part="part01_tasks")` mỗi tác vụ, ghi `_records.csv` bằng pandas.
  - `extract_code(text: str) -> str` (cell 1.3): lấy khối ```` ```python ... ``` ```` đầu tiên; không có khối thì lấy cả output.
  - Cell 1.3: `exec(code, ns)` với `ns = {}` riêng, chỉ gọi `ns["is_even"](n)`; bọc `try/except Exception` và ghi lỗi vào `judge_note`, không để lỗi code của model làm dừng notebook.
  - Cell 1.4: chấm theo `check_method`: hỏi đáp, dịch, sinh mã chấm tự động bằng điều kiện ở mục Đánh giá; tóm tắt chấm tay, Toàn điền `judged` trong dict ngay trong cell (ghi rõ "chấm tay: Toàn").
- **Gotchas:**
  - `apply_chat_template` của transformers 5.x có thể trả `BatchEncoding` thay vì tensor; kiểm kiểu trả về khi code, không đoán (đã thấy transformers 5.20.2 trên Kaggle).
  - Qwen chat template tự chèn system prompt mặc định khi không có message `system`; ghi vào log chuỗi prompt sau định dạng của 1 tác vụ để người xem thấy.
  - Code do model sinh có thể sai cú pháp, thiếu hàm `is_even`, hoặc có lệnh nguy hiểm: chỉ `exec` trong namespace riêng, không truyền biến của notebook; Toàn đọc code trước khi chạy normal run.
  - Không chép output vào slide trước khi có normal run.

## Đánh giá  [Người viết]
- **Đánh giá cái gì:** model có làm đúng yêu cầu của từng tác vụ không.
- **Đo bằng gì:** thang 2 mức `đạt / không đạt`, kèm lý do trong `judge_note`:
  - Hỏi đáp: output có chứa "Tokyo" (không phân biệt hoa thường) → đạt.
  - Tóm tắt (chấm tay): đạt khi (1) đúng 1 câu, (2) giữ ý chính của đoạn, (3) không thêm ý không có trong đoạn; thiếu điều nào ghi điều đó.
  - Dịch: đạt khi câu tiếng Anh có nghĩa "thời tiết hôm nay rất đẹp" (chứa "today" và một trong "weather / nice / beautiful / lovely"); câu tham chiếu "The weather is very nice today."
  - Sinh mã: đạt khi `is_even(0), is_even(1), is_even(2), is_even(-3)` = `True, False, True, False`.
- **Thế nào là đạt:** cả 4 dòng có output thật và có chấm kèm lý do. Số tác vụ "đạt" **không** phải điều kiện; tác vụ không đạt thì ghi rõ sai ở đâu và nhắc trên slide (liên hệ mục 9).
- **Output:** `outputs/results/tasks_outputs.csv`, cột `judged`, `judge_note`.

## Success criteria  [Người viết]
- **Assert trong notebook:** cell 1.1 output không rỗng; cell 1.2 đủ 4 dòng, mỗi `output` không rỗng, `n_output_tokens <= max_new_tokens`; cell 1.4 cột `judged` chỉ có `đạt` hoặc `không đạt`, không để trống.
- **Smoke run:** có `outputs/_smoke/results/tasks_outputs.csv` (4 dòng), `outputs/_smoke/logs/<ts>_smoke/part01_tasks.log` có dòng START, 4 dòng mỗi tác vụ, dòng END + tổng kết; `part01_tasks_records.csv` 4 dòng.
- **Normal run:** `outputs/results/tasks_outputs.csv` 4 dòng có chấm; slide 4 lấy đúng từ file này, chân slide ghi `run_id`.

## Kết quả audit (chưa audit)  [Toàn điền sau normal run]
- chưa audit

## Slide
- Slide 3: Language model, LLM và chatbot — dòng định nghĩa LM + công thức chain rule; bảng phân biệt 3 khái niệm; câu chốt. Nguồn: Jurafsky & Martin; Zhao et al. 2023; OpenAI 2022.
- Slide 4: LLM làm được những tác vụ gì — lưới 2×2 (tác vụ, prompt rút gọn, output thật rút gọn, đạt/không đạt); một dòng phân tích; chân slide "Notebook phần 1, cell 1.2–1.4 · Qwen2.5-1.5B-Instruct · greedy · run_id …".

## Q&A dự kiến (khoảng 5 câu)
| Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|
| LM và LLM khác nhau ở đâu? | Cùng mục tiêu đoán token tiếp theo; LLM khác ở quy mô, kiến trúc Transformer và khả năng như in-context learning | Jurafsky & Martin; Zhao et al. 2023 |
| Bao nhiêu tham số thì gọi là "large"? | Chưa có ngưỡng chính thức; model nhóm 1.54 tỉ (đếm bằng code ở cell 0.5) là LLM cỡ nhỏ | Zhao et al. 2023; notebook cell 0.5 |
| Vì sao một model làm nhiều tác vụ mà không train lại? | Tác vụ mô tả bằng văn bản trong prompt, model chỉ viết tiếp; GPT-3 làm vậy mà không cập nhật trọng số | Brown et al. 2020 |
| Vì sao dùng greedy? | Bản Instruct mặc định bốc mẫu ngẫu nhiên (`do_sample=True`, cell 0.4); greedy cho output lặp lại được để chấm | notebook cell 0.4; mục 6 |
| Qwen2.5-1.5B-Instruct có phải chatbot không? | Chưa; là model đã post-training, thành chatbot khi có giao diện hội thoại và lớp an toàn | Model card Qwen; OpenAI 2022 |

## Links
- Description cả bài: [../presentation_description_v2.md](../presentation_description_v2.md)
- Research: [../research/noi_dung_muc01_gioi_thieu_v1.md](../research/noi_dung_muc01_gioi_thieu_v1.md)
- Detail plan notebook: [../plans/detail_plan_notebook_v1.md](../plans/detail_plan_notebook_v1.md) phần 1
- Tiến độ và bàn giao code: [../../progress_status.md](../../progress_status.md)
- Notebook: `notebook/llm_presentation.ipynb`, phần 1 (cell 1.1–1.4)

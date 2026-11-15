# Khuôn description từng mục (v2)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-16T01:40:17+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-16T01:40:17+07:00
- Trạng thái: **Duyệt: Toàn, 2026-11-16**
- Dựa trên: `PART_TEMPLATE_v1.md`; gốc `Lab-DL/labs/lab1/docs/templates/DESC_TEMPLATE_v3.md` và mục "Kết quả audit" của `Lab-DL/labs/lab1/docs/phases/*_v1.md`.
- Thay đổi so với v1:
  - Ghi rõ mục nào **người phụ trách viết dàn ý**, mục nào AI chi tiết hoá.
  - Thêm Backbone, Input/Output (lấy từ đâu, lưu ở đâu), Đánh giá, Log, Kết quả audit.
  - How to do phải nói rõ từng bước làm gì.
  - Người code mọi phần là Toàn; Links trỏ thẳng `notebook/llm_presentation.ipynb` (bỏ `notebook/parts/`).
  - v1 giữ nguyên, không dùng.
- Dùng cho: `docs/parts/mucNN_<tên>_vN.md`, mỗi mục của đề một file.

## Cách dùng
- **[Người viết]**: người phụ trách mục tự viết và tự sắp xếp. AI không viết thay.
- **[AI chi tiết hoá]**: AI viết sau bước Discussion, chỉ dựa trên dàn ý; ý AI tự thêm đánh `[đề xuất]`.
- **Goals / Purpose**: 2–4 dòng, mỗi dòng là một **kết quả kiểm được** (file nào, bảng nào, slide nào), không phải một hoạt động.
  - Viết: "Notebook ghi `outputs/results/token_counts.csv` có số token của 2 câu × 2 tokenizer".
  - Không viết: "tìm hiểu tokenization".
- **Không làm:** chỉ ghi việc người đọc hoặc AI **dễ tưởng** thuộc mục này, và nói rõ việc đó thuộc mục nào hoặc vì sao ngoài đề.
- **Đánh giá:** viết tiêu chí **trước khi** chạy. Chấm tay thì ghi sẵn thang chấm.
- **How to do:** mỗi bước một small prompt: làm gì, input, tham số, output file; luôn có smoke run trước normal run.
- **Số liệu** chỉ ghi khi đã có file kết quả của normal run (kèm `run_id`); chưa có thì ghi "chưa chạy".

## How to do: từng bước
Làm theo `rules/RULES_v2.md` §1.1 bước 1–4 (Description → Discussion → Detail plan → Duyệt); sau normal run làm bước 8 (điền "Kết quả audit"). Tóm tắt:
1. Tạo `docs/parts/mucNN_<tên>_v1.md`, copy khối "Khuôn" bên dưới.
2. Người phụ trách điền mọi mục [Người viết].
3. AI hỏi lại tối đa 5 câu; ghi câu trả lời vào "Quyết định".
4. AI viết "Detailed plan", đánh `[đề xuất]` chỗ tự thêm.
5. Toàn duyệt, ghi "Duyệt".
6. Sau normal run: Toàn điền "Kết quả audit", đổi Status.

## Khuôn

    # Mục <NN> — <tên ngắn> (v<N>)

    - **Created**: YYYY-MM-DDTHH:MM:SS+07:00
    - **Last Updated**: YYYY-MM-DDTHH:MM:SS+07:00 (<người cập nhật>)
    - **Người phụ trách mục** (dàn ý, lý thuyết, slide, thuyết trình): <tên> | **Code**: Toàn
    - **Slide**: <số slide> | **Notebook**: phần <NN> | **Thời lượng**: <phút>
    - **Status**: To Do | In Progress | Done | **Duyệt**: <Toàn, ngày>

    ---

    ## Đề của cô  [Người viết]
    - Nội dung: <chép đúng cột "Nội dung sinh viên cần trình bày">
    - Minh hoạ: <chép đúng cột "Yêu cầu minh họa">

    ## Goals / Purpose  [Người viết]
    - <kết quả kiểm được>
    - Không làm:
      - <việc dễ tưởng thuộc mục này> — <thuộc mục nào / vì sao ngoài đề>

    ## Backbone  [Người viết]
    - Model / tokenizer: <tên> (model card: docs/models/<tên>_v2.md, có sơ đồ kiến trúc)

    ## Input / Output  [Người viết dàn ý, AI chi tiết hoá]
    - **Input:** <tên> — lấy từ <Hugging Face / data/<file> / output phần NN> — bằng <hàm / lệnh>
    - **Output:** <tên> — lưu ở <outputs/results/<file>, outputs/figures/<file>> — cột <…>
    - **Log:** outputs/logs/<ts>_normal/partNN_<tên>.log và _records.csv (cột theo RUN_LOG_TEMPLATE_v2 §3)

    ## Lý thuyết cần trình bày (5W1H)  [Người viết dàn ý, AI chi tiết hoá]
    | | |
    |---|---|
    | What | <…> |
    | Why | <…> |
    | When | <…> |
    | Where | <…> |
    | Who | <…> |
    | How | <…> |
    Nguồn: <[Slide ChX sN] / [Bổ sung: tác giả năm]>

    ## How to do it (general plan)  [Người viết]
    1. <bước = 1 small prompt: làm gì, input, tham số, output>
    2. ...
    n. Smoke run (SMOKE = True) → kiểm <file, log>
    n+1. Normal run (Save & Run All)

    ## Detailed plan  [AI chi tiết hoá]
    - **<Tham số / quyết định>:** <giá trị> — <lý do>
    - **Cell / hàm:** `<tên hàm(tham số) -> kiểu trả về>` (<cell số mấy trong phần>)
    - **Gotchas:**
      - <lỗi dễ gặp / giới hạn> (<nguồn>)

    ## Đánh giá  [Người viết]
    - **Đánh giá cái gì:** <…>
    - **Đo bằng gì:** <metric + công thức, hoặc thang chấm tay ghi sẵn>
    - **Thế nào là đạt:** <ngưỡng / điều kiện>
    - **Output:** <file kết quả, cột chứa điểm>

    ## Success criteria  [Người viết]
    - **Assert trong notebook:** <assert nào>
    - **Smoke run:** <file nào phải có, log có đủ 3 phần>
    - **Normal run:** <bảng / số / hình phải có>

    ## Kết quả audit (<YYYY-MM-DD>)  [Toàn điền sau normal run]
    - **Assert trong notebook:** <cell> → đạt ✔ / lỗi ✘
    - **Smoke run:** <thư mục outputs/_smoke/logs/<ts>_smoke/> ✔ / ✘
    - **Normal run:** <run_id, file, số thật> ✔ / ✘
    - **Đánh giá:** <đạt / không đạt theo mục Đánh giá> ✔ / ✘
    - **Slide khớp file kết quả:** <slide số> ✔ / ✘
    - **Chưa sửa (lý do):** <…> | chưa audit thì ghi "chưa audit"

    ## Slide
    - Slide <số>: <tiêu đề> — <nội dung chính, minh hoạ, chân slide ghi nguồn + "Notebook phần NN, cell …">

    ## Q&A dự kiến (khoảng 5 câu)
    | Câu hỏi | Ý trả lời | Nguồn |
    |---|---|---|

    ## Links
    - Description cả bài: [../presentation_description_v2.md](../presentation_description_v2.md)
    - Research: [../research/<file>.md](../research/)
    - Tiến độ và bàn giao code: [../../progress_status.md](../../progress_status.md)
    - Notebook: `notebook/llm_presentation.ipynb`, phần <NN> (cell <…>)

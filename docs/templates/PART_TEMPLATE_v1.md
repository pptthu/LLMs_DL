# Khuôn desc từng mục (v1)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00
- Dựa trên: `Lab-DL/labs/lab1/docs/templates/DESC_TEMPLATE_v3.md` (khuôn phase doc của Lab 1). Đổi "module/script" thành "mục của đề + phần notebook"; thêm phần lý thuyết, minh hoạ, slide, Q&A.
- Dùng cho: `docs/parts/mucNN_<tên>_vN.md`, mỗi mục của đề một file.

## Cách dùng
- Copy khối bên dưới, thay phần `<...>`. Khoảng 40–80 dòng.
- **Goals / Purpose** (`[đề xuất]` tới khi duyệt): 2–4 dòng. Mỗi dòng là một **kết quả kiểm được** (file nào, bảng nào, slide nào), không phải một hoạt động.
  - Viết: "Notebook ghi `outputs/results/token_counts.csv` có số token của 2 câu × 2 tokenizer".
  - Không viết: "tìm hiểu tokenization".
- **Không làm:** chỉ ghi việc người đọc hoặc AI **dễ tưởng** thuộc mục này; nói rõ việc đó thuộc mục nào hoặc vì sao ngoài đề.
- **Success criteria** (`[đề xuất]` tới khi duyệt): mỗi tiêu chí kiểm được bằng **một cell, một file hoặc một slide**. Chia 3 tầng: assert trong notebook → smoke run → normal run.
- **Lý thuyết:** ghi rõ **[Slide ChX sN]** hay **[Bổ sung]**; khẳng định nào cũng có nguồn ở `references.md`.
- **How to do it:** mỗi bước một small prompt, ghi rõ tham số; luôn có smoke run trước normal run.
- **Số liệu** chỉ ghi khi có file kết quả của lần chạy chính thức, kèm đường dẫn; chưa có thì ghi "chưa chạy".

## Khuôn

    # Mục <NN> — <tên ngắn> (v<N>)

    - **Created**: YYYY-MM-DDTHH:MM:SS+07:00
    - **Last Updated**: YYYY-MM-DDTHH:MM:SS+07:00 (<người cập nhật>)
    - **Người phụ trách**: <tên>
    - **Slide**: <số slide> | **Notebook**: phần <NN> | **Thời lượng**: <phút>
    - **Status**: To Do | In Progress | Done

    ---

    ## Đề của cô
    - Nội dung: <chép đúng cột "Nội dung sinh viên cần trình bày">
    - Minh hoạ: <chép đúng cột "Yêu cầu minh họa">

    ## Goals / Purpose  [đề xuất]
    - <kết quả kiểm được>
    - Không làm:
      - <việc dễ tưởng thuộc mục này> — <thuộc mục nào / vì sao ngoài đề>

    ## Lý thuyết cần trình bày (5W1H)
    | | |
    |---|---|
    | What | <…> |
    | Why | <…> |
    | When | <…> |
    | Where | <…> |
    | Who | <…> |
    | How | <…> |
    Nguồn: <[Slide ChX sN] / [Bổ sung: tác giả năm]>

    ## Input / Output
    - **Input:** <model, tokenizer, dữ liệu, bộ câu trong data/>
    - **Output:** <outputs/results/…, outputs/figures/…>

    ## How to do it (general plan)
    1. <bước = 1 small prompt, có tham số>
    2. ...
    n. Smoke run (SMOKE = True) → kiểm <…>
    n+1. Normal run (Save & Run All)

    ## Detailed plan
    - **<Tham số / quyết định>:** <giá trị> — <lý do>
    - **Cell / hàm:** `<tên hàm(tham số) -> kiểu trả về>` (<cell số mấy trong phần>)
    - **Gotchas:**
      - <lỗi dễ gặp / giới hạn> (<nguồn>)

    ## Success criteria  [đề xuất]
    - **Assert trong notebook:** <assert nào>
    - **Smoke run:** <file nào phải có, kiểm gì>
    - **Normal run:** <bảng / số / hình phải có>

    ## Slide
    - Slide <số>: <tiêu đề> — <nội dung chính, minh hoạ, chân slide ghi nguồn + "Notebook phần NN, cell …">

    ## Q&A dự kiến (khoảng 5 câu)
    | Câu hỏi | Ý trả lời | Nguồn |
    |---|---|---|

    ## Links
    - Desc cả bài: [../presentation_description_v1.md](../presentation_description_v1.md)
    - Research: [../research/<file>.md](../research/)
    - Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md)
    - Notebook: `notebook/parts/part_<NN>_<tên>.ipynb` → `notebook/llm_presentation.ipynb` phần <NN>

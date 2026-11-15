# Khuôn description cả bài (v2)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-16T01:40:17+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-16T01:40:17+07:00
- Trạng thái: **Duyệt: Toàn, 2026-11-16**
- Dựa trên: `DESC_TEMPLATE_v1.md`; gốc `Lab-DL/labs/lab1/docs/templates/DESC_TEMPLATE_v1.md` (Thư).
- Thay đổi so với v1: thêm Backbone, Input/Output ghi rõ lấy từ đâu / lưu ở đâu, mục Đánh giá, How-to-do phải ghi cụ thể từng bước; ghi rõ phần nào người viết, phần nào AI chi tiết hoá. v1 giữ nguyên, không dùng.
- Dùng cho: `docs/presentation_description_vN.md`. Desc từng mục dùng `PART_TEMPLATE_v2.md`.

## Cách dùng
- **Người viết dàn ý** (scaffold): mọi mục có nhãn **[Người viết]**. Mỗi dòng phải cụ thể đủ để AI không phải đoán: tên model, file, con số, slide.
- **AI chi tiết hoá** các mục có nhãn **[AI chi tiết hoá]**, chỉ dựa trên dàn ý; ý nào AI tự thêm đánh `[đề xuất]`.
- Success criteria và Đánh giá phải có con số, file hoặc slide cụ thể; không viết chung chung kiểu "trình bày tốt".
- How-to-do: mỗi bước là một small prompt, nói rõ làm gì, input nào, output file nào.
- Đổi nội dung thì lưu `_vN+1`, không sửa đè.

## How to do: từng bước
1. Copy khối "Khuôn" bên dưới sang `docs/presentation_description_vN.md` (bản mới).
2. Người viết điền các mục [Người viết]: Background, Purpose, Backbone, Success criteria.
3. Người viết ghi dàn ý cho Input/Output, Đánh giá, How-to-do (mỗi ý một dòng).
4. Đưa AI hỏi lại (tối đa 5 câu), trả lời, ghi vào "Quyết định".
5. AI chi tiết hoá các mục [AI chi tiết hoá], đánh `[đề xuất]` chỗ tự thêm.
6. Toàn duyệt từng `[đề xuất]`, ghi "Duyệt: Toàn, <ngày>".

## Khuôn

    # Presentation LLMs — <tên ngắn>  (v<N>)
    - Tác giả: <tên> | Tạo: YYYY-MM-DDTHH:MM:SS+07:00 | Cập nhật: <tên>, <thời điểm>
    - Trạng thái: To Do | In Progress | Done | Duyệt: <Toàn, ngày>

    ## Background  [Người viết]
    <1–3 câu: đề của cô, mục tiêu, góp ý đã nhận>

    ## Purpose  [Người viết]
    - Làm: <sản phẩm cụ thể: slide, notebook, bộ câu, tài liệu>
    - Không làm: <ngoài phạm vi, kèm lý do>

    ## Backbone  [Người viết]
    | Model / tokenizer | Nguồn | Dùng ở phần | Model card |
    |---|---|---|---|

    ## Input / Output  [Người viết dàn ý, AI chi tiết hoá]
    - Input: <tên> — lấy từ <Hugging Face / data/… / phần trước> — bằng <lệnh / hàm>
    - Output: <tên> — lưu ở <outputs/results/… | outputs/figures/… | outputs/logs/<ts>_normal/…> — cột <…>
    - Code: notebook/llm_presentation.ipynb, phần <0–10>

    ## Success criteria  [Người viết]
    - <yêu cầu của đề / của cô> → <slide nào, file nào chứng minh>

    ## Đánh giá  [Người viết dàn ý, AI chi tiết hoá]
    | Phần | Đánh giá cái gì | Đo bằng gì (metric / tiêu chí chấm) | Thế nào là đạt | Output |
    |---|---|---|---|---|

    ## How-to-do  [Người viết dàn ý, AI chi tiết hoá]
    1. <bước = 1 small prompt: làm gì, input, output>
    2. ...
    n. Smoke run → kiểm <file, log>
    n+1. Normal run → log outputs/logs/<ts>_normal/
    n+2. Audit: đối chiếu với file này

    ## Quyết định (người ghi)
    - <ngày>: <quyết định> — <lý do>

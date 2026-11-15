# Khuôn desc cả bài (v1)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00
- Dựa trên: `Lab-DL/labs/lab1/docs/templates/DESC_TEMPLATE_v1.md` (Thư)
- Dùng cho: desc cả bài `docs/presentation_description_vN.md`. Desc từng mục dùng `PART_TEMPLATE_v1.md`.

## Cách dùng
- Copy khối bên dưới, thay phần `<...>`. Giữ khoảng 20–30 dòng.
- Người viết dàn ý (Purpose, Success criteria, phần "Không làm"). AI chỉ chi tiết hoá rồi đưa người duyệt.
- Success criteria phải có con số, file hoặc slide cụ thể, không ghi chung chung kiểu "trình bày tốt".
- How-to-do: mỗi bước là một small prompt.
- Đổi nội dung thì lưu bản mới `_vN+1`, không sửa đè bản cũ.

## Khuôn

    # Presentation LLMs — <tên ngắn>  (v<N>)
    - Tác giả: <tên> | Tạo: YYYY-MM-DDTHH:MM:SS+07:00 | Cập nhật: <tên>, <thời điểm>
    - Trạng thái: To Do | In Progress | Done
    ## Background
    <1–3 câu: đề của cô, mục tiêu, góp ý đã nhận>
    ## Purpose
    - Làm: <sản phẩm cụ thể: slide, notebook, bộ câu, tài liệu>
    - Không làm: <ngoài phạm vi, ví dụ fine-tune>
    ## Input / Output
    - Input: <model, dữ liệu, bộ câu> (<nguồn, link>)
    - Output: <file kết quả, hình, slide> (<đường dẫn>)
    - Code: notebook/llm_presentation.ipynb, phần <0–10>
    ## Success criteria
    - <yêu cầu của đề / của cô> → <slide nào, file nào chứng minh>
    - <thời lượng, số slide>
    ## How-to-do
    1. <bước 1 = 1 small prompt>
    2. ...
    ## Quyết định (người ghi)
    - <ngày>: <quyết định> — <lý do>

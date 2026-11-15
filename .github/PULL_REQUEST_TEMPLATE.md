## Việc đã làm
<phần notebook / tài liệu nào, thay đổi gì>
<!-- Tiêu đề PR: <type>(<phạm vi>): <việc đã làm>, ví dụ feat(part07): add zero-shot and 3-shot prompt builder -->

Closes #<số issue>

## Vì sao
<mục tiêu>

## Kiểm chứng thế nào
<smoke run / normal run, cell nào, thư mục log `outputs/.../logs/<ts>_…/`, output thật tóm tắt>

## Checklist
- [ ] Chỉ sửa đúng phần nêu trong issue (code notebook chỉ Toàn sửa trong `notebook/llm_presentation.ipynb`, quyết định 15/11)
- [ ] Mỗi cell code có markdown trước, comment, cell "Phân tích" sau output
- [ ] Tham số sinh (`do_sample`, `max_new_tokens`, …) truyền tường minh
- [ ] `progress_status.md` đã cập nhật
- [ ] `AI_usage.md` đã cập nhật nếu có dùng AI
- [ ] Không commit dữ liệu tải về, model, `outputs/_smoke/`, token
- [ ] Tôi giải thích được từng dòng code trong PR này

# Tài liệu bài Presentation LLMs — đọc file này trước (v2)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-16T01:40:17+07:00 | Trạng thái: **Duyệt: Toàn, 2026-11-16**
- Dựa trên: `Lab-DL/labs/lab1/docs/00_README_v2.md`
- Thay đổi so với v1: trỏ các bản `_v2`/`_v3` ngày 16/11; Toàn duyệt, Thư review. v1 giữ nguyên, không dùng.

## Quy tắc version
- Tên file có hậu tố `_vN`. Luôn dùng bản có **số lớn nhất**; bản cũ giữ nguyên, không sửa đè.
- Muốn đổi nội dung: tạo bản `_vN+1` sau khi Toàn duyệt (xem `rules/RULES_v2.md`).

## Description hiện hành
- Description cả bài: **`presentation_description_v2.md`**. AI sinh slide, báo cáo từ file này và các file nó trỏ tới.
- `../project_description.md`: bản tổng quan Define + Design.
- `../progress_status.md`: tiến độ, bàn giao code, bằng chứng chạy.

## Mục lục (bản hiện hành)

| File / thư mục | Nội dung |
|---|---|
| `presentation_description_v2.md` | Description cả bài: Purpose, Backbone, I/O, Success criteria, Đánh giá, How-to-do |
| `references.md` | Tài liệu tham khảo (sản phẩm nộp số 4) |
| `rules/RULES_v2.md` | Quy trình Description → Discussion → Detail → Implementation → Smoke run → Normal run → Audit; luật chung; phân vai; phân công |
| `rules/FOLDER_STRUCTURE_v2.md` | Cây thư mục hiện tại, input lấy từ đâu, output lưu ở đâu, log, model lưu ở đâu |
| `templates/DESC_TEMPLATE_v2.md` | Khuôn description cả bài |
| `templates/PART_TEMPLATE_v2.md` | Khuôn description từng mục → `parts/`, có Đánh giá và Kết quả audit |
| `templates/RUN_LOG_TEMPLATE_v2.md` | Khuôn log mỗi lần chạy (thư mục theo ngày giờ, nội dung bắt buộc) |
| `templates/MODEL_CARD_TEMPLATE_v2.md` | Khuôn model card, bắt buộc có sơ đồ kiến trúc |
| `templates/EXPERIMENT_NOTE_TEMPLATE_v1.md` | Khuôn ghi chú thí nghiệm |
| `templates/BUG_REPORT_TEMPLATE_v1.md` | Khuôn báo lỗi |
| `templates/SCRIPT_TEMPLATE_v1.md` | Khuôn kịch bản thuyết trình và Q&A |
| `plans/ke_hoach_presentation_llms_v2.md` | Kế hoạch nhóm (nội dung, phương án demo, hạn từng người) |
| `plans/khung_presentation_llms_v2.md` | Khung slide, khung notebook, danh sách việc và hạn |
| `plans/plan_presentation_llm_v3.md` | Plan cấu trúc + plan code hiện hành |
| `plans/detail_plan_notebook_v1.md` | Từng bước code phần 0–10: cell, việc làm, cách kiểm, file ra |
| `plans/huong_dan_tung_buoc_v1.md` | Từng bước: quy trình làm một phần, smoke run / normal run trên Kaggle, duyệt và commit |
| `parts/` | Description từng mục `mucNN_<tên>_v1.md` (người phụ trách mục viết dàn ý; chưa có) |
| `research/research_presentation_llm_v1.md` | Kiểm chứng kỹ thuật: Qwen, UIT-VSFC, Kaggle, API |
| `research/nam_kien_thuc_5w1h_v1.md` | Tóm tắt 10 mục theo 5W1H |
| `research/huong_dan_research_presentation_v1.md` | Hướng dẫn tự research |
| `research/noi_dung_mucNN_*` | Nội dung chi tiết từng mục (bản số lớn nhất; mục 2 v2 có so sánh LLM với mô hình khác; mục 7, 10 trong `noi_dung_toan_muc07_muc10_v2.md`) |
| `drafts/` | Bản nháp báo cáo, lý thuyết và chữ slide mục 7, 10 (Toàn); không phải sản phẩm nộp |
| `models/*_v2.md` | Model card có sơ đồ kiến trúc: Qwen2.5-1.5B-Instruct, Qwen2.5-1.5B (base), GPT-2 tokenizer |
| `models/bert-base-multilingual-cased_v1.md`, `models/qwen2-vl-2b-instruct_v1.md` | Model card 2 model dùng cho demo so sánh ở mục 2 |
| `experiments/experiment_log.md` | Sổ tóm tắt các lần chạy, trỏ tới `outputs/logs/` |
| `scripts/`, `qa/`, `bugs/` | Chưa có; tạo khi có file đầu tiên |

## Thứ tự đọc
1. File này
2. `rules/RULES_v2.md`
3. `rules/FOLDER_STRUCTURE_v2.md`
4. `presentation_description_v2.md`
5. `plans/plan_presentation_llm_v3.md`, `plans/huong_dan_tung_buoc_v1.md`, `plans/detail_plan_notebook_v1.md`
6. `research/noi_dung_mucNN_*` của mục mình (bản số lớn nhất)
7. Template cần dùng

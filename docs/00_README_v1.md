# Tài liệu bài Presentation LLMs — đọc file này trước (v1)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00 | Trạng thái: **[đề xuất]**, chờ Thư duyệt
- Dựa trên: `Lab-DL/labs/lab1/docs/00_README_v2.md`

## Quy tắc version
- Tên file có hậu tố `_vN`. Luôn dùng bản có **số lớn nhất**; bản cũ giữ nguyên, không sửa đè.
- Muốn đổi nội dung: tạo bản `_vN+1` sau khi Thư duyệt (xem `rules/RULES_v1.md`).

## Desc hiện hành
- Desc cả bài: **`presentation_description_v1.md`**. AI sinh slide, báo cáo từ file này và các file nó trỏ tới.
- `../project_description.md`: bản tổng quan Define + Design.
- `../progress_status.md`: tiến độ và bằng chứng chạy.

## Mục lục

| File / thư mục | Nội dung |
|---|---|
| `presentation_description_v1.md` | Desc cả bài: đề, sản phẩm nộp, success criteria, các bước |
| `references.md` | Danh sách tài liệu tham khảo (sản phẩm nộp số 4) |
| `rules/RULES_v1.md` | Quy trình làm việc, luật chung, phân vai, phân công |
| `rules/FOLDER_STRUCTURE_v1.md` | Cây thư mục, kết quả nào lưu ở đâu, khác Lab 1 ở đâu |
| `templates/DESC_TEMPLATE_v1.md` | Khuôn desc cả bài |
| `templates/PART_TEMPLATE_v1.md` | Khuôn desc từng mục → `parts/` |
| `templates/RUN_LOG_TEMPLATE_v1.md` | Khuôn ghi lần chạy (smoke / normal) |
| `templates/MODEL_CARD_TEMPLATE_v1.md` | Khuôn model card (model dùng, không train) |
| `templates/EXPERIMENT_NOTE_TEMPLATE_v1.md` | Khuôn ghi chú thí nghiệm |
| `templates/BUG_REPORT_TEMPLATE_v1.md` | Khuôn báo lỗi (6 mục cô yêu cầu + 3 mục sau khi sửa) |
| `templates/SCRIPT_TEMPLATE_v1.md` | Khuôn kịch bản thuyết trình và Q&A |
| `plans/ke_hoach_presentation_llms_v1.md` | Kế hoạch nhóm (nội dung, phương án demo, hạn từng người) |
| `plans/khung_presentation_llms_v1.md` | Khung slide, khung notebook, danh sách việc và hạn |
| `plans/plan_presentation_llm_v2.md` | Plan cấu trúc + plan code **hiện hành** (v1 giữ lại, không dùng) |
| `parts/` | Desc từng mục `mucNN_<tên>_v1.md` (người phụ trách viết; chưa có) |
| `research/research_presentation_llm_v1.md` | Kiểm chứng kỹ thuật: Qwen, UIT-VSFC, Kaggle, API, link tài liệu |
| `research/nam_kien_thuc_5w1h_v1.md` | Tóm tắt 10 mục theo 5W1H + câu tự kiểm tra |
| `research/huong_dan_research_presentation_v1.md` | Hướng dẫn tự research: baseline, 3 mức hiểu, 8 bước |
| `research/noi_dung_mucNN_*_v1.md` | Nội dung chi tiết từng mục (mục 7, 10 trong `noi_dung_toan_muc07_muc10_v1.md`) |
| `models/` | Model card: Qwen2.5-1.5B-Instruct, Qwen2.5-1.5B (base), GPT-2 tokenizer |
| `experiments/experiment_log.md` | Sổ ghi các lần chạy và thí nghiệm |
| `scripts/`, `qa/`, `bugs/` | Chưa có; tạo khi có file đầu tiên |

## Thứ tự đọc
1. File này
2. `rules/RULES_v1.md`
3. `rules/FOLDER_STRUCTURE_v1.md`
4. `presentation_description_v1.md`
5. `plans/plan_presentation_llm_v2.md`
6. `research/noi_dung_mucNN_*` của mục mình
7. Template cần dùng

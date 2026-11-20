# Tài liệu bài Presentation LLMs — đọc file này trước (v3)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-21
- Thay đổi so với v2 (v2 giữ nguyên, không dùng): trỏ bản hiện hành ngày 21/11; Toàn trình bày cả 10 mục; tách rõ tài liệu **hiện hành** và **lịch sử**.
- Lần chạy chính thức: **`20261121_010447_normal`** (Kaggle T4, 9.2 phút). Mọi số trên slide lấy từ `../outputs/results/` của lần chạy này.

## Quy tắc version
- Tên file có hậu tố `_vN`. Luôn dùng bản có **số lớn nhất**; bản cũ giữ nguyên, không sửa đè.
- Muốn đổi nội dung: tạo bản `_vN+1` sau khi Toàn duyệt (`rules/RULES_v3.md`).

## Tài liệu hiện hành

| Cần gì | File |
|---|---|
| Yêu cầu của cô (đề, góp ý, yêu cầu thêm) | `plans/yeu_cau_cua_co_v1.md` |
| Tổng quan dự án: 10 mục → phần notebook → slide, file vào / ra | `../project_description.md` |
| Description cả bài, đánh giá, kết quả, quyết định | `presentation_description_v3.md` |
| **Giải thích lý thuyết, code, output từng cell, cách trình bày, câu hỏi** | `giai_thich_du_an_v4.md` |
| Tóm gọn 10 mục (lý thuyết + cell + số) | `parts/ban_giao_toan_v2.md` |
| **Kịch bản 16 slide, giải thích hình, thứ tự ôn** | `scripts/kich_ban_toan_v2.md` |
| Khung 16 slide (slide ảnh: `../slides/infographic/llm_slides.pdf`) | `plans/khung_presentation_llms_v5.md` |
| Tài liệu tham khảo (sản phẩm nộp số 4) | `references.md` |
| Sổ ghi mọi lần chạy | `experiments/experiment_log.md` |
| Phân tích lỗi mục 10 | `experiments/error_analysis_v2.md` |
| Luật, phân vai | `rules/RULES_v3.md` |
| Cây thư mục, input / output lưu ở đâu | `rules/FOLDER_STRUCTURE_v3.md` |
| Khuôn log mỗi lần chạy | `templates/RUN_LOG_TEMPLATE_v3.md` |
| Model card (có sơ đồ kiến trúc) | `models/qwen2.5-1.5b-instruct_v2.md`, `models/qwen2.5-1.5b-base_v2.md`, `models/gpt2-tokenizer_v2.md` |
| Tiến độ, nhật ký | `../progress_status.md` |
| Khai báo dùng AI | `../AI_usage.md` |

## Thứ tự đọc để trình bày
1. `plans/yeu_cau_cua_co_v1.md`
2. `scripts/kich_ban_toan_v2.md` Phần A (thứ tự ôn, chỗ coi kĩ)
3. `giai_thich_du_an_v4.md` theo thứ tự ôn ở bước 2
4. `scripts/kich_ban_toan_v2.md` Phần B, C (tập nói, bấm giờ)

## Tài liệu lịch sử (giữ nguyên, không dùng làm nguồn số liệu)
Viết trước khi notebook xong hoặc trước khi đổi phân công; cell, số liệu, phân công trong các file này có thể đã khác.

| Nhóm | File |
|---|---|
| Bản cũ của tài liệu hiện hành | `00_README_v1.md`, `_v2.md`; `presentation_description_v1.md`, `_v2.md`; `giai_thich_du_an_v1.md`–`_v3.md`; `parts/ban_giao_toan_v1.md`; `scripts/kich_ban_toan_v1.md`; `rules/*_v1.md`, `*_v2.md`; `templates/*_v1.md`, `RUN_LOG_TEMPLATE_v2.md`; `experiments/error_analysis_v1.md`; `models/*_v1.md` |
| Desc từng mục (16–17/11) | `parts/muc01_gioi_thieu_v1.md`, `_v2.md`, `muc02_kien_truc_v1.md`, `muc03_tokenization_v1.md`, `muc04_nguyen_ly_hoc_v1.md`, `muc05_giai_doan_huan_luyen_v1.md` |
| Kế hoạch, plan, hướng dẫn | `plans/` (trừ `yeu_cau_cua_co_v1.md`, `khung_presentation_llms_v5.md`) |
| Prompt làm slide, tìm video | `plans/prompt_anh_infographic_tung_slide_v1.md`, `_v2.md`, `prompt_gemini_slide_infographic_v1.md`, `prompt_tim_video_youtube_v1.md` |
| Research từng mục (14–17/11) | `research/` (tham khảo lý thuyết; nguồn đã chuyển vào `references.md`) |
| Model so sánh mục 2 (notebook mở rộng, không nộp) | `models/bert-base-multilingual-cased_v1.md`, `models/qwen2-vl-2b-instruct_v1.md` |
| File trao đổi, bàn giao giữa thành viên | Đã xoá 21/11, xem lịch sử git |

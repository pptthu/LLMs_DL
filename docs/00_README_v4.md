# Tài liệu bài Presentation LLMs — đọc file này trước (v4)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-28 | Trạng thái: **[đề xuất]**, chờ Toàn duyệt
- Thay đổi so với v3: chỉ giữ trên GitHub tài liệu của phần đã làm; nội dung từng mục chuyển vào `research/mucNN_*_v1.md`. Tài liệu cũ, kế hoạch, hướng dẫn, prompt làm slide giữ ở máy (xem lịch sử git cho bản đã commit trước 28/11).
- Lần chạy chính thức: **`20261128_041850_normal`** (Kaggle, GPU T4 x2, 22.0 phút, 188 cell; G-Eval chấm 20 lần lấy trung bình). Mọi số đọc từ `../outputs/results/` của lần chạy này, trừ VRAM 6.6 / 9.5 lấy run `20261127_234527_normal` (run mới đo lệch). Run `20261125_214650_normal` (giám khảo Qwen2.5-7B) dùng để so sánh giám khảo ở mục 10.

## Quy tắc version
- Tên file có hậu tố `_vN`; dùng bản số lớn nhất. Đổi nội dung thì tạo `_vN+1` sau khi Toàn duyệt (`rules/RULES_v3.md`).

## Research từng mục (lý thuyết, code từng cell, output thật, nhận xét, nguồn)

| Mục | File | Cell notebook |
|---|---|---|
| Đề, kiến thức nền, Setup | `research/muc00_setup_v1.md` | 0.1–0.8 |
| 1. Giới thiệu: LLM là gì, 4 tác vụ | `research/muc01_gioi_thieu_v1.md` | 1.1–1.6 |
| 2. Kiến trúc: decoder-only, causal attention | `research/muc02_kien_truc_v1.md` | 2.1–2.5 |
| 3. Tokenization | `research/muc03_tokenization_v1.md` | 3.1–3.5 |
| 4. Nguyên lý học: next-token prediction, cross-entropy | `research/muc04_nguyen_ly_hoc_v1.md` | 4.1–4.6 |
| 5. Giai đoạn huấn luyện: pretraining, SFT, alignment | `research/muc05_giai_doan_huan_luyen_v1.md` | 5.0–5.4 |
| 6. Sinh văn bản: greedy, temperature, top-p | `research/muc06_sinh_van_ban_v1.md` | 6.1–6.6 |
| 7. In-context learning: zero-shot vs few-shot | `research/muc07_in_context_learning_v1.md` | 7.1–7.5 |
| 8. Prompting, fine-tuning, RAG | `research/muc08_rag_v1.md` | 8.1–8.4 |
| 9. Hạn chế: hallucination, thiên lệch, ngữ cảnh | `research/muc09_han_che_v1.md` | 9.1–9.6 |
| 10. Thực nghiệm, bộ metric, kết luận | `research/muc10_thuc_nghiem_ket_luan_v1.md` | 10.1–10.7 |

## Tài liệu khác

| Cần gì | File |
|---|---|
| Yêu cầu của cô (đề, góp ý, yêu cầu thêm) | `plans/yeu_cau_cua_co_v1.md` |
| Tổng quan dự án: 10 mục → phần notebook, file vào / ra | `../project_description.md` |
| Desc cả bài, đánh giá, quyết định | `presentation_description_v3.md` |
| Tài liệu tham khảo (sản phẩm nộp số 4) | `references.md` |
| Sổ ghi mọi lần chạy | `experiments/experiment_log.md` |
| Phân tích lỗi mục 10 | `experiments/error_analysis_v2.md` |
| Model card (có sơ đồ kiến trúc) | `models/qwen2.5-1.5b-instruct_v2.md`, `models/qwen2.5-1.5b-base_v2.md`, `models/gpt2-tokenizer_v2.md` |
| Luật, phân vai | `rules/RULES_v3.md` |
| Cây thư mục, input / output lưu ở đâu | `rules/FOLDER_STRUCTURE_v3.md` |
| Khuôn (log, desc, model card, bug, ghi chú thí nghiệm) | `templates/` |
| Tiến độ, nhật ký | `../progress_status.md` |
| Khai báo dùng AI | `../AI_usage.md` |

## Thứ tự đọc
1. `plans/yeu_cau_cua_co_v1.md`
2. `research/muc00_setup_v1.md`
3. `research/muc01_…` → `muc10_…` theo thứ tự mục, mở notebook cùng lúc để đối chiếu từng cell

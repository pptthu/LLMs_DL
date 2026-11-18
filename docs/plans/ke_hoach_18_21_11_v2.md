# Kế hoạch 18/11 → 21/11 (trình bày thứ 7)

- **Created**: 2026-11-18 (Toàn, AI hỗ trợ) · **Last Updated**: 2026-11-19 (Toàn, AI hỗ trợ)
- Trạng thái: **Duyệt: Toàn, 2026-11-19**, theo [`chot_khung_v4_ke_hoach_cho_Toan.md`](chot_khung_v4_ke_hoach_cho_Toan.md) và [`khung_presentation_llms_v4.md`](khung_presentation_llms_v4.md) §4.
- Thay [`ke_hoach_18_21_11_v1.md`](ke_hoach_18_21_11_v1.md). v1 giữ nguyên.
- Chỉ làm đủ khung v4. Việc nào không phục vụ một ô trong khung đó thì không làm.

## 1. Không làm

- Mục 2: so sánh với mBERT, VLM, LSTM nằm ở `notebook/llm_extension_part02.ipynb` (không nộp); chỉ làm khi phần chính xong (file chốt §5).
- Mục 7, 10: nhiều bộ ví dụ few-shot, Bonferroni (chỉ 1 phép so sánh).
- Mục 8: RAGAS, chấm mù, Cohen's kappa.
- Mục 9: demo thiên lệch và demo giới hạn ngữ cảnh (trình bày bằng lời).

## 2. Notebook: trạng thái từng phần (19/11)

| Phần | Slide | Trạng thái | Việc còn lại |
|---|---|---|---|
| 0, 1 | 2, 3 | smoke đạt (`20261117_161656_smoke`) | — |
| 2 | 4 | smoke đạt; đã bỏ cell 2.5–2.10 | Smoke lại cell 2.5 tổng kết mới |
| 3, 4 | 5, 6 | smoke đạt | — |
| 5 | 7 | chạy hết, **chưa chấm** (`MANUAL_JUDGE5` trống) | Toàn chấm 8 dòng theo `follow_rule` |
| 6 | 8 | code xong, thử CPU đạt | Smoke run Kaggle |
| 7 | 9, 12 | code xong, thử CPU đạt | Smoke run: đọc dòng "Ước tính … phút" ở cell 7.5 để chốt `N_TEST` |
| 8 | 10 | code xong, thử CPU đạt | Smoke run; đọc `rag_questions_used.csv`, loại câu mơ hồ vào `VIQUAD_EXCLUDE_IDS` |
| 9 | 11 | code xong, thử CPU đạt | Smoke run; nếu không có câu sai thì dùng dự phòng (SimpleQA hỏi bằng tiếng Việt) |
| 10 | 12, 13 | code xong, thử CPU đạt | Sau normal run: gán nhóm lỗi (`MANUAL_ERROR_GROUPS`), viết `docs/experiments/error_analysis_v1.md` |

"Thử CPU" = chạy cả notebook trên CPU bằng model Qwen2 tí hon (trọng số ngẫu nhiên), tokenizer và dữ liệu thật: chỉ kiểm code chạy hết và qua `assert`, không có số liệu.

## 3. Lịch

Quy tắc cắt: **12h trưa thứ 6**, phần nào chưa smoke run đạt thì slide mục đó chỉ còn lý thuyết + minh họa của đề.

| Ngày | Toàn (code, dữ liệu, chấm tay mọi phần) | Muội | Phúc | Thư |
|---|---|---|---|---|
| **T5 19/11** | Smoke run phần 0–10 trên Kaggle; sửa lỗi; chấm phần 5; chốt `N_TEST`; **tối:** Save & Run All chính thức (`SMOKE = False`), tải `outputs/` + notebook đã chạy về repo | Slide mục 1, 2, 3 | Slide mục 4, 5, 6 | Review notebook |
| **T6 20/11** | Sáng: chạy lại nếu lần tối thứ 5 lỗi. **12h: cắt.** Chiều: phân tích lỗi, bảng kết quả, slide 2, 12, 13; bàn giao code | Slide mục 8; thay số chính thức; README; ghép slide (**20h**) | Slide mục 9; thay số chính thức | Review slide |
| **T6 tối** | Tập toàn bài, bấm giờ | ← | ← | Bấm giờ, hỏi thử từng dòng code |
| **T7 21/11** sáng | Xuất PDF; nộp 4 sản phẩm (khung v4 §3) | Tập phần mình | Tập phần mình | Kiểm checklist |

- Muội, Phúc: chỉ làm slide và thuyết trình mục mình; vẫn phải giải thích được code phần mình sau khi Toàn bàn giao (yêu cầu thêm 5). [cần chốt: Toàn xác nhận phân vai slide]
- Mỗi người nộp 5 câu Q&A kèm đáp án cho mục mình trước **18h thứ 6**.

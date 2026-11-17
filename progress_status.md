# Tiến độ bài Presentation LLMs

- **Created**: 2026-11-15T16:30:00+07:00 (Toàn)
- **Last Updated**: 2026-11-17T11:25:29+07:00 (Thư)

Cập nhật file này sau MỖI bước. Bảng dưới đây là nguồn thông tin duy nhất về việc ai đang làm gì.

Code mọi phần: **Toàn** (quyết định 15/11). Cột "Lý thuyết + thuyết trình" là người phải giải thích được code phần đó sau khi bàn giao.

| Phần notebook | Mục | Lý thuyết + thuyết trình | Desc (`docs/parts/`) | Code (chưa làm / đang làm / xong) | Smoke run (đạt / lỗi / chưa chạy) | Normal run (đạt / lỗi / chưa chạy) | Đã bàn giao code | Slide | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| 0. Setup | — | Toàn (cả nhóm phải hiểu) | — | xong | đạt (16/11, Kaggle T4) | chưa chạy | chưa | 2 | Ưu tiên số 1: cách lấy model / tokenizer / dữ liệu |
| 1. Tác vụ mẫu | 1 | Muội (demo: Toàn) | `muc01_gioi_thieu_v2.md` (duyệt 16/11) | xong | đạt (16/11, Kaggle T4, 4/4) | chưa chạy | chưa | 3–4 | |
| 2. Kiến trúc | 2 | Muội | chưa có | chưa làm | chưa chạy | chưa chạy | chưa | 5–6 | Cần `attn_implementation="eager"` |
| 3. Tokenization | 3 | Muội | chưa có | chưa làm | chưa chạy | chưa chạy | chưa | 7 | |
| 4. Nguyên lý học | 4 | Phúc | chưa có | chưa làm | chưa chạy | chưa chạy | chưa | 8–9 | |
| 5. Base vs instruct | 5 | Phúc | chưa có | chưa làm | chưa chạy | chưa chạy | chưa | 10–11 | Chờ cô xác nhận |
| 6. Sinh văn bản | 6 | Phúc | chưa có | chưa làm | chưa chạy | chưa chạy | chưa | 12–13 | |
| 7. Zero / few-shot | 7 | Toàn | chưa có | chưa làm | chưa chạy | chưa chạy | — | 14 | Chờ cô xác nhận UIT-VSFC |
| 8. RAG | 8 | Muội | chưa có | chưa làm | chưa chạy | chưa chạy | chưa | 15–16 | Chờ cô xác nhận |
| 9. Hạn chế | 9 | Phúc | chưa có | chưa làm | chưa chạy | chưa chạy | chưa | 17–18 | |
| 10. Tổng hợp | 10 | Toàn | chưa có | chưa làm | chưa chạy | chưa chạy | — | 19–20 | |
| Save & Run All (chính thức) | — | Toàn | — | chưa làm | — | chưa chạy | — | — | |
| README | — | Muội | — | khung (Toàn) | — | — | — | — | |
| Ghép slide | — | Muội | — | chưa làm | — | — | — | 1, 2, 21 | |

## Bàn giao code cho người thuyết trình (mỗi phần)
Toàn chỉ đánh dấu "Đã bàn giao" khi đủ cả 4 điều kiện:
- [ ] Phần đó smoke run đạt, ghi đủ file kết quả theo `project_description.md` §2.3
- [ ] Có markdown, comment, cell "Phân tích" theo `docs/rules/RULES_v2.md` §2
- [ ] Toàn đã giải thích từng cell cho người phụ trách mục (gặp trực tiếp hoặc gọi), chỉ file kết quả và số cho slide
- [ ] Người phụ trách mục tự giải thích lại được 3 cell bất kỳ do Thư chọn

## Kiểm chứng
- 2026-11-16 smoke run phần 0 (`20261116_025328_smoke`, Kaggle T4): đạt; tóm tắt trong `docs/experiments/experiment_log.md`.

Log mỗi lần chạy ở `outputs/logs/<YYYYMMDD_HHMMSS>_<smoke|normal>/` theo `docs/templates/RUN_LOG_TEMPLATE_v2.md`; tóm tắt ở `docs/experiments/experiment_log.md`.

## Vấn đề mở
- Chờ cô xác nhận: cách demo mục 5, mục 8; dùng UIT-VSFC.
- Cần chốt: không còn (mục §8 của `docs/plans/plan_presentation_llm_v3.md` đã chốt hết ngày 16/11). `AI_usage.md` để sau.

## Nhật ký (ghi ngày, mỗi bước một dòng)
- 2026-11-15 Toàn: clone repo `pptthu/LLMs_DL` (repo trống) về `D:\File\File\UTH\UTH\Học sâu - DL\LLMs_DL`; tạo khung thư mục và tài liệu theo rule Lab 1 đã sửa cho bài presentation (AI hỗ trợ) → commit `578300a`, chưa push, chưa chạy code; chờ Thư duyệt
- 2026-11-15 Toàn: quyết định Toàn viết toàn bộ code notebook (phần 0–10); Muội, Phúc làm lý thuyết, slide, thuyết trình mục mình và nhận bàn giao code → sửa `docs/rules/RULES_v1.md` §4, `docs/presentation_description_v1.md`, `project_description.md`, file này, `AGENTS.md`, `CONTRIBUTING.md`, `README.md`
- 2026-11-16 Toàn: quyết định Toàn duyệt 4 cổng, Thư review; log mỗi lần chạy thay cho log huấn luyện (không fine-tune); thêm so sánh LLM với RNN/LSTM, BERT, VLM vào mục 2 → tạo bản mới (v1 giữ nguyên): `docs/00_README_v2.md`, `docs/presentation_description_v2.md`, `docs/rules/RULES_v2.md`, `docs/rules/FOLDER_STRUCTURE_v2.md`, `docs/templates/{DESC,PART,RUN_LOG,MODEL_CARD}_TEMPLATE_v2.md`, `docs/models/*_v2.md` (sơ đồ kiến trúc), `docs/plans/ke_hoach_presentation_llms_v2.md`, `khung_presentation_llms_v2.md`, `plan_presentation_llm_v3.md`, `docs/research/noi_dung_muc02/03/04/06/08/09_*_v2.md`; sửa `AGENTS.md`, `CONTRIBUTING.md`, `README.md`, `project_description.md`, `docs/references.md`, `docs/experiments/experiment_log.md`, mẫu PR/issue (AI hỗ trợ) → chỉ sửa tài liệu, chưa commit, chờ Toàn duyệt
- 2026-11-16 Toàn: thêm hướng dẫn từng bước `docs/plans/huong_dan_tung_buoc_v1.md` (quy trình, smoke/normal run trên Kaggle, duyệt và commit) và `docs/plans/detail_plan_notebook_v1.md` (từng bước code phần 0–10) (AI hỗ trợ) → chưa commit, chờ Toàn duyệt
- 2026-11-16 Toàn: duyệt toàn bộ bản v2 (rules, templates, description, plans, detail plan, hướng dẫn, model card, thiết kế demo mục 2); chốt Bonferroni mục 10, parse nhận nhãn Việt + thử nhiều mẫu prompt trên validation, repo `pptthu/LLMs_DL` (chủ: Thư), Ngày 1 = 15/11; danh sách file `data/` của Muội, Phúc → chưa commit
- 2026-11-16 Toàn: chốt cách chấm mục 8, 9 (SQuAD, RGB, RAGAS, SimpleQA, FActScore; 2 người chấm mù, % đồng ý + kappa) → ghi vào `research/noi_dung_muc08_*_v2.md` §3.4, `noi_dung_muc09_*_v2.md` §5.1b, `presentation_description_v2.md`, `detail_plan_notebook_v1.md`, `references.md` → chưa commit
- 2026-11-16 Toàn: chốt phần 0 Setup (lý thuyết + code) do Toàn; mục 2 phải có demo so sánh (thiết kế [đề xuất]: mBERT vs Qwen, Qwen2.5 vs Qwen2-VL-2B, +1 slide 6b); thêm "How to do: từng bước" vào `RULES_v2` §1.1, `FOLDER_STRUCTURE_v2` §7 và 4 template v2; thêm 2 model card; xoá `_to_delete/` → chưa commit, chờ Toàn duyệt
- 2026-11-16 Toàn: phần 0 Setup: viết 8 cell code 0.1–0.8 (cấu hình, log, môi trường, tải Qwen2.5-1.5B-Instruct + GPT-2 tokenizer, kiến trúc + assert số tham số, UIT-VSFC, env.json/config.json) theo `docs/plans/detail_plan_notebook_v1.md` (AI hỗ trợ), nhánh `part/00-setup` → chạy thử cục bộ cell 0.1, 0.2, 0.7 (dữ liệu giả), 0.8 không có GPU: đạt; **chưa smoke run trên Kaggle**
- 2026-11-16 Toàn: smoke run phần 0 trên Kaggle (T4 x2, `20261116_025328_smoke`): đạt (số tham số 1,543,714,304 khớp; UIT-VSFC khớp số dòng). Sau lần chạy: cell 0.4 đổi `device_map={"": 0}`, cell 0.3 thêm `gpu_count`, điền cell Phân tích (AI hỗ trợ) → chưa commit
- 2026-11-16 Toàn: smoke run lại phần 0 (`20261116_030403_smoke`): revision `989aa79…` và `gpu_count` = 2 ghi đúng (AI hỗ trợ ghi log) → chưa commit
- 2026-11-16 Toàn: desc mục 1 `docs/parts/muc01_gioi_thieu_v1.md` và `data/tasks_prompts.json` (AI viết theo yêu cầu Toàn) → Toàn duyệt; Toàn làm demo phần 1; dữ liệu `data/` lên Kaggle bằng Kaggle Dataset (cách (a))
- 2026-11-16 Toàn: code phần 1 (cell 1.1–1.5: hàm sinh greedy, 4 tác vụ in rõ INPUT/OUTPUT, chạy `is_even` của model, chấm + cột `truncated`, thử prompt khác) (AI hỗ trợ); `tasks_prompts.json` v2 (trần token 256/512), desc `muc01_gioi_thieu_v2.md`; smoke run đạt 4/4
- 2026-11-17 Thư: review repo (main tới PR #3) và notebook phần 0–1; ghi góp ý vào `docs/plans/gop_y_thu_cho_toan_v1.md` (AI hỗ trợ): §1 việc Thư đã chốt, §2 cần bàn (cách chấm mục 8–9, Bonferroni, chọn prompt trên validation), §3 đối chiếu yêu cầu của cô theo 10 mục, §4 demo B mục 2 và 3 phương án, §5 câu hỏi về docs/code/git → chờ Toàn trả lời; chưa sửa file nào khác của Toàn

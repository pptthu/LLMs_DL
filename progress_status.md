# Tiến độ bài Presentation LLMs

- **Created**: 2026-11-15T16:30:00+07:00 (Toàn)
- **Last Updated**: 2026-11-19T02:30:00+07:00 (Toàn, AI hỗ trợ)

Cập nhật file này sau MỖI bước. Bảng dưới đây là nguồn thông tin duy nhất về việc ai đang làm gì.

Code mọi phần: **Toàn** (quyết định 15/11). Cột "Lý thuyết + thuyết trình" là người phải giải thích được code phần đó sau khi bàn giao.

| Phần notebook | Mục | Lý thuyết + thuyết trình | Desc (`docs/parts/`) | Code (chưa làm / đang làm / xong) | Smoke run (đạt / lỗi / chưa chạy) | Normal run (đạt / lỗi / chưa chạy) | Đã bàn giao code | Slide | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| 0. Setup | — | Toàn (cả nhóm phải hiểu) | — | xong | đạt (16/11, Kaggle T4) | đạt (19/11, `20261119_025437_normal`) | chưa | 2 | Ưu tiên số 1: cách lấy model / tokenizer / dữ liệu |
| 1. Tác vụ mẫu | 1 | Muội (demo: Toàn) | `muc01_gioi_thieu_v2.md` (duyệt 16/11) | xong | đạt (16/11, Kaggle T4, 4/4) | đạt (19/11, `20261119_025437_normal`) | chưa | 3–4 | |
| 2. Kiến trúc | 2 | Muội | `muc02_kien_truc_v1.md` (nháp AI 17/11, chờ người phụ trách sửa, Toàn duyệt) | xong; 19/11 bỏ cell 2.5–2.10 (sang `llm_extension_part02.ipynb`), thêm 2.5 tổng kết | đạt 2.1–2.4 (17/11, `20261117_161656_smoke`); 2.5 mới đạt (19/11, `20261119_021425_smoke`) | đạt (19/11, `20261119_025437_normal`) | chưa | 5–6 | Cell 2.8 (VLM), 2.9 (RNN) chờ Thư chốt |
| 3. Tokenization | 3 | Muội | `muc03_tokenization_v1.md` (nháp AI 17/11, chờ người phụ trách sửa, Toàn duyệt) | xong | đạt (17/11, `20261117_161656_smoke`) | đạt (19/11, `20261119_025437_normal`) | chưa | 7 | |
| 4. Nguyên lý học | 4 | Phúc | `muc04_nguyen_ly_hoc_v1.md` (nháp AI 17/11, chờ người phụ trách sửa, Toàn duyệt) | xong | đạt (17/11, `20261117_161656_smoke`) | đạt (19/11, `20261119_025437_normal`) | chưa | 8–9 | |
| 5. Base vs instruct | 5 | Phúc | `muc05_giai_doan_huan_luyen_v1.md` (nháp AI 17/11, chờ người phụ trách sửa, Toàn duyệt) | xong | chạy hết (17/11), **chưa chấm** → chưa đạt | đạt (19/11, `20261119_025437_normal`) | chưa | 10–11 | `data/base_vs_instruct.json` nháp AI, chờ Phúc duyệt; `repetition_penalty` 1.0 chờ Thư chốt |
| 6. Sinh văn bản | 6 | Phúc | chưa có | xong (19/11, thử CPU đạt) | đạt (19/11, `20261119_021425_smoke`) | đạt (19/11, `20261119_025437_normal`) | chưa | 12–13 | |
| 7. Zero / few-shot | 7 | Toàn | chưa có | xong (19/11, thử CPU đạt) | đạt (19/11, `20261119_021425_smoke`) | đạt (19/11, `20261119_025437_normal`) | — | 14 | Chờ cô xác nhận UIT-VSFC |
| 8. RAG | 8 | Muội | chưa có | xong (19/11, UIT-ViQuAD 2.0, thử CPU đạt) | đạt (19/11, `20261119_021425_smoke`) | đạt (19/11, `20261119_025437_normal`) | chưa | 15–16 | Chờ cô xác nhận |
| 9. Hạn chế | 9 | Phúc | chưa có | xong (19/11, UIT-ViQuAD 2.0, thử CPU đạt) | đạt (19/11, `20261119_021425_smoke`) | đạt (19/11, `20261119_025437_normal`) | chưa | 17–18 | |
| 10. Tổng hợp | 10 | Toàn | chưa có | xong (19/11, thử CPU đạt) | đạt (19/11, `20261119_021425_smoke`) | đạt (19/11, `20261119_025437_normal`) | — | 19–20 | |
| Save & Run All (chính thức) | — | Toàn | — | xong | — | đạt (19/11, `20261119_025437_normal`, 7.1 phút) | — | — | |
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
- 2026-11-17 Toàn: trả lời góp ý của Thư → `docs/plans/tra_loi_toan_cho_thu_v1.md` (AI hỗ trợ): §A 9 điểm góp ý chưa nêu (bài dài 22 phút, tiến độ trễ, mục 5 khác repetition_penalty/chat template/token kết thúc, chưa có người thứ 2 ở mục 8–9, mục 9 thiếu mốc văn bản ngắn…), §B–§H trả lời từng ý, §I 10 câu cần Thư quyết → chờ Thư; chưa sửa file nào khác
- 2026-11-17 Toàn: code phần 2–5 trên nhánh `part/02-05-arch-token-loss-base` (AI hỗ trợ, Claude Code): phần 2 cell 2.1–2.10 (config + tham số theo thành phần, causal mask, attention eager, top-5, demo A mBERT vs Qwen + phép thử đổi token cuối 'đẹp'/'tối', VLM đọc cấu trúc Qwen2-VL-2B trên meta [cần chốt], LSTM vs Transformer đo thời gian [cần chốt]); phần 3 cell 3.1–3.5; phần 4 cell 4.1–4.5 (thêm `results/loss_summary.csv`); phần 5 cell 5.1–5.4 (`repetition_penalty` 1.0 cả 2 bản [cần chốt]); `data/base_vs_instruct.json` v1 (nháp AI, chờ Phúc duyệt) → chạy thử cục bộ trên CPU bằng model Qwen2 tí hon (trọng số ngẫu nhiên) + tokenizer, mBERT, config VLM thật: hết phần 0, 2–5 không lỗi; **chưa smoke run trên Kaggle**, chưa commit; desc mục 2–5 chưa có
- 2026-11-17 Toàn: desc mục 2–5 `docs/parts/muc02_kien_truc_v1.md`, `muc03_tokenization_v1.md`, `muc04_nguyen_ly_hoc_v1.md`, `muc05_giai_doan_huan_luyen_v1.md` (AI viết nháp theo yêu cầu Toàn, sau khi có code) → chờ Muội (2, 3), Phúc (4, 5) sửa và ký, Toàn duyệt; chưa commit
- 2026-11-18 Toàn: `docs/plans/yeu_cau_cua_co_v1.md` (chỉ yêu cầu của cô: đề, góp ý 11/11, yêu cầu thêm, sản phẩm nộp); `khung_presentation_llms_v3.md` (14 slide, ~19.5 phút, kèm khung của Thư để đối chiếu); `ke_hoach_18_21_11_v1.md` (lịch 18–21/11; giữ demo mục 5, 8 theo góp ý 11/11) (AI hỗ trợ) → [đề xuất], chờ Toàn duyệt; chưa commit
- 2026-11-18 Toàn: `docs/plans/khung_presentation_llms_v4.md` (14 slide, chỉ yêu cầu của cô: nội dung + minh họa chép nguyên đề, demo theo yêu cầu thêm 3 và góp ý 11/11; bỏ phút, bỏ khung của Thư) và `ke_hoach_18_21_11_v2.md` (bỏ cell 2.5–2.10, McNemar, Bonferroni, RAGAS, kappa, demo thiên lệch và ngữ cảnh) (AI hỗ trợ) → [đề xuất], chờ Toàn duyệt; chưa commit
- 2026-11-19 Toàn: duyệt file chốt khung v4 của Thư kèm 9 điểm sửa (lịch dời 1 ngày, sinh theo lô chạy đủ 2,999 câu test, few-shot 4 ví dụ, chọn prompt trên 200 câu validation, lưu id câu, phân tích lỗi 30 câu few-shot, mục 6 seed + 3 lần, mục 9 dẫn lỗi lẫn tiếng Trung, Toàn chấm phần 5); mục 8, 9 dùng UIT-ViQuAD 2.0 thay 3 file soạn tay; Toàn code, soạn dữ liệu, chấm tay mọi phần → sửa `khung_presentation_llms_v4.md`, `ke_hoach_18_21_11_v2.md`, `references.md`, `experiment_log.md` (ghi smoke `20261117_161656_smoke` phần 0–5)
- 2026-11-19 Toàn: notebook: cell 0.1 thêm tham số phần 7–10; phần 2 bỏ cell 2.5–2.10 (chuyển sang `notebook/llm_extension_part02.ipynb`, không nộp), thêm 2.5 tổng kết; code phần 6 (6.1–6.4), 7 (7.1–7.5), 8 (8.1–8.4), 9 (9.1–9.2), 10 (10.1–10.4) (AI hỗ trợ, Claude Code) → chạy thử cả notebook trên CPU bằng model Qwen2 tí hon (trọng số ngẫu nhiên) + tokenizer, dữ liệu thật: hết 125 cell, qua mọi `assert`; **chưa smoke run trên Kaggle**, chưa commit
- 2026-11-19 Toàn: smoke run phần 0–10 trên Kaggle (`20261119_021425_smoke`): đạt, 2.3 phút; ước tính phần 7 chạy toàn bộ 2,999 câu mất 3.1 phút → giữ `N_TEST = None`; phần 5 còn chưa chấm; cell 7.4 sửa ghi log (`float`) sau lần chạy → chưa commit
- 2026-11-19 Toàn: normal run `20261119_022224_normal` (7.2 phút, đạt); tải `outputs/` về repo. Phần 7: zero-shot 0.8986, few-shot 0.9230, McNemar p = 1.5e-08 (2,999 câu). Phần 8, 9: chấm tự động sai vài câu → sửa cell 8.4 (dấu câu → khoảng trắng), thêm `MANUAL_JUDGE8`, `MANUAL_JUDGE9` (đề xuất AI, Toàn đọc lại) → chờ chạy lại normal run; chưa commit
- 2026-11-19 Toàn: chạy lại normal run `20261119_023934_normal` (7.1 phút, đạt) sau khi sửa chấm phần 8, 9; đây là lần chạy chính thức, số cho slide lấy từ lần này. Mục 7: 0.8986 / 0.9230, p = 1.5e-08; mục 8: không RAG 0/15, có RAG 10/15; mục 9: sai 12/15 mỗi cách hỏi → chưa commit
- 2026-11-19 Toàn: điền cell Phân tích phần 0–10 bằng số của `20261119_023934_normal`; gán nhóm lỗi 30 câu (`MANUAL_ERROR_GROUPS`, [đề xuất] AI, chờ Toàn duyệt); cell 10.3 thêm đếm câu góp ý; viết `docs/experiments/error_analysis_v1.md` (AI hỗ trợ) → cell 10.3 đổi code nên **cần một lần Save & Run All cuối** để output và `error_analysis_sample.csv` khớp; chưa commit
- 2026-11-19 Toàn: Save & Run All cuối `20261119_025437_normal` (7.1 phút, đạt; số trùng lần trước); notebook nộp và `outputs/` lấy từ lần này → chưa commit

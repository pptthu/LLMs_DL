# Hướng dẫn từng bước: quy trình, chạy trên Kaggle, duyệt và commit (v1)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-16T02:00:00+07:00 | **Duyệt: Toàn, 2026-11-16**
- Đi kèm: `rules/RULES_v2.md` (luật), `plans/detail_plan_notebook_v1.md` (từng bước code phần 0–10), `templates/RUN_LOG_TEMPLATE_v2.md` (log).
- Chỗ ghi **[kiểm khi làm]** là tên nút trên giao diện Kaggle / GitHub; AI chưa mở kiểm, giao diện có thể khác.

---

## A. Quy trình làm một phần

Đã chuyển vào `rules/RULES_v2.md` §1.1 (8 bước, mỗi bước có các bước con).

---

## B. Chạy trên Kaggle

### B0. Chuẩn bị (một lần)
1. Tài khoản Kaggle đã xác minh số điện thoại (cần cho GPU và Internet).
2. Tạo notebook mới hoặc import `notebook/llm_presentation.ipynb` từ máy / GitHub **[kiểm khi làm]**.
3. Settings → Accelerator → **GPU T4**; Settings → **Internet: On** **[kiểm khi làm]**.
4. (Phần 8) Upload slide môn học thành Kaggle Dataset **riêng tư**, Add Data vào notebook.

### B1. Smoke run
1. Cell 0.1: đặt `SMOKE = True`.
2. Run All (hoặc chạy riêng phần đang làm, nhưng luôn chạy phần 0 trước).
3. Kiểm thư mục `outputs/_smoke/logs/<YYYYMMDD_HHMMSS>_smoke/`:
   - [ ] `run.log` có `RUN START` và `RUN END`, không có dòng lỗi
   - [ ] `config.json`, `env.json` có đủ phiên bản và tham số
   - [ ] mỗi phần có `partNN_<tên>.log` (đủ phần đầu / mỗi mẫu / cuối) và `_records.csv` đúng cột
   - [ ] `outputs/_smoke/results/` có đủ file của phần, đúng cột theo `project_description.md` §2.3
   - [ ] mọi `assert` qua
4. Lỗi thì sửa, chạy lại; lỗi khó ghi `docs/bugs/<YYYYMMDD>_<tên>.md` theo `BUG_REPORT_TEMPLATE_v1.md`.
5. Thêm 1 mục "smoke run" vào `docs/experiments/experiment_log.md` (ngày, phần, thư mục log, đạt / lỗi).
6. Số của smoke run **không** đưa lên slide.

### B2. Normal run (chạy chính thức)
1. Điều kiện: mọi phần đã smoke run đạt; prompt mục 7 đã chốt trên split validation.
2. Cell 0.1: đặt `SMOKE = False`.
3. Bấm **Save Version → Save & Run All (Commit)** **[kiểm khi làm]**. Ghi giờ bắt đầu.
4. Chạy xong, mở tab Output của version **[kiểm khi làm]**, tải về:
   - thư mục `outputs/logs/<YYYYMMDD_HHMMSS>_normal/`
   - `outputs/results/`, `outputs/figures/`
   - notebook đã chạy (có output)
5. Chép vào repo trên máy: thư mục `outputs/` tương ứng; notebook đè `notebook/llm_presentation.ipynb`.
6. Kiểm như checklist ở B1 bước 3, nhưng trong `outputs/logs/` và `outputs/results/`.
7. Mở notebook đã chạy: không có đường dẫn cá nhân hay token trong output.
8. Xuất HTML dự phòng (File → Download / Export HTML **[kiểm khi làm]**).
9. Ghi 1 mục "normal run" vào `docs/experiments/experiment_log.md`: `run_id`, thời gian chạy, file kết quả, kết quả chính (số lấy từ file).
10. Điền `README.md` mục "Thời gian chạy dự kiến"; ghim phiên bản trong `requirements.txt` theo `env.json`.
11. Sửa code sau bước này thì **chạy lại toàn bộ** (thư mục log mới) rồi mới cập nhật slide.

---

## C. Duyệt và commit các file tài liệu ngày 16/11

### C1. Duyệt (Toàn đọc, theo thứ tự)
1. `docs/rules/RULES_v2.md`
2. `docs/rules/FOLDER_STRUCTURE_v2.md`
3. `docs/templates/RUN_LOG_TEMPLATE_v2.md`, `DESC_TEMPLATE_v2.md`, `PART_TEMPLATE_v2.md`, `MODEL_CARD_TEMPLATE_v2.md`
4. `docs/presentation_description_v2.md`
5. `docs/plans/detail_plan_notebook_v1.md`, file này
6. `docs/models/*_v2.md`
7. `docs/plans/ke_hoach_presentation_llms_v2.md`, `khung_presentation_llms_v2.md`, `plan_presentation_llm_v3.md`
8. `docs/research/noi_dung_muc02_kien_truc_v2.md` (§11 so sánh), các `noi_dung_muc03/04/06/08/09_*_v2.md`
9. File gốc đã sửa: `git diff` từng file (`AGENTS.md`, `CONTRIBUTING.md`, `README.md`, `project_description.md`, `progress_status.md`, `AI_usage.md`, `docs/references.md`, `docs/experiments/experiment_log.md`, `.github/...`)

Mỗi file: chỗ nào không đồng ý thì sửa trực tiếp (file chưa commit, chưa ai dùng) hoặc ghi lại để nhờ AI sửa. Chỗ `[cần chốt]` thì quyết rồi ghi vào mục "Quyết định" của `presentation_description_v2.md`.

### C2. Dọn
1. (Đã xoá `_to_delete/` ngày 16/11.)
2. `git status` → chỉ còn các file của bạn; `docs/drafts/*` để riêng, không gộp vào commit này.

### C3. Commit (Git Bash hoặc PowerShell, trong thư mục `LLMs_DL`)
```bash
git checkout main
git pull                                   # nếu repo đã có trên GitHub
git checkout -b docs/rules-v2              # nhánh theo CONTRIBUTING: <type>/<chủ đề>

# Commit 1: luật, cấu trúc thư mục, khuôn
git add docs/rules/RULES_v2.md docs/rules/FOLDER_STRUCTURE_v2.md
git add docs/templates/DESC_TEMPLATE_v2.md docs/templates/PART_TEMPLATE_v2.md docs/templates/RUN_LOG_TEMPLATE_v2.md docs/templates/MODEL_CARD_TEMPLATE_v2.md
git diff --staged --stat
git commit -m "docs(repo): add v2 rules, folder structure and templates"

# Commit 2: description, plan, hướng dẫn
git add docs/00_README_v2.md docs/presentation_description_v2.md
git add docs/plans/plan_presentation_llm_v3.md docs/plans/ke_hoach_presentation_llms_v2.md docs/plans/khung_presentation_llms_v2.md
git add docs/plans/detail_plan_notebook_v1.md docs/plans/huong_dan_tung_buoc_v1.md
git commit -m "docs(repo): add description v2, plan v3 and step-by-step guide"

# Commit 3: model card có sơ đồ
git add docs/models/qwen2.5-1.5b-instruct_v2.md docs/models/qwen2.5-1.5b-base_v2.md docs/models/gpt2-tokenizer_v2.md
git add docs/models/bert-base-multilingual-cased_v1.md docs/models/qwen2-vl-2b-instruct_v1.md
git commit -m "docs(repo): add model cards with architecture diagrams"

# Commit 4: research
git add docs/research/noi_dung_muc02_kien_truc_v2.md docs/research/noi_dung_muc03_tokenization_v2.md docs/research/noi_dung_muc04_nguyen_ly_hoc_v2.md docs/research/noi_dung_muc06_sinh_van_ban_v2.md docs/research/noi_dung_muc08_prompting_finetuning_rag_v2.md docs/research/noi_dung_muc09_han_che_danh_gia_v2.md docs/research/noi_dung_toan_muc07_muc10_v2.md docs/references.md
git commit -m "docs(repo): add model comparison to part 2 and drop src references"

# Commit 5: file gốc và mẫu GitHub
git add notebook/llm_presentation.ipynb AGENTS.md CONTRIBUTING.md README.md project_description.md progress_status.md AI_usage.md docs/experiments/experiment_log.md .github/PULL_REQUEST_TEMPLATE.md .github/ISSUE_TEMPLATE/part_task.md
git commit -m "docs(repo): point repo docs to v2 files and assign all code to toan"

git log --oneline -5
git push -u origin docs/rules-v2
```
- Trước mỗi `git commit`: `git diff --staged` xem lại, đúng file mới commit.
- Điền `[Toàn điền]` trong `AI_usage.md` trước commit 5 (phần này phải do bạn tự viết).

### C4. Pull Request
1. Trên GitHub mở PR từ `docs/rules-v2` vào `main` theo mẫu `.github/PULL_REQUEST_TEMPLATE.md` **[kiểm khi làm]**.
2. Tiêu đề: `docs(repo): v2 rules, templates, description and step-by-step guide`.
3. Thư review, góp ý trong PR; Toàn sửa nếu cần, rồi merge.
4. Ghi 1 dòng vào nhật ký `progress_status.md` (số PR, đã merge).

# Quy tắc Git

Dựa trên `Lab-DL/CONTRIBUTING.md`, sửa cho repo bài presentation (không có module code, chia theo phần notebook).

- **Nhánh:**
  - tạo từ `main`;
  - làm một phần notebook: `part/<NN>-<tên>` (ví dụ `part/07-icl`, `part/00-setup`);
  - việc khác: `<type>/<chủ đề>` (ví dụ `docs/references`, `chore/scaffold`).
  - Không commit trực tiếp vào `main`, trừ commit khung ban đầu.
- **Commit:** mỗi bước logic một commit, định dạng Conventional Commits `<type>(<phạm vi>): <việc đã làm>` (bảng dưới). Tiêu đề PR dùng cùng định dạng.
- **Stage từng file:** `git add <file>`; xem `git status` và `git diff --staged` trước khi commit.
- **Không commit:** dữ liệu UIT-VSFC, trọng số model, slide môn học, `outputs/_smoke/`, token, `.env` (`.gitignore` đã chặn phần lớn). Không force-push nhánh dùng chung.
- **Notebook:** chỉ Toàn sửa `notebook/llm_presentation.ipynb` (Toàn viết toàn bộ code, quyết định 15/11). Thành viên khác góp ý qua issue hoặc PR review, không sửa trực tiếp notebook. Trước khi commit notebook, kiểm output không chứa token hay đường dẫn cá nhân.
- **Pull Request:** mở PR vào `main` theo mẫu, ghi `Closes #<số issue>`; Thư review (góp ý), Toàn duyệt rồi merge.
- **Trước khi mở PR:**
  - chạy lại phần notebook liên quan (smoke run đạt, có log trong `outputs/_smoke/logs/`);
  - cập nhật `progress_status.md`;
  - ghi việc dùng AI vào `AI_usage.md`.
- **Lỗi khó:** ghi theo `docs/templates/BUG_REPORT_TEMPLATE_v1.md` vào `docs/bugs/`.

## Tin nhắn commit

Định dạng: `<type>(<phạm vi>): <việc đã làm>`.
- Phạm vi: `part<NN>` cho phần notebook (ví dụ `part07`), hoặc `docs`, `slides`, `data`, `repo`.
- Mô tả: tiếng Anh, dạng mệnh lệnh, chữ thường, không chấm cuối, khoảng 72 ký tự trở xuống.

| Type | Dùng khi | Ví dụ |
|---|---|---|
| `feat` | thêm code / phân tích mới trong notebook | `feat(part07): add zero-shot and 3-shot prompt builder` |
| `fix` | sửa lỗi (kết quả đang sai) | `fix(part07): pass do_sample=False for greedy decoding` |
| `docs` | chỉ tài liệu: `*.md`, chữ markdown trong notebook | `docs(part07): add analysis cells for icl results` |
| `data` | thêm hoặc sửa file trong `data/` | `data(part09): add 20 hallucination questions with sources` |
| `chore` | khung, cấu hình, `.gitignore`, `requirements.txt` | `chore(repo): scaffold folders and docs from lab1 rules` |
| `slides` | thêm hoặc sửa slide | `slides: merge parts into v1 deck` |

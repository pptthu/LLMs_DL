# Hướng dẫn cho AI agent

Dựa trên `Lab-DL/AGENTS.md` và `Lab-DL/docs/agent/dl-lab-workflow/SKILL.md`, sửa cho repo bài presentation.

1. Đọc `docs/00_README_v2.md`, `docs/rules/RULES_v2.md` và `docs/rules/FOLDER_STRUCTURE_v2.md` trước mọi việc, rồi làm theo.
2. Đọc `project_description.md`, `progress_status.md`, desc của mục đang làm trong `docs/parts/` (nếu có) và nội dung trong `docs/research/`.
3. Mỗi lần chỉ làm **một phần notebook**, đúng phần người dùng nêu, trong `notebook/llm_presentation.ipynb` (Toàn viết toàn bộ code, quyết định 15/11). Thay đổi nhỏ, chạy và kiểm, rồi giải thích đủ để Toàn bàn giao lại cho người thuyết trình mục đó.
4. Code chỉ nằm trong notebook. Không tạo file `.py` hay thư mục mã nguồn riêng.
5. Không bao giờ nói "chạy được" khi chưa chạy. Không bịa số liệu, API, tham số; chưa chắc thì nói rõ và nói cách kiểm.
6. Gọi `generate()` luôn truyền tham số sinh tường minh (bản Qwen2.5-Instruct mặc định lấy mẫu ngẫu nhiên).
7. Tách rõ **[Slide]** và **[Bổ sung]**; khẳng định kỹ thuật phải có nguồn trong `docs/references.md`.
8. Người dùng phải giải thích được từng dòng mình nộp: sau mỗi thay đổi, giải thích ngắn và hỏi lại một câu.
9. Tài liệu viết tiếng Việt, giữ thuật ngữ tiếng Anh. Trao đổi với người dùng bằng tiếng Việt.
10. Đánh dấu `[đề xuất]` / `[cần chốt]` cho tới khi Toàn duyệt (Thư chỉ review); không tự viết dàn ý description thay người; không tự thêm thí nghiệm, không đổi cấu trúc thư mục.
11. Mọi lần chạy phải có log theo `docs/templates/RUN_LOG_TEMPLATE_v2.md`.
12. Cập nhật `progress_status.md` sau mỗi bước; ghi việc dùng AI vào `AI_usage.md`.

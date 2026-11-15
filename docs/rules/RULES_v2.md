# Luật làm việc bài Presentation LLMs (v2)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-16T01:40:17+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-16T01:40:17+07:00
- Trạng thái: **Duyệt: Toàn, 2026-11-16** (Thư review)
- Dựa trên: `RULES_v1.md` (bản này), gốc là `Lab-DL/labs/lab1/docs/rules/RULES_v1.md` (Thư).
- Thay đổi so với v1:
  - Quy trình viết lại theo thứ tự Description → Discussion → Detail description → Implementation (small prompt) → Smoke run → Normal run → Audit; nói rõ việc của người và việc của AI ở từng bước.
  - Description phải đủ: Purpose, Input/Output (lấy từ đâu, lưu ở đâu), Backbone, Success criteria, Đánh giá, How-to-do.
  - Thêm luật log (§2.3), luật đánh giá (§2.4).
  - Toàn duyệt 4 cổng và merge PR; Thư review (quyết định của Toàn, 16/11).
  - v1 giữ nguyên, không dùng.
- Áp dụng cho: cả nhóm (Muội, Phúc, Toàn, Thư) và AI agent. Luật này luôn bật, đọc trước mọi việc.

## 1. Quy trình

Mỗi phần (mục 1–10 của đề, ứng với phần 0–10 của notebook) đi theo thứ tự sau, không nhảy cóc.

| Bước | Ai làm | Làm gì | Ra file gì |
|---|---|---|---|
| 1. **Description** (dàn ý, scaffold) | **Người** (người phụ trách mục; cả bài: Toàn) | Tự viết và tự sắp xếp dàn ý theo khuôn: Purpose, Input/Output, Backbone, Success criteria, Đánh giá, How-to-do. AI không viết bước này thay người | `parts/mucNN_<tên>_v1.md` (khuôn `templates/PART_TEMPLATE_v2.md`); cả bài: `presentation_description_vN.md` (khuôn `templates/DESC_TEMPLATE_v2.md`) |
| 2. **Discussion** | AI hỏi, người trả lời | AI đọc description, hỏi lại chỗ chưa rõ, mỗi lượt tối đa 5 câu. Chưa rõ thì hỏi, không tự đoán | Câu trả lời ghi vào mục "Quyết định" của description |
| 3. **Detail description / detail plan** | AI viết, người duyệt | AI chi tiết hoá **đúng theo dàn ý**: tham số, tên hàm, cell, file vào/ra, gotcha có nguồn. Ý nào AI tự thêm phải đánh `[đề xuất]` | Mục "Detailed plan" trong desc mục, hoặc `plans/` |
| 4. **Duyệt** | **Toàn** (Thư review, góp ý) | Đọc detail plan, đối chiếu với dàn ý; nói duyệt hay sửa | Ghi "Duyệt <ngày>" trong desc |
| 5. **Implementation bằng small prompt** | Toàn + AI | Mỗi prompt chỉ làm **một bước** của detail plan (ví dụ: dựng prompt → parse nhãn → tính metric). Làm xong bước nào chạy kiểm bước đó, Toàn đọc hiểu từng dòng rồi mới sang bước sau | Cell trong `notebook/llm_presentation.ipynb` |
| 6. **Smoke run (dry run)** | Toàn | `SMOKE = True`, mỗi phần vài mẫu, seed 42, ghi vào `outputs/_smoke/` (kể cả log). Đạt khi: chạy hết không lỗi; đủ file kết quả, đúng cột; log có đủ phần đầu / mỗi mẫu / cuối; `env.json` đủ phiên bản. Số smoke run **không** dùng để báo cáo | `outputs/_smoke/logs/<ts>_smoke/` |
| 7. **Normal run** | Toàn | `SMOKE = False`, **Save & Run All** trên Kaggle (GPU T4). Log theo `templates/RUN_LOG_TEMPLATE_v2.md` | `outputs/logs/<ts>_normal/`, `outputs/results/`, `outputs/figures/`; 1 mục trong `experiments/experiment_log.md` |
| 8. **Audit** | Toàn (Thư review) | Đối chiếu code, log, kết quả và slide với **description** (bước 1): từng mục Success criteria và Đánh giá đạt / không đạt, kèm bằng chứng. Lỗi không sửa thì ghi lý do. Chỉ đánh dấu Done khi audit xong | Mục "Kết quả audit" trong desc mục |

### 1.1 How to do: từng bước cho mỗi phần

**Bước 1. Description** (người phụ trách mục; AI không viết thay)
1. Tạo file `docs/parts/mucNN_<tên>_v1.md`, copy khối "Khuôn" trong `templates/PART_TEMPLATE_v2.md` vào.
2. Điền header: ngày tạo, người phụ trách mục, slide, phần notebook, `Status: To Do`.
3. Chép đúng 2 cột của đề (nội dung, minh hoạ) vào "Đề của cô".
4. Viết Goals: 2–4 dòng, mỗi dòng là một kết quả kiểm được có tên file hoặc slide.
5. Viết "Không làm": việc dễ tưởng thuộc mục này, kèm lý do.
6. Ghi Backbone (model, model card) và Input/Output (lấy từ đâu, lưu ở đâu, theo `FOLDER_STRUCTURE_v2.md` §3).
7. Viết How to do: các bước, mỗi bước một dòng (làm gì, input, output).
8. Viết Đánh giá (đo bằng gì, thế nào là đạt, output file nào) **trước** khi chạy.
9. Viết Success criteria 3 tầng: assert, smoke run, normal run.
10. Gửi Toàn. Toàn kiểm dàn ý đã đủ cụ thể để AI khỏi phải đoán chưa; chưa đủ thì trả lại.

**Bước 2. Discussion**
1. Đưa file desc cho AI kèm câu: "Đọc desc này, hỏi lại tối đa 5 câu về chỗ chưa rõ. Chưa viết thêm gì."
2. Người phụ trách trả lời từng câu.
3. Ghi câu trả lời quan trọng vào mục "Quyết định" (ngày, quyết định, lý do).

**Bước 3. Detail description / detail plan**
1. Nhờ AI: "Chi tiết hoá mục Detailed plan và Input/Output theo đúng dàn ý. Ý nào tự thêm thì đánh [đề xuất]."
2. Đối chiếu với `plans/detail_plan_notebook_v1.md` phần NN; chỗ khác nhau thì chọn một, ghi lý do.
3. Kiểm mỗi bước có đủ: cell, hàm, tham số, file ra, cách kiểm.

**Bước 4. Duyệt** (Toàn)
1. Đọc lại cả desc mục; mỗi `[đề xuất]`: giữ (bỏ nhãn) hoặc xoá.
2. Ghi `**Duyệt**: Toàn, <ngày>` ở header. Thư đọc, góp ý (không phải cổng duyệt).

**Bước 5. Implementation bằng small prompt** (Toàn)
1. Mở `notebook/llm_presentation.ipynb`, đến phần NN; `SMOKE = True`.
2. Với mỗi bước của detail plan:
   1. Prompt AI chỉ cho đúng bước đó (tên hàm, input, output).
   2. Thêm markdown trước cell (mục đích, đầu vào, đầu ra), code, comment tiếng Việt.
   3. Chạy cell; xem output; chạy `assert` của bước.
   4. Đọc hiểu từng dòng; dòng nào chưa giải thích được thì hỏi lại hoặc viết lại cho đơn giản.
   5. Thêm cell "Phân tích" (khung; chưa ghi số).
3. Hết các bước thì sang bước 6.

**Bước 6. Smoke run (dry run)** (Toàn)
1. `SMOKE = True`, chạy phần 0 rồi phần NN (hoặc Run All).
2. Mở `outputs/_smoke/logs/<ts>_smoke/`, kiểm theo checklist `plans/huong_dan_tung_buoc_v1.md` §B1.
3. Lỗi thì sửa, chạy lại; lỗi khó ghi `docs/bugs/`.
4. Ghi 1 mục "smoke run" vào `experiments/experiment_log.md`; cập nhật `progress_status.md` (cột Smoke run).
5. Bàn giao code cho người phụ trách mục (giải thích từng cell, chỉ log và file kết quả); đủ 4 điều kiện ở `progress_status.md` thì đánh dấu "Đã bàn giao".

**Bước 7. Normal run** (Toàn; chỉ khi mọi phần đã smoke run đạt)
1. `SMOKE = False`, Save & Run All trên Kaggle (`plans/huong_dan_tung_buoc_v1.md` §B2).
2. Tải `outputs/` và notebook đã chạy về repo; kiểm thư mục `outputs/logs/<ts>_normal/`.
3. Ghi 1 mục "normal run" vào `experiments/experiment_log.md` (`run_id`, thời gian, file, kết quả chính).

**Bước 8. Audit** (Toàn; Thư review)
1. Mở desc mục, đi từng dòng Success criteria và Đánh giá.
2. Mỗi dòng: tìm bằng chứng (cell, file trong `outputs/results/`, dòng log, `run_id`), ghi ✔ / ✘ vào "Kết quả audit".
3. Đối chiếu số trên slide với file kết quả.
4. ✘ mà không sửa thì ghi lý do; sửa code thì quay lại bước 6 rồi 7.
5. Đủ hết thì đổi `Status: Done`; cập nhật `progress_status.md`.

### 1.2 Ghi chú
- Chạy trên Kaggle, duyệt, commit từng bước: `plans/huong_dan_tung_buoc_v1.md`. Từng bước code phần 0–10: `plans/detail_plan_notebook_v1.md`.
- Lưu có version: tên file `_v0`, `_v1`… Luôn dùng bản số lớn nhất; đổi nội dung thì tạo `_vN+1`, bản cũ giữ nguyên.
- **Vì sao người phải tự viết dàn ý:** dàn ý quá chung thì AI tự điền theo thói quen của nó, người lại phải đi kiểm AI có làm đúng ý mình không, tức là người chạy theo AI. Dàn ý do người viết, đủ cụ thể, thì AI chỉ còn chi tiết hoá, và audit có thước đo rõ ràng.

## 2. Luật chung

### 2.1 Code và notebook
- **Toàn viết toàn bộ code** (phần 0–10), quyết định 15/11. Muội, Phúc viết dàn ý desc mục mình, làm lý thuyết, slide, thuyết trình và nhận bàn giao code (§4).
- **Code chỉ nằm trong notebook.** Không có thư mục mã nguồn riêng. Hàm dùng lại được định nghĩa ngay trong cell trước chỗ dùng, kèm docstring ngắn và 2–3 dòng `assert`.
- **Notebook vừa giữ output vừa ghi file.** Notebook nộp là bản đã chạy, có output; mỗi phần ghi kết quả ra `outputs/results/`, `outputs/figures/`, log ra `outputs/logs/`.
- **Tham số đặt ở một chỗ:** cell cấu hình đầu notebook (tên model, seed, `SMOKE`, số câu, đường dẫn). Các phần khác đọc từ đó, không gõ lại.
- **Tham số sinh luôn truyền tường minh** khi gọi `generate()`: `do_sample`, `max_new_tokens`; khi lấy mẫu thêm `temperature`, `top_p`, `top_k`, `repetition_penalty`. Bản Qwen2.5-Instruct mặc định lấy mẫu ngẫu nhiên (`generation_config.json`).
- **Giải thích được từng dòng:** mỗi cell code có markdown trước (mục đích, đầu vào, đầu ra), comment tiếng Việt cho dòng không hiển nhiên, cell "Phân tích" 2–3 câu sau output.

### 2.2 Input / Output
- Description mỗi phần phải ghi **input lấy từ đâu** (Hugging Face Hub, file trong `data/`, kết quả phần trước) và **output lưu ở đâu** (đường dẫn file, tên cột). Bảng chung: `rules/FOLDER_STRUCTURE_v2.md` §3.
- Không có trọng số tự train (bài không fine-tune); model tải từ Hugging Face vào bộ nhớ đệm, không commit (`FOLDER_STRUCTURE_v2.md` §5).
- Mỗi model dùng phải có model card trong `docs/models/` **kèm sơ đồ kiến trúc** (mermaid) và số tham số (khuôn `templates/MODEL_CARD_TEMPLATE_v2.md`).

### 2.3 Log
- Mọi lần chạy (smoke và normal) đều có log, lưu trong thư mục đặt tên theo ngày giờ bắt đầu: `outputs/logs/<YYYYMMDD_HHMMSS>_<smoke|normal>/`.
- Log phải có: thời điểm bắt đầu/kết thúc; môi trường; **tham số chạy**; **kiến trúc model** (in từ `model.config`, số tham số đếm bằng code); dữ liệu vào (file, số mẫu); **mỗi mẫu một dòng**; tổng kết (metric, thời gian, file đã ghi, lỗi). Khuôn: `templates/RUN_LOG_TEMPLATE_v2.md`.
- Không có dòng epoch, training loss/accuracy vì không train. Loss có ở phần 4 (cross-entropy từng token), accuracy ở phần 7, 10.

### 2.4 Đánh giá
- Description mỗi phần phải có mục **Đánh giá**: đánh giá cái gì; đo bằng gì (metric, công thức, hoặc tiêu chí chấm tay ghi sẵn trước khi chạy); thế nào là đạt; kết quả ghi vào file nào.
- Tiêu chí chấm tay (phần 1, 5, 8, 9) phải viết **trước** khi xem output, để không chấm theo cảm tính.

### 2.5 Số liệu, nguồn, ngôn ngữ
- **Số liệu** trên slide hoặc trong tài liệu chỉ lấy từ file kết quả của **normal run**, ghi kèm `run_id`. Không chép số từ smoke run, từ trí nhớ, hay từ output của AI.
- **Ghi nguồn:** tách rõ **[Slide ChX sN]** và **[Bổ sung]** (nguồn ngoài, đã mở kiểm tra). Khẳng định kỹ thuật nào cũng có nguồn ở `references.md`.
- **Không commit:** dữ liệu tải về (UIT-VSFC), trọng số model, slide môn học, token, file `.env`, `outputs/_smoke/`.
- **Ngôn ngữ:** tài liệu viết tiếng Việt, giữ thuật ngữ tiếng Anh. Tên biến, hàm, file viết tiếng Anh.
- **Không fine-tune** (góp ý của cô ngày 11/11). Mục 4, 5 minh hoạ bằng inference: tính loss, so base với instruct.

## 3. Phân vai người và AI
- **Người** viết dàn ý description (Purpose, I/O, Backbone, Success criteria, Đánh giá, How-to-do, phần "không làm"). AI chỉ hỏi lại và chi tiết hoá.
- Người quyết định, ghi kèm ngày và lý do. AI đề xuất và đánh dấu `[đề xuất]` hoặc `[cần chốt]` cho tới khi được duyệt.
- **Toàn duyệt** ở 4 cổng: description, detail plan, smoke run, audit; Toàn merge PR. Thư review (đọc, góp ý, hỏi Q&A thử), không phải cổng duyệt.
- AI không tự thêm thí nghiệm, không đổi cấu trúc thư mục ngoài plan đã duyệt, không tự viết dàn ý thay người.
- Chưa chắc thì AI hỏi, không đoán. AI không khẳng định "chạy được" khi chưa có output thật.

## 4. Phân công
Quyết định ngày 15/11 (Toàn): **Toàn viết toàn bộ code notebook (phần 0–10)**.

| Người | Mục: dàn ý desc + lý thuyết + slide + thuyết trình | Code notebook | Việc khác |
|---|---|---|---|
| Muội | 1, 2 (gồm phần so sánh LLM với mô hình khác), 3, 8 | — | Ghép slide; README; soạn `data/tasks_prompts.json`, `data/rag_questions.csv` (Toàn duyệt 16/11) |
| Phúc | 4, 5, 6, 9 | — | Soạn `data/base_vs_instruct.json`, `data/hallucination.csv`, `data/bias_prompts.json`, `data/long_context.json` (Toàn duyệt 16/11) |
| Toàn | 7, 10; description cả bài | **Phần 0–10** | Smoke run, normal run, log, `outputs/`, phân tích lỗi; duyệt 4 cổng, merge PR |
| Thư (hỗ trợ) | Review, hỏi Q&A thử, bấm giờ | — | Review description, plan, slide (góp ý, không duyệt) |

- Toàn viết thẳng vào `notebook/llm_presentation.ipynb`. `notebook/parts/` không bắt buộc, chỉ dùng khi Toàn muốn thử riêng một phần.
- **Người thuyết trình mục nào phải giải thích được từng dòng code phần đó** (yêu cầu của cô). Toàn code xong phần nào thì **bàn giao**: giải thích từng cell, chỉ log và file kết quả, ghi vào `progress_status.md`. Muội, Phúc hỏi đến khi tự giải thích được.

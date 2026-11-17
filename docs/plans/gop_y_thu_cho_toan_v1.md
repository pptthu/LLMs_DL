# Góp ý repo LLMs_DL gửi Toàn

- **Created**: 2026-11-17T10:42:37+07:00
- **Last Updated**: 2026-11-17T11:25:29+07:00 (Thư, AI hỗ trợ)
- **Người viết**: Thư
- **Trạng thái**: góp ý, chờ Toàn trả lời §2, §4.4, §5. Toàn trả lời thì ghi quyết định vào `presentation_description_v2.md` mục "Quyết định"; file này giữ nguyên làm bằng chứng.
- **Đã đọc**: `main` tới PR #3 (commit `d1ec7e7`) và notebook phần 0–1 (output smoke run `20261116_035414_smoke`).
- **Plan đối chiếu**: `present_llms.md` (bản có góp ý của cô ngày 11/11; trong repo là `docs/plans/ke_hoach_presentation_llms_v2.md`), `khung_presentation_llms.md` (phân công; trong repo là `docs/plans/khung_presentation_llms_v2.md`). Trích dẫn ghi tên `present_llms.md` / `khung` là nói nội dung chung của 2 bản này.
- **Ký hiệu bằng chứng**:
  - `file:dòng` hoặc `file §mục` là trích từ repo;
  - **[HF]** là số đọc trực tiếp từ `config.json` hoặc API của Hugging Face ngày 17/11;
  - **[Nguồn]** là bài báo hoặc tài liệu, link ở cuối file;
  - **[Tự tính]** là phép tính của Thư, có ghi công thức;
  - **[Đoán]** là ước lượng chưa kiểm chứng.

## Cách đọc
| Phần | Nội dung | Toàn cần làm |
|---|---|---|
| §1 | Thư đã chốt | Cập nhật docs theo |
| §2 | Cần bàn lại: Thư có ý kiến nhưng chưa chốt | Đọc lý do, trả lời đồng ý hay không |
| §3 | Đối chiếu yêu cầu của cô theo từng mục | Xem mục nào chưa đạt và đề xuất sửa |
| §4 | Demo B mục 2: vấn đề và đề xuất chi tiết | Trả lời câu 2.1 |
| §5 | Câu hỏi về docs, code, git | Trả lời từng câu |

---

## 0. Điểm tốt (giữ nguyên)
- Notebook phần 0–1 đúng quy ước `RULES_v2.md` §2.1:
  - markdown trước mỗi cell, comment tiếng Việt, `assert`;
  - log `run.log`, `partNN.log`, `_records.csv`;
  - smoke và normal ghi ra hai thư mục riêng;
  - in đường dẫn tương đối, không lộ thông tin máy.
- Greedy truyền đủ `do_sample=False, temperature=None, top_p=None, top_k=None` (cell 1.1). Đúng chú ý số 1 ở `plan_presentation_llm_v3.md` §6, vì bản Instruct mặc định lấy mẫu ngẫu nhiên.
- Smoke run có bằng chứng: `progress_status.md:52` ghi *"số tham số 1,543,714,304 khớp; UIT-VSFC khớp số dòng"*.
- Output cell 1.1: prompt "Xin chào" thì model trả lời *"你好！有什么我可以帮助你的吗？"* (tiếng Trung). Đây là ví dụ thật, tốt cho phần phân tích mục 1 và mục 9.

---

## 1. Thư đã chốt (Toàn cập nhật docs theo)

| # | Đã chốt | Lý do | Sửa ở đâu |
|---|---|---|---|
| 1.1 | **Toàn là người chính, duyệt 4 cổng** (description, detail plan, smoke run, audit). Thư chỉ review sơ: xem tiến độ, kiểm có lệch plan không | Toàn code toàn bộ nên nắm rõ nhất; theo khung, Thư không nhận mục nào | Giữ `RULES_v2.md:125` |
| 1.2 | **AI được viết description; Toàn discussion lại và duyệt** | Luật và thực tế đang mâu thuẫn. `RULES_v2.md` §1 bước 1 ghi *"AI không viết bước này thay người"*, còn `progress_status.md:54` ghi desc mục 1 *"AI viết theo yêu cầu Toàn"*. Chốt theo cách đang làm | `RULES_v2.md` §1 (bảng bước 1 và mục 1.1 "Bước 1"), §3 dòng đầu; `AGENTS.md` mục 10 (*"không tự viết dàn ý description thay người"*) |
| 1.3 | **Toàn code toàn bộ phần 0–10.** Muội, Phúc review code sau để học, đủ để giải thích khi vấn đáp | Chỉ một người sửa notebook thì không xung đột khi merge (`plan_presentation_llm_v3.md:176`) | Giữ nguyên |
| 1.4 | **Khối lượng: kịp** | Toàn đã xác nhận được | — |
| 1.5 | **Cô đã trả lời "cứ làm"** cho mục 5, mục 8 và việc dùng UIT-VSFC | Không còn gì phải chờ | Bỏ chữ "chờ cô" ở `progress_status.md` (dòng 17, 19, 20, 40), `khung_presentation_llms_v2.md`, `noi_dung_muc05_*_v1.md` §3; ghi 1 dòng vào mục "Quyết định" của `presentation_description_v2.md` |
| 1.6 | **Commit viết tiếng Việt** | Cả nhóm đọc được | `CONTRIBUTING.md:25` (đang ghi *"Mô tả: tiếng Anh"*) và cột ví dụ của bảng commit |
| 1.7 | Commit `efdddef` *"hore(repo)"* là gõ nhầm của `chore(repo)`; để nguyên | Không viết lại lịch sử (`Lab-DL/CONTRIBUTING.md`: *"Old messages on main stay as they are"*) | — |
| 1.8 | **Nhánh `main` đã được bảo vệ** | Tránh push thẳng lên `main` | Xem câu 5.7 |

---

## 2. Cần bàn lại (Thư có ý kiến, chưa chốt)

### 2.1 So sánh LLM với VLM ở mục 2
- **Thư muốn** bài có mở rộng sang VLM.
- **Chưa chốt** hình thức: demo B như plan hiện tại, đổi cặp model, hay chỉ nói 1 dòng và dùng demo khác.
- Phân tích và đề xuất ở **§4**. Toàn đọc rồi chọn một trong 3 phương án ở §4.4.

### 2.2 Chấm mục 8 (RAG) và mục 9 (hallucination)
Plan hiện tại (`noi_dung_muc08_*_v2.md` §3.4, `noi_dung_muc09_*_v2.md` §5.1b) là: chấm tự động, sau đó **2 người chấm mù toàn bộ**, báo % đồng ý và **Cohen's kappa**.

**a) Vì sao nhóm chỉ "mượn ý tưởng" các bộ chấm, không làm đúng phương pháp của chúng**

Plan ghi *"theo SimpleQA"*, *"theo cách của SQuAD"*... nhưng không bộ nào được làm đúng như bài gốc:

| Bộ | Bài gốc làm thế nào [Nguồn] | Nhóm làm được gì | Vì sao không làm đúng như gốc |
|---|---|---|---|
| **SimpleQA** (Wei et al. 2024) | 4,326 câu tiếng Anh, soạn theo kiểu đối kháng với GPT-4; **chấm bằng một ChatGPT classifier** (3 nhãn) | Khoảng 20 câu tiếng Việt, **người chấm**, dùng 3 nhãn | Plan đã chốt *"Không dùng LLM làm giám khảo chính"* (`noi_dung_muc08_*_v2.md` §3.4), lại không có API; quy mô và ngôn ngữ khác |
| **RGB** (Chen et al., AAAI 2024) | Bộ dữ liệu song ngữ Anh–Trung từ tin tức, đo **4 khả năng**: chịu nhiễu, từ chối khi không có thông tin (negative rejection), ghép thông tin, phát hiện thông tin sai | Chỉ lấy **1/4 khả năng**: thêm 3–5 câu mà slide không có đáp án | Không có bộ dữ liệu của RGB cho slide môn học tiếng Việt |
| **RAGAS** (Es et al. 2023) | Faithfulness: **một LLM** tách câu trả lời thành các khẳng định rồi chấm Có/Không với từng khẳng định | Recall@3 tính bằng code (so với slide nguồn ghi sẵn); faithfulness chấm tay | Như SimpleQA: không dùng LLM chấm |
| **SQuAD** (Rajpurkar et al. 2016) | Exact match và token F1 sau khi chuẩn hoá chuỗi | Dùng "output **có chứa** đáp án" và token F1 | "Có chứa" lỏng hơn exact match, vì câu trả lời của LLM dài hơn đáp án mẫu |
| **FActScore** (Min et al. 2023) | Tách tiểu sử thành các ý nhỏ, kiểm từng ý bằng truy xuất Wikipedia và LM | Chấm tay tỉ lệ ý có nguồn | Không có hệ truy xuất và LM kiểm chứng |

→ **Hệ quả:** trên slide và trong vấn đáp phải nói *"dựa trên ý tưởng của SimpleQA, RGB, RAGAS"*. Nếu nói "theo chuẩn SimpleQA", cô hỏi "nhóm có chạy bộ SimpleQA không, chấm bằng gì?" thì nhóm không trả lời được.

**b) Kappa: Thư đề xuất bỏ**

Lý do có bằng chứng:
- Chính repo đã ghi: *"với khoảng 20 câu, kappa dao động mạnh nên luôn báo kèm % đồng ý"* (`noi_dung_muc08_*_v2.md` §3.4).
- **[Tự tính]** Ví dụ 20 câu, 2 nhãn chia đều (pe = 0.5), hai người đồng ý 85% (po = 0.85):
  - κ = (0.85 − 0.5)/(1 − 0.5) = **0.70**;
  - sai số chuẩn theo công thức xấp xỉ √(po(1−po)/(n(1−pe)²)) ≈ 0.16;
  - khoảng tin cậy 95% ≈ **0.39 → 1.0**, tức trải từ mức "fair" tới "almost perfect" theo thang Landis–Koch. Con số này gần như không nói được gì.
  - Công thức xấp xỉ này vốn dành cho mẫu lớn, nên với n = 20 còn kém chính xác hơn nữa.
- Sim & Wright (2005) đã lập bảng cỡ mẫu tối thiểu cho nghiên cứu dùng kappa [Nguồn], tức kappa cần cỡ mẫu được tính trước. 20 câu chưa được tính như vậy.

**c) Người chấm thứ 2: Thư đề xuất KHÔNG bỏ hẳn, mà thu hẹp lại**

Nói rõ để không hiểu nhầm: Thư bỏ *kappa* và bỏ *việc 2 người chấm toàn bộ*, nhưng vẫn giữ *một bước kiểm của người thứ 2*. Lý do:
1. Nếu chỉ một người chấm, người đó là Toàn, cũng là người viết prompt và code. Không có gì kiểm việc chấm. Repo cũng tự đặt luật chống chấm theo cảm tính (`RULES_v2.md` §2.4: *"Tiêu chí chấm tay… phải viết trước khi xem output, để không chấm theo cảm tính"*).
2. Chấm tự động cũng sai: bộ chấm tự động gốc của SimpleQA có **15 false negative trên 1,000 câu** do người viết (Epoch AI) [Nguồn]. Lỗi thường nằm ở những câu mà máy và người chấm khác nhau, nên kiểm đúng những câu đó là hiệu quả nhất.

**Ba phương án để Toàn chọn:**

| Phương án | Cách làm | Tốn công | Rủi ro |
|---|---|---|---|
| A. Giữ như plan | 2 người chấm mù toàn bộ, báo % đồng ý và kappa | Cao nhất: mỗi câu trả lời chấm 2 lần. Mục 8 có khoảng 20–25 câu × 2 cách = 40–50 câu trả lời, mục 9 khoảng 20 | Kappa không có ý nghĩa với cỡ mẫu này (mục b) |
| **B. Thư đề xuất** | Chấm tự động → Toàn chấm mù (trộn thứ tự, ẩn cột có RAG / không RAG) → **người thứ 2 chỉ chấm lại những câu mà nhãn của Toàn khác nhãn tự động** → báo % đồng ý trên phần chấm lại; bỏ kappa | Chỉ chấm lại phần lệch. [Đoán] dưới 1/4 số câu | Câu mà cả máy lẫn Toàn cùng chấm sai sẽ không được phát hiện |
| C. Bỏ hẳn | Chấm tự động + Toàn chấm | Thấp nhất | Không có kiểm độc lập; vấn đáp bị hỏi "ai kiểm người chấm?" |

**d) Các thành phần khác của cách chấm**

| Thành phần | Đề xuất | Lý do |
|---|---|---|
| Nhãn 3 mức Correct / Incorrect / Not attempted | Giữ | Tách "bịa" với "từ chối". Mục 9 cần đúng điều này (`present_llms.md`: *"Câu trả lời sai + cách kiểm chứng"*) |
| 3–5 câu không có đáp án trong slide | Giữ | Là phép thử negative rejection, khả năng mà RGB kết luận LLM *"struggle significantly"* [Nguồn]. Có khả năng cao cho ra ví dụ thật để phân tích |
| Recall@3 | Giữ | Tính bằng code, chỉ cần cột slide nguồn. Tách được lỗi tìm sai đoạn với lỗi model sinh sai, rất hợp cho phần "phân tích lỗi" mà cô yêu cầu nộp |
| Faithfulness chấm tay | Làm nếu còn thời gian | Khoảng 20 câu có RAG |
| FActScore | Bỏ | Chỉ áp dụng cho loại câu "xin trích dẫn", tốn công tách ý nhất |

### 2.3 Mục 7, 10: Bonferroni và chọn prompt trên validation
- **Bonferroni: Thư đề xuất giữ.** Mục 7 so zero-shot với 3 bộ few-shot, tức 3 lần kiểm định McNemar (`noi_dung_toan_muc07_muc10_v2.md:246`). Mỗi lần có 5% khả năng ra "có ý nghĩa" do may mắn; 3 lần thì khả năng ít nhất một lần may mắn tăng lên khoảng 1 − 0.95³ ≈ **14%** [Tự tính]. Bonferroni hạ ngưỡng còn 0.05/3 ≈ 0.017 để đưa về lại khoảng 5%. Chỉ tốn một phép chia; file đã ghi báo kèm p chưa hiệu chỉnh.
- **Chọn prompt trên validation: giữ, nhưng thêm giới hạn.** File ghi *"Không đổi prompt sau khi đã xem kết quả test"* (dòng 60). Đúng: sửa prompt theo kết quả test là chọn bằng tập test, làm accuracy bị đẩy lên. Nhưng dòng 206 ghi *"thử nhiều mẫu prompt"* mà không có giới hạn. **Đề xuất:** tối đa 4 mẫu × 100 câu validation = 400 lượt sinh, ghi mọi mẫu đã thử và tỉ lệ không parse được vào log.

---

## 3. Đối chiếu yêu cầu của cô theo từng mục

- **Căn cứ:** cột "Nội dung cần trình bày" và dòng *"Góp ý của cô (11/11)… mọi mục đều có demo trên model thật kèm phân tích output"* trong `present_llms.md`.
- **Lưu ý chung:** bảng đánh giá **trên plan**. Hiện chỉ phần 0 và 1 có output thật (`progress_status.md` bảng tiến độ). Mục nào cũng chỉ thật sự đạt khi đã chạy và có cell "Phân tích".

| Mục | Cô yêu cầu | Plan đang làm | Đánh giá | Vì sao | Đề xuất | Nguồn research trong repo |
|---|---|---|---|---|---|---|
| **1. Giới thiệu** | LLM là gì; phân biệt language model, LLM, chatbot | 4 tác vụ trên bản Instruct (đã chạy, 4/4) | **Đạt một phần** | Lý thuyết phân biệt 3 khái niệm đã đủ (sơ đồ LLM → instruct → chatbot, bảng 3 khái niệm). Nhưng demo 4 tác vụ chỉ cho thấy "một model nhiều tác vụ", không cho thấy LM, LLM, chatbot khác nhau | Trên slide 3 nói kèm: "base = LM chỉ viết tiếp; instruct = đã post-training; chatbot = instruct + giao diện + lớp an toàn", trỏ sang demo mục 5. Không cần thêm code | `noi_dung_muc01_gioi_thieu_v1.md` §3.2, §3.3, §4 |
| **2. Nền tảng kiến trúc** | Vai trò Transformer; Decoder-only và causal attention | Config thật, ma trận causal mask, attention thật (eager), top-5 token; demo A BERT vs Qwen; demo B LLM vs VLM | **Đạt** (trừ demo B) | Config, mask, top-5 đúng yêu cầu. Demo A cho thấy trực tiếp causal (Qwen chỉ nhìn bên trái) so với hai chiều (BERT). Demo B **ngoài trọng tâm**, xem §4 | Giữ demo A; demo B xem §4 | `noi_dung_muc02_kien_truc_v2.md` §3–§6, §11 |
| **3. Tokenization** | Chia token; token ID và embedding | Tokenize câu Việt và Anh; số token GPT-2 vs Qwen; kích thước embedding | **Đạt** | Đủ 3 ý của đề | Đếm token với `add_special_tokens=False` (`plan_v3` §6 #7) | `noi_dung_muc03_tokenization_v2.md` |
| **4. Nguyên lý học** | Next-token prediction; dữ liệu huấn luyện; cross-entropy loss | Cặp ngữ cảnh → token đích; loss tự tính so với `outputs.loss` (lệch < 1e-3) | **Đạt** | "Dữ liệu huấn luyện" chỉ trình bày lý thuyết (GPT-3, Qwen2.5). Hợp lý, vì cô cho phép không fine-tune | — | `noi_dung_muc04_nguyen_ly_hoc_v2.md` §2, §3, §4 |
| **5. Các giai đoạn huấn luyện** | Pretraining, SFT, preference alignment; RLHF/DPO ở mức ý tưởng | Cùng prompt trên base và Instruct, `do_sample=False` | **Đạt một phần** | Qwen2.5-Instruct đã qua **SFT, DPO và GRPO** cùng lúc. Demo chỉ thấy tác dụng *gộp* của post-training, không tách riêng SFT với alignment được, vì không có bản chỉ qua SFT | Slide 11 ghi rõ: "khác biệt là tác dụng gộp của SFT + alignment". Câu Q&A dự kiến: "Sao không tách SFT riêng?" | `noi_dung_muc05_giai_doan_huan_luyen_v1.md` §2.5, §3.1 |
| **6. Quá trình sinh văn bản** | Autoregressive; greedy, temperature, top-p | Sinh từng token; 5 cấu hình × 5 mẫu; kiểm greedy chạy 2 lần cho cùng kết quả | **Đạt** | Đủ yêu cầu | Mỗi cấu hình truyền đủ tham số (`plan_v3` §6 #1) | `noi_dung_muc06_sinh_van_ban_v2.md` |
| **7. Khả năng của LLM** | In-context learning; làm theo chỉ dẫn; xử lý tác vụ ngôn ngữ | Zero-shot vs 3-shot × 3 bộ, 500 câu UIT-VSFC; "làm theo chỉ dẫn" dẫn lại mục 1 | **Đạt** | Thí nghiệm có kiểm soát, chỉ đổi một biến | Xem §2.3 | `noi_dung_toan_muc07_muc10_v2.md` |
| **8. Thích nghi và bổ sung tri thức** | Phân biệt prompting, fine-tuning, RAG | Bảng 3 cách; demo RAG mới; prompting dẫn lại mục 7; fine-tuning dẫn lại mục 5 | **Đạt** | Đủ cả 3 cách, mỗi cách có minh hoạ trên model thật | Cách chấm xem §2.2. `data/rag_questions.csv` chưa có | `noi_dung_muc08_prompting_finetuning_rag_v2.md` §3 |
| **9. Hạn chế và đánh giá** | Hallucination, thiên lệch, giới hạn ngữ cảnh; chất lượng phụ thuộc tác vụ và dữ liệu | 4 demo, mỗi ý một demo; nhãn nhiễu UIT-VSFC | **Đạt** (nếu làm đủ) | Ý "phụ thuộc dữ liệu" có căn cứ: UIT-VSFC có độ đồng thuận người gán nhãn trên 91% | Ý "phụ thuộc tác vụ" lấy từ bảng tổng hợp mục 10, nên mục 10 phải xong trước slide 18 | `noi_dung_muc09_han_che_danh_gia_v2.md` §1–§5 |
| **10. Demo và kết luận** | Thử nghiệm có kiểm soát, tổng hợp kết quả, nhận xét | Accuracy, confusion matrix, tỉ lệ parse lỗi, McNemar, phân tích lỗi, bảng tổng hợp | **Đạt** | Phần chặt chẽ nhất | Xem §2.3 | `noi_dung_toan_muc07_muc10_v2.md` §7 |

---

## 4. Demo B ở mục 2 (LLM vs VLM): vấn đề và đề xuất chi tiết

### 4.1 Demo B hiện tại là gì
Theo `noi_dung_muc02_kien_truc_v2.md` §11.3 và `detail_plan_notebook_v1.md:47–48`:
- Cho một ảnh có chữ tiếng Việt (`data/vlm_image.png`) và câu hỏi về nội dung ảnh (`data/compare_prompts.json`).
- **Qwen2.5-1.5B-Instruct** chỉ nhận câu hỏi; **Qwen2-VL-2B-Instruct** nhận cả ảnh và câu hỏi.
- Thêm cell 2.7, 2.8 và slide 6b (khoảng 1 phút; *"tổng bài thành 22 slide"*).

### 4.2 Vì sao tốn công: từng khoản và bằng chứng

| Khoản | Tốn gì | Bằng chứng |
|---|---|---|
| Code mới, không dùng lại được hàm cũ | VLM phải nạp bằng `AutoProcessor` và `Qwen2VLForConditionalGeneration`, prompt phải chèn ảnh vào. Hàm `chat_generate` ở cell 1.1 chỉ nhận chữ (`tok.apply_chat_template(messages…)`), nên phải viết hàm sinh mới | `detail_plan_notebook_v1.md:47`; cell 1.1 trong notebook |
| Quản lý bộ nhớ GPU | Phần 2 đã có model chính (khoảng 3.1 GB) và nạp thêm bản eager (khoảng 3.1 GB). VLM có **2,208,985,600** tham số, ở fp16 khoảng **4.4 GB**. Phải giải phóng bản eager trước khi nạp VLM, đúng như detail plan ghi. T4 trên Kaggle có 14.6 GB mỗi GPU, nên vừa nhưng phải sắp thứ tự nạp và giải phóng cẩn thận | [HF] số tham số; output cell 0.3: `gpu_vram_gb: 14.6`; `detail_plan_notebook_v1.md:47` *"giải phóng model eager"* |
| Dữ liệu chưa có người làm | `vlm_image.png` và `compare_prompts.json` chỉ xuất hiện trong detail plan. Danh sách file `data/` đã phân công (`RULES_v2.md` §4) không có 2 file này. Ảnh phải tự tạo, không vi phạm bản quyền, có đáp án ghi sẵn | `detail_plan_notebook_v1.md:37`; `RULES_v2.md` §4 |
| Thời gian chạy | Mỗi lần Save & Run All tải thêm khoảng 4.4 GB | [HF], [Đoán] vài phút |
| Thời lượng trình bày | +1 phút, trong khi tổng thời gian các slide trong khung đã là 20.5 phút | `khung_presentation_llms_v2.md:42`: *"Tổng khoảng 20.5 phút, chưa tính thời gian chuyển sang notebook"* |
| ~~Lỗi phiên bản `transformers`~~ | **Gần như không còn rủi ro.** Kaggle đang chạy `transformers 5.20.2`, mới hơn nhiều so với bản đầu tiên hỗ trợ Qwen2-VL | Output cell 0.3 (`20261116_035414_smoke`) |

### 4.3 Vì sao thu được ít
1. **Kết quả biết trước khi chạy.** Model chỉ đọc chữ thì không thấy ảnh, nên chắc chắn không trả lời được câu hỏi về ảnh. Không có gì để phân tích, trong khi cô yêu cầu *"phân tích output"* ở mọi mục.
2. **So sánh không sạch.** [HF] cho thấy phần LLM bên trong Qwen2-VL-2B có config **giống hệt** Qwen2.5-1.5B (28 layer, hidden 1536, 12 head, intermediate 8960), nhưng trọng số thuộc **thế hệ Qwen2** (bài báo Qwen2-VL ghi bản 2B = ViT 675M + LLM 1.5B [Nguồn]), tức được huấn luyện khác. Ngoài chuyện có ảnh hay không, hai model còn khác nhau ở quá trình huấn luyện LLM.
3. **Bảng và demo không khớp.** Bảng §11.1 ghi VLM là *"Qwen2.5-VL… khởi tạo từ trọng số Qwen2.5"*, còn demo dùng Qwen2-VL-2B.
4. **Ngoài trọng tâm của đề.** Mục 2 yêu cầu *"tập trung vào Decoder-only và causal attention"*. Các tác vụ mà cô nêu ở mục 1 (hỏi đáp, tóm tắt, dịch, sinh mã) đều là tác vụ chữ.

Lý do Toàn không dùng Qwen2.5-VL-3B là đúng: [HF] cho thấy model này có **3,754,622,976** tham số, vượt luật "không dùng model lớn hơn 3B" (`presentation_description_v2.md`, mục "Không làm").

### 4.4 Ba phương án

| | Phương án 1 (Thư đề xuất) | Phương án 2 | Phương án 3 (giữ như plan) |
|---|---|---|---|
| VLM | 1 dòng chốt phạm vi ở slide 2 hoặc 3 | Demo với cặp cùng phần LLM bên trong | Demo B hiện tại |
| Demo thêm ở mục 2 | Qwen2.5-0.5B vs 1.5B (§4.5) | Qwen2-1.5B-Instruct vs Qwen2-VL-2B trên 4 tác vụ chữ | Qwen2.5-1.5B vs Qwen2-VL-2B trên câu hỏi về ảnh |
| Kết quả biết trước? | Không | Không | **Có** |
| Đúng trọng tâm decoder-only? | Có | Một phần | Không |
| Dùng lại `chat_generate`? | Có | Có cho bản text; VLM vẫn cần hàm riêng | Không |
| Model tải thêm | 0.5B (khoảng 1 GB) | 2 model (khoảng 3.1 GB + 4.4 GB) | 1 model (khoảng 4.4 GB) |
| File data mới | Không (dùng `tasks_prompts.json`) | Không (dùng `tasks_prompts.json`) | 2 file, chưa có người làm |

### 4.5 Phương án 1 chi tiết (để Toàn code)

**Bằng chứng hai model cùng kiến trúc, chỉ khác kích thước** [HF]:

| | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct |
|---|---|---|
| Kiến trúc | `Qwen2ForCausalLM` | `Qwen2ForCausalLM` |
| Số tham số | 494,032,768 | 1,543,714,304 (khớp `progress_status.md:52`) |
| Số layer | 24 | 28 |
| Hidden size | 896 | 1536 |
| Attention head / KV head | 14 / 2 | 12 / 2 |
| Intermediate size | 4864 | 8960 |
| Vocab | 151,936 | 151,936 |
| Context tối đa | 32,768 | 32,768 |

**Vì sao chọn cặp này:**
- Cùng kiến trúc và cùng tokenizer, nên khác biệt chủ yếu đến từ kích thước. Gắn với chữ "**Large**" ở mục 1 và bảng config ở mục 2.
- Kết quả **chưa biết trước**, nên có cái để phân tích.
- 0.5B vốn là model dự phòng trong `present_llms.md` (*"`Qwen2.5-0.5B-Instruct` nếu chạy chậm"*). Research của repo cũng đã ghi đây là cách *"Dễ nhất"* (`noi_dung_muc02_kien_truc_v2.md` §11.2).

**Cell đề xuất** (thay cell 2.7, 2.8 trong `detail_plan_notebook_v1.md`):

| Cell | Làm gì | Ra file | Kiểm |
|---|---|---|---|
| 2.7 | Nạp `Qwen/Qwen2.5-0.5B-Instruct` (`DTYPE`, `device_map={"": 0}` như cell 0.4); `.eval()` | — | `assert` số tham số == 494,032,768 |
| 2.8 | Bảng config 2 model cạnh nhau (layer, hidden, head, KV head, intermediate, số tham số) | `results/compare_size_config.csv` | Khớp bảng trên |
| 2.9 | Chạy 4 tác vụ trong `data/tasks_prompts.json` trên model 0.5B bằng `chat_generate` (cùng `max_new_tokens` như phần 1); lấy lại kết quả của 1.5B từ phần 1, không chạy lại; tính token/giây = `n_out / time_s` | `results/compare_size_tasks.csv`: `id, task, model, output, n_output_tokens, time_s, tok_per_s, truncated, judged` | Đủ 4 tác vụ × 2 model; chấm `judged` theo cột `check_method` đã có |
| 2.10 | `del` model 0.5B, `torch.cuda.empty_cache()` | — | In VRAM còn trống |

- Thêm cell "Phân tích" 2–3 câu sau output.
- Log theo `RUN_LOG_TEMPLATE_v2.md` vào `part02_*.log`.

**Lưu ý khi trình bày** (để không bị bắt bẻ khi vấn đáp):
- Ghi "**minh hoạ**", không ghi "chứng minh". Chỉ có 2 model và 4 câu; hai model khác nhau cả số layer, hidden size, số head, không riêng số tham số.
- Không dùng chữ "độ thông minh", vì không đo được. Dùng đúng/sai theo tiêu chí ghi sẵn.
- Tốc độ phải **đo rồi mới nói**. [Đoán] Sinh từng token với batch 1 thì thời gian phụ thuộc nhiều vào việc đọc trọng số từ bộ nhớ, nên 0.5B chưa chắc nhanh gấp 3 lần 1.5B.
- Không chạy trực tiếp lúc thuyết trình; chỉ vào output đã có (`present_llms.md`: *"Notebook đã chạy sẵn có output"*).
- Nếu muốn có con số accuracy: chạy zero-shot 100 câu UIT-VSFC trên 0.5B, đặt ở **phần 10** chứ không đặt ở phần 2, vì hàm prompt và parse của mục 7 được định nghĩa sau phần 2.

**Slide:**
- Slide 2 hoặc 3 thêm 1 dòng: *"Phạm vi: LLM xử lý văn bản. VLM = bộ mã hoá ảnh + LLM decoder-only, không thuộc phạm vi bài."*
- Slide 6b: bảng config và bảng 4 tác vụ × 2 model.
- Bù 1 phút bằng cách rút slide 12, theo cách khung đã ghi sẵn (`khung_presentation_llms_v2.md:42`): *"Nếu quá giờ, rút slide 12 trước; slide 2 **không rút**"*. Slide 2 chỉ thêm 1 dòng phạm vi, không thêm thời gian đáng kể. Ý "sinh từng token" nói gộp vào slide 13.

### 4.6 Nếu chọn phương án 2
- [HF] cho thấy phần LLM của Qwen2-VL-2B có config **giống hệt** Qwen2-1.5B-Instruct (28 layer, hidden 1536, 12 head, KV 2, intermediate 8960, vocab 151,936). Hai model cùng thế hệ Qwen2, nên khác biệt chỉ đến từ việc ghép ViT và huấn luyện thêm trên dữ liệu ảnh–chữ.
- Câu hỏi chính: *"Sau khi ghép thêm phần ảnh, model còn giữ được 4 tác vụ chữ ở mục 1 không?"*. Câu hỏi về ảnh chỉ để minh hoạ phụ.
- Vẫn tốn các khoản ở §4.2 (code VLM riêng, bộ nhớ GPU), cộng thêm 1 model text không phải model chính.

---

## 5. Câu hỏi về docs, code, git (Toàn trả lời)

| # | Câu hỏi | Bằng chứng |
|---|---|---|
| 5.1 | **Kaggle Dataset cho `data/`**: chốt theo code rồi sửa plan? | `plan_presentation_llm_v3.md:20` ghi *"Bỏ `src/`, `tests/`, Kaggle Dataset"*; notebook cell 0.1 in ra *"DATA_DIR: Kaggle Dataset llms-dl-data"* |
| 5.2 | **`device_map`**: sửa plan theo code? | `plan_presentation_llm_v3.md:128` ghi `"auto"`; `progress_status.md:52` ghi *"cell 0.4 đổi `device_map={"": 0}`"* |
| 5.3 | **Ai code** trong plan v3? | `plan_presentation_llm_v3.md:23` ghi *"Task code vẫn chia cho Muội và Phúc… Toàn ghép"*, mâu thuẫn với dòng 10 cùng file |
| 5.4 | **File phân tích lỗi** đặt ở đâu? | `plan_presentation_llm_v3.md:71`: `docs/error_analysis.md`; `README.md:83`: `docs/experiments/error_analysis_v1.md` |
| 5.5 | **`long_context.json`** có làm không? | Có trong `RULES_v2.md` §4 (việc của Phúc), thiếu trong cây thư mục `plan_presentation_llm_v3.md` §2 |
| 5.6 | **Khi nào ghim phiên bản** vào `requirements.txt`? | `present_llms.md` yêu cầu ghim; `requirements.txt:2` ghi *"sau lần chạy chính thức"*. Đã có sẵn phiên bản thật từ cell 0.3 (`transformers 5.20.2`, `torch 2.11.0+cu130`…) |
| 5.7 | `main` đã bảo vệ: **PR của Toàn thì ai approve?** | Nếu luật bảo vệ yêu cầu ít nhất 1 approve thì GitHub không cho tự approve PR của chính mình |
| 5.8 | **Docs lỗi thời**: sửa đè hay tạo `_vN+1`? | `FOLDER_STRUCTURE_v2.md` §1: notebook *"chưa có code"*, `data/` *"trống"*, còn `_to_delete/`; thư mục `docs/drafts/` vẫn còn dù commit `6bf0854` nói đã bỏ; `README.md:5` *"chưa chạy notebook"*; `docs/00_README_v2.md:36` ghi `parts/` *"chưa có"*; `progress_status.md:41` *"Cần chốt: không còn"* |

---

## Nguồn

- Wei et al. 2024, *Measuring short-form factuality in large language models* (SimpleQA): https://arxiv.org/abs/2411.04368 ; https://openai.com/index/introducing-simpleqa/
- Epoch AI, *SimpleQA Verified review* (15 false negative / 1,000 câu): https://epoch.ai/benchmarks/simple-qa-verified/review
- Chen et al., AAAI 2024, *Benchmarking LLMs in Retrieval-Augmented Generation* (RGB, 4 khả năng): https://arxiv.org/abs/2309.01431
- Es et al. 2023, *RAGAS*; tài liệu metric faithfulness: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- Rajpurkar et al. 2016, *SQuAD*: https://arxiv.org/abs/1606.05250
- Min et al. 2023, *FActScore*: https://arxiv.org/abs/2305.14251
- Sim & Wright 2005, *The kappa statistic in reliability studies: use, interpretation, and sample size requirements*, Physical Therapy 85(3): https://pubmed.ncbi.nlm.nih.gov/15733050/
- Wang et al. 2024, *Qwen2-VL* (bản 2B = ViT 675M + LLM 1.5B): https://arxiv.org/abs/2409.12191
- Config và số tham số [HF], đọc ngày 17/11:
  - https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
  - https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
  - https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
  - https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
  - https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

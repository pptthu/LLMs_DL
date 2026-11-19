# Chốt khung v4 và kế hoạch 18–21/11 (Thư → Toàn)

- **Created**: 2026-11-18T20:37:31+07:00
- **Last Updated**: 2026-11-18T20:51:07+07:00 (Thư, AI hỗ trợ)
- **Người chốt**: Thư
- **Áp dụng cho**: `khung_presentation_llms_v4.md`, `ke_hoach_18_21_11_v2.md`
- **Trạng thái**: **ĐÃ CHỐT, không bàn thêm.** Toàn sửa 2 file trên theo file này rồi làm luôn.
- File thảo luận, chưa đưa vào repo, nên sửa đè, không đánh version.

## Mục tiêu
1. **Đạt 100% yêu cầu của cô:** đủ 10 mục, đủ 4 sản phẩm nộp, đủ 6 yêu cầu thêm.
2. **Khả thi:** làm xong trước 12h trưa thứ 6.
3. **Có thời gian thì mở rộng phần so sánh** (§5). Phần mở rộng không được làm ảnh hưởng số liệu chính thức.

Không bắt buộc theo đúng `present_llms.md`. Chỉ giữ những gì cần cho 3 mục tiêu trên.

---

## 1. Kiểm khung v4 so với yêu cầu của cô

| Yêu cầu | Khung v4 đã có | Còn thiếu → chốt |
|---|---|---|
| Mục 1: LLM là gì, phân biệt LM, LLM, chatbot; ví dụ 4 tác vụ | Đủ | Phân tích output nối với slide 7: base ≈ LM chỉ viết tiếp, Instruct đã post-training, chatbot = Instruct + giao diện + lớp an toàn |
| Mục 2: Transformer; **decoder-only và causal attention**; sơ đồ token → xác suất token tiếp theo | Chỉ có xác suất token tiếp theo | **C1:** thêm **ma trận causal mask thật** của câu ví dụ vào cột demo slide 4 (cell đã có) |
| Mục 3: token, token ID, embedding | Đủ | — |
| Mục 4: next-token, dữ liệu huấn luyện, cross-entropy | Đủ | — |
| Mục 5: pretraining, SFT, alignment, RLHF/DPO | Có demo base vs Instruct | **C2:** so sánh cho công bằng (§2.1) |
| Mục 6: greedy, temperature, top-p; cùng prompt, nhiều cấu hình | Đủ | — |
| Mục 7: zero-shot vs few-shot cùng tác vụ | Có | Thiết kế thí nghiệm ở §2.2 |
| Mục 8: prompting, fine-tuning, RAG; bảng so sánh | Có bảng + demo RAG | **C3:** hai cách nhận cùng câu dặn (§2.3) |
| Mục 9: hallucination, thiên lệch, giới hạn ngữ cảnh, phụ thuộc tác vụ và dữ liệu; một câu trả lời sai và cách kiểm chứng | Có câu sai + kiểm chứng; thiên lệch, ngữ cảnh trình bày bằng lời | **C4:** ý "phụ thuộc dữ liệu" lấy từ câu sai do nhãn nhiễu ở phần 10, không code thêm |
| Mục 10: thử nghiệm có kiểm soát, bảng kết quả, ví dụ đầu ra, phân tích lỗi | Có | Thiết kế ở §2.2, §2.4 |
| Yêu cầu thêm 1, 2: thư viện, môi trường | Slide 2 + phần 0 | — |
| Yêu cầu thêm 3: demo output thật trên slide | Cột demo | — |
| Yêu cầu thêm 4: thử nghiệm có kiểm soát (thiết lập, kết quả, phân tích) | Slide 12–13 | §2.2 |
| Yêu cầu thêm 5, 6: giải thích từng dòng; mở notebook khi slide không có code | Quy định chung | Notebook nộp chỉ chứa phần trình bày (§5) |
| Thời lượng 15–20 phút | **Chưa có cột thời gian** | **C5:** thêm cột thời gian, tổng ≤ 20 phút gồm cả lúc mở notebook. Bản khung trước cộng ra 22 phút |
| 4 sản phẩm nộp | Có checklist | **C6:** ghi đường dẫn cho từng ô (§6) |

---

## 2. Thiết kế từng phần cần chốt

### 2.1 Mục 5: base vs Instruct
- Cả hai bản truyền **giống nhau**: `do_sample=False`, `repetition_penalty=1.0`, cùng `max_new_tokens`.
  - Lý do: theo `generation_config.json` trên Hugging Face, bản Instruct mặc định `repetition_penalty: 1.1`, bản base không đặt tham số này (tức 1.0). Không truyền giống nhau thì bản base lặp nhiều hơn một phần vì tham số, không phải vì thiếu SFT (chính Toàn đã nêu ở A4).
- Slide 7 ghi 2 khác biệt không tránh được: base nhận văn bản thô, Instruct nhận chat template; hai bản dừng ở token kết thúc khác nhau.
- Lời thoại nói *"tác dụng gộp của SFT + alignment"*, không nói "hoàn toàn do SFT". Lý do: Qwen2.5-Instruct đã qua SFT, DPO và GRPO.

### 2.2 Mục 7, 10: zero-shot vs few-shot trên UIT-VSFC

**Số câu test: chạy toàn bộ tập test 2 nhãn nếu thời gian cho phép, không cố định 500.**
- **Dữ liệu** (Thư đọc parquet ngày 18/11): split test có **3,166 câu** = 1,590 positive + 1,409 negative + 167 neutral. Bỏ neutral thì còn **2,999 câu**. Câu ngắn: trung vị 11 từ, trung bình 14.2.
- **Luật chọn số câu:**
  1. Smoke run đo số giây mỗi câu cho cả 2 cách.
  2. Nếu 2 × 2,999 × (giây/câu) ≤ **20 phút** thì chạy **toàn bộ 2,999 câu**.
  3. Nếu không thì lấy **500 câu** (250/250, seed 42).
- **Vì sao:**
  - Chạy toàn bộ tập test thì khỏi phải giải thích cách lấy mẫu.
  - Sai số khoảng ±1.4 điểm % (500 câu là ±3.5, 100 câu là ±7.8, với accuracy khoảng 80%).
  - Code giống hệt nhau, chỉ khác thời gian chạy.
  - [Ước lượng] Câu trả lời chỉ vài token, mà log phần 1 cho thấy câu ngắn mất khoảng 0.3–0.4 giây.

**Thiết kế (chỉ đổi một biến là có ví dụ hay không):**
- Cùng model, cùng prompt template, greedy, `max_new_tokens` khoảng 5.
- **1 bộ few-shot** gồm 3 ví dụ lấy từ train, có cả hai nhãn, không trùng câu test, seed 42.
- Parse nhận cả nhãn tiếng Anh lẫn tiếng Việt. Câu không parse được thì ghi riêng, không tính là đúng.
- **Chọn prompt trên validation, tối đa 4 mẫu**, chốt rồi mới chạy test, không sửa sau khi đã xem kết quả test.

**Metric:**
- Accuracy, confusion matrix, tỉ lệ không parse được.
- **McNemar exact** cho 1 phép so sánh: báo b, c, p.
- Lý do giữ McNemar: Dietterich (1998) kết luận với thuật toán chỉ chạy được một lần, McNemar là kiểm định duy nhất có sai lầm loại I chấp nhận được. Code chỉ 1 lệnh. Toàn đã viết sẵn phần giải thích ở `noi_dung_toan_muc07_muc10_v2.md` §7, nên không tốn thêm công học.
- Không cần Bonferroni vì chỉ có 1 phép so sánh.

**Giới hạn ghi trên slide 12:** *"1 bộ ví dụ few-shot; đổi ví dụ hoặc thứ tự ví dụ có thể cho kết quả khác"*. Theo Zhao et al. (2021), chỉ đổi ví dụ và thứ tự đã có thể đẩy accuracy few-shot từ gần mức đoán ngẫu nhiên lên gần mức tốt nhất.

### 2.3 Mục 8: RAG
- Khoảng 15 câu hỏi về slide môn học. Mỗi câu ghi đáp án và slide nguồn.
- Lấy đoạn liên quan bằng TF-IDF, top-3, `ngram_range=(1, 2)`.
- **Hai cách nhận cùng một câu dặn:** *"Nếu không biết hoặc tài liệu không có thông tin, trả lời 'Không có thông tin'"*. Lý do: chỉ cách có RAG được dặn thì so khả năng từ chối không công bằng (chính Toàn đã nêu ở A7).
- Thêm 3 câu mà slide không có đáp án.
- **Chấm:** Đúng / Sai / Từ chối. Chấm tự động (output có chứa đáp án), sau đó Toàn đọc lại từng câu. Báo số câu đúng của mỗi cách và recall@3 (đoạn đúng có nằm trong top-3 không). **Bỏ** kappa, chấm mù, RAGAS.

### 2.4 Mục 10: phân tích lỗi
- **Ghi sẵn nhóm lỗi trước khi đọc output:**
  - (a) không parse được;
  - (b) nhãn gốc có vẻ sai;
  - (c) câu vừa khen vừa chê;
  - (d) ý ngầm hoặc mỉa mai;
  - (e) model sai rõ ràng.
- Lấy ngẫu nhiên 30 câu sai (seed 42), gán nhóm, ghi `docs/experiments/error_analysis_v1.md` và CSV.
- Lý do ghi nhóm trước: tránh chấm theo cảm tính sau khi đã thấy output.
- Nhóm (b) chính là bằng chứng cho ý "phụ thuộc dữ liệu" ở mục 9 (C4).

### 2.5 Mục 9: câu trả lời sai và cách kiểm chứng
- Khoảng 15 câu có đáp án và nguồn. Gồm fact ít phổ biến và **2–3 câu về thực thể không tồn tại**.
  - Lý do: đề yêu cầu phải có **một câu trả lời sai thật**. Câu về thực thể không tồn tại gần như chắc chắn cho ra câu bịa, nên chắc chắn có ví dụ.
- **Kiểm chứng** gồm: đối chiếu nguồn, và hỏi lại câu đó 2 lần với cách hỏi khác để xem câu trả lời có nhất quán không.
- Thiên lệch và giới hạn ngữ cảnh trình bày bằng lời (có nêu context tối đa 32,768 token lấy từ config), không demo.

---

## 3. Lịch và thứ tự ưu tiên

**Bỏ hạn 23h tối nay.** Thay bằng 2 mốc:
- **Tối nay 22h:** Toàn gửi **định dạng file** (tên cột, 2–3 dòng mẫu) cho `rag_questions.csv` (Muội) và `hallucination.csv` (Phúc). Toàn code phần 8, 9 trên dữ liệu mẫu đó.
- **9h sáng thứ 5:** Muội, Phúc nộp dữ liệu thật đúng định dạng. Phúc duyệt `base_vs_instruct.json`.
- Lý do: code không phụ thuộc vào nội dung câu hỏi, chỉ phụ thuộc định dạng file. Chốt định dạng trước thì Toàn không phải ngồi chờ, còn Muội, Phúc có thêm một buổi sáng.

**Thứ tự code ngày thứ 5:** phần **7 → 10 → 6 → 9 → 8**.
- Lý do: 7 và 10 là thí nghiệm chính, chiếm 2 slide và là yêu cầu thêm 4. Phần 8 nhiều bước nhất (chia đoạn, TF-IDF, 2 cách chạy) nên để cuối.
- Nếu đến 12h trưa thứ 6 phần nào chưa smoke run đạt thì slide mục đó chỉ còn lý thuyết và minh hoạ của đề.

| Ngày | Toàn | Muội | Phúc | Thư |
|---|---|---|---|---|
| **T4 18/11** tối | Sửa khung v4 và plan v2 theo file này; **22h** gửi định dạng file; tách nhánh `part/02-extension`; push phần 2–5; smoke run phần 2–5 | Slide mục 1; bảng so sánh mục 8 | Sơ đồ mục 5 | Gửi file chốt này |
| **T5 19/11** | **9h** nhận data; code và smoke run 7 → 10 → 6 → 9 → 8; **tối:** Save & Run All chính thức + tải HTML dự phòng; bàn giao phần 1–6 và **8, 9** | **9h** nộp `rag_questions.csv`; slide mục 2, 3; nhận bàn giao phần 1–3, 8 | **9h** nộp `hallucination.csv`, duyệt `base_vs_instruct.json`; slide mục 4, 5, 6; nhận bàn giao phần 4–6, 9 | Review slide |
| **T6 20/11** | **12h: cắt.** Nếu lần chạy tối thứ 5 lỗi thì chạy lại buổi sáng. Chiều: slide 2, mục 7, 10; bàn giao phần 7, 10; nếu dư giờ thì làm mở rộng (§5) | Thay số chính thức; slide mục 8; README; tài liệu tham khảo; **20h:** ghép slide | Slide mục 9; thay số chính thức | Review notebook và slide |
| **T6** tối | Tập toàn bài, bấm giờ | ← | ← | Bấm giờ, hỏi thử từng dòng code |
| **T7 21/11** sáng | Xuất PDF; nộp 4 sản phẩm | Tập phần mình | Tập phần mình | Kiểm checklist (§6) |

- **Bàn giao phần 8, 9 tối thứ 5**, vì Muội, Phúc phải giải thích được từng dòng (yêu cầu thêm 5).
- **Chạy chính thức tối thứ 5** để còn buổi sáng thứ 6 chạy lại nếu lỗi.
- **Mỗi người nộp 5 câu Q&A kèm đáp án** cho mục mình trước **18h thứ 6**, dùng cho buổi hỏi thử.

---

## 4. Toàn gửi ngay (tối nay)
1. `yeu_cau_cua_co_v1.md`, `khung_presentation_llms_v3.md`, `ke_hoach_18_21_11_v1.md`. Khung v4 dẫn tới các file này, nhưng Thư không tìm thấy trong `Discussion/present` hay trên GitHub.
2. Push code phần 2–5 lên các nhánh `part/02-…` → `part/05-…`. Plan ghi "code xong" nhưng GitHub chưa có.

## 5. Mở rộng (chỉ làm sau khi phần chính xong)
- Code cell 2.5–2.10 (so sánh BERT, VLM…) để trên nhánh **`part/02-extension`**, notebook nộp không có.
- Nếu còn giờ: làm phần so sánh thành **phụ lục ở cuối notebook**, hoặc một notebook riêng **chạy riêng**.
  - Lý do: không phải chạy lại Save & Run All chính thức, nên số liệu trên slide giữ nguyên `run_id`.
- Khi làm, ưu tiên các so sánh có kết quả **chưa biết trước** và **cùng phần LLM bên trong**:
  - BERT vs Qwen, thêm phép thử đổi token cuối;
  - nếu làm VLM thì dùng cặp Qwen2-1.5B-Instruct vs Qwen2-VL-2B.
- Slide mở rộng chỉ thêm nếu tổng thời gian vẫn ≤ 20 phút.

## 6. Sản phẩm nộp → đường dẫn
- [ ] Slide PDF/PPTX: `slides/`
- [ ] Notebook + hướng dẫn chạy: `notebook/llm_presentation.ipynb` + `README.md`
- [ ] Bộ câu thử nghiệm, kết quả, phân tích lỗi: `data/`, `outputs/results/`, `docs/experiments/error_analysis_v1.md`
- [ ] Tài liệu tham khảo: `docs/references.md`

## Nguồn
- Dietterich (1998), *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*, Neural Computation 10(7): https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf
- Zhao et al. (2021), *Calibrate Before Use*, ICML: https://arxiv.org/abs/2102.09690
- `generation_config.json` của Qwen2.5-1.5B và Qwen2.5-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/generation_config.json , https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/generation_config.json
- Số câu UIT-VSFC: Thư đọc file parquet test ngày 18/11 từ https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback

# Mục 04 — Nguyên lý học: next-token prediction, dữ liệu huấn luyện, cross-entropy (v1)

- **Created**: 2026-11-17T16:30:00+07:00
- **Last Updated**: 2026-11-17T16:30:00+07:00 (Toàn, AI hỗ trợ)
- **Người phụ trách mục** (dàn ý, lý thuyết, slide, thuyết trình): Phúc | **Code**: Toàn
- **Slide**: 8–9 | **Notebook**: phần 4 (cell 4.1–4.5) | **Thời lượng**: khoảng 2 phút
- **Status**: In Progress | **Duyệt**: chưa
- Ghi chú quy trình: **AI viết nháp theo yêu cầu của Toàn (17/11)**, sau khi đã có code phần 4. Cách làm "AI viết nháp → Phúc sửa, ký → Toàn duyệt" đang chờ Thư chốt (`plans/tra_loi_toan_cho_thu_v1.md` §I câu 4) [cần chốt].

---

## Đề của cô  [Người viết]
- Nội dung: Next-token prediction, dữ liệu huấn luyện và cross-entropy loss.
- Minh hoạ: Tạo các cặp ngữ cảnh và token mục tiêu từ một câu.

## Goals / Purpose  [Người viết]
- Output cell 4.1: bảng cặp ngữ cảnh → token mục tiêu của "Hôm nay trời rất đẹp.", dùng **token thật** của Qwen. Dùng cho slide 8.
- `outputs/results/per_token_loss.csv`: với mỗi vị trí, xác suất model gán cho token đúng và loss = −ln p, cho 3 câu (đúng trật tự, xáo trộn, tiếng Anh). Dùng cho slide 9.
- `outputs/results/loss_summary.csv`: loss trung bình tự tính, perplexity, `outputs.loss` của `transformers` cho 3 câu; loss tự tính khớp `outputs.loss` (lệch < 1e-3).
- Không làm:
  - Train hay fine-tune, gọi `loss.backward()`: không (góp ý của cô 11/11). Chỉ tính loss của model đã train.
  - Thống kê dữ liệu huấn luyện bằng code: không có dữ liệu gốc; "dữ liệu huấn luyện" chỉ trình bày lý thuyết (GPT-3, Qwen2.5).
  - So perplexity giữa 2 model khác tokenizer: sai cách so (HF *Perplexity*).
  - Sinh văn bản: thuộc mục 6.

## Backbone  [Người viết]
- `Qwen/Qwen2.5-1.5B-Instruct` (`model`, `tok`, cell 0.4; model card: `docs/models/qwen2.5-1.5b-instruct_v2.md`). Văn bản thô, không chat template.

## Input / Output  [Người viết dàn ý, AI chi tiết hoá]
- **Input:** `model`, `tok` (phần 0); `SENT_VI`, `SENT_EN` (cell 3.1); câu xáo trộn "đẹp rất trời nay Hôm." viết trong cell 4.5.
- **Output** (smoke: `outputs/_smoke/results/`):
  - `outputs/results/per_token_loss.csv`: `sentence_id, sentence, idx, context, target_token, target_id, prob_target, loss_token`.
  - `outputs/results/loss_summary.csv`: `sentence_id, sentence, n_tokens, loss_mean, perplexity, hf_loss` [đề xuất: file thêm, không có trong `project_description.md` §2.3].
- **Log:** `outputs/logs/<ts>_<mode>/part04_loss.log` và `part04_loss_records.csv`, cột `idx, id, context, target_token, prob_target, loss_token, time_s`; cuối log có `loss_mean`, perplexity, so với `outputs.loss` (RUN_LOG_TEMPLATE_v2 §3, phần 4).

## Lý thuyết cần trình bày (5W1H)  [Người viết dàn ý, AI chi tiết hoá]
| | |
|---|---|
| What | Next-token prediction: với mọi vị trí, đoán token kế tiếp; loss = cross-entropy = −ln P(token đúng), lấy trung bình |
| Why | Nhãn chính là token kế tiếp trong văn bản (tự giám sát), nên dùng được dữ liệu cực lớn không cần gán nhãn |
| When | Giai đoạn pretraining; SFT cũng dùng cùng loại loss (mục 5) |
| Where | Web, sách, code, Wikipedia; Qwen2.5 khoảng 18T token |
| Who | Mọi LLM decoder-only: GPT-3 (OpenAI), Qwen2.5 (Qwen Team) |
| How | Dịch nhãn sang trái 1 vị trí → softmax → −log p(token đúng) → trung bình. Nhờ causal mask, một lần chạy cho cả n − 1 dự đoán. Nhóm chỉ **tính** loss, không train |
Nguồn: `research/noi_dung_muc04_nguyen_ly_hoc_v2.md` §1–§4 [Bổ sung: Brown et al. 2020; Qwen2.5 Technical Report; HF *Perplexity of fixed-length models*]; Slide Ch2 s7, s9, s11–13.

## How to do it (general plan)  [Người viết]
1. Cell 4.1: `make_context_target_pairs(ids)`, in bảng ngữ cảnh → token mục tiêu.
2. Cell 4.2: `per_token_loss(model, tok, câu)`, in bảng P(token đúng) và loss từng vị trí.
3. Cell 4.3: so loss trung bình tự tính với `model(ids, labels=ids).loss`.
4. Cell 4.4: perplexity = exp(loss), so với mốc đoán bừa ln(151,936).
5. Cell 4.5: thêm câu xáo trộn và câu tiếng Anh; ghi `per_token_loss.csv`, `loss_summary.csv`; dòng END.
6. Smoke run: kiểm 2 file kết quả, log đủ 3 phần.
7. Normal run: điền slide 8, 9, ghi `run_id`.

## Detailed plan  [AI chi tiết hoá]
- **Câu ví dụ:** "Hôm nay trời rất đẹp." có 8 token Qwen, nên có 7 cặp (Qwen không thêm token đầu câu, nên token đầu "H" không có dự đoán). Số cặp theo token, không theo số từ.
- **Tính loss:** logits chuyển sang `float32` trước `log_softmax`; chạy trong `torch.no_grad()`, `model.eval()` (đã đặt ở cell 0.4).
- **Mốc đoán bừa:** loss = ln(`model.config.vocab_size`) = ln(151,936) ≈ 11.93, perplexity = 151,936 [Tự tính, research §4.2].
- **Cell / hàm:**
  - `make_context_target_pairs(token_ids) -> list[tuple[list[int], int]]` (4.1), có `assert` trên ví dụ `[1, 2, 3, 4]`.
  - `per_token_loss(model, tok, text) -> pd.DataFrame` (4.2): `idx, context, target_token, target_id, prob_target, loss_token`.
  - `hf_loss(model, tok, text) -> float` (4.3): `model(ids, labels=ids).loss`.
- **Gotchas:**
  - Đếm cặp phải dùng token thật: "Hôm" bị tách thành 2 token.
  - `loss_mean` phải là `float` Python trước khi đưa vào `torch.tensor` (lỗi "Double did not match Float" đã gặp và sửa khi chạy thử 17/11).
  - Không viết trước kết quả câu xáo trộn lên slide; ghi đúng như đo.

## Đánh giá  [Người viết]
- **Đánh giá cái gì:** hiểu đúng công thức cross-entropy và cách `transformers` dịch nhãn.
- **Đo bằng gì:** loss trung bình tự tính so với `outputs.loss`: `torch.allclose(..., atol=1e-3)`; loss so với mốc đoán bừa.
- **Thế nào là đạt:** lệch < 1e-3 ở cả 3 câu. Loss câu xáo trộn **không** phải điều kiện; ghi đúng số đo.
- **Output:** `outputs/results/loss_summary.csv` (`loss_mean`, `hf_loss`), `outputs/results/per_token_loss.csv` (`loss_token`).

## Success criteria  [Người viết]
- **Assert trong notebook:**
  - 4.1: hàm đúng trên ví dụ nhỏ; số cặp = số token − 1.
  - 4.2: đúng n − 1 vị trí; loss ≥ 0; xác suất trong [0, 1]; mục tiêu khớp các cặp ở 4.1.
  - 4.3, 4.5: loss tự tính khớp `outputs.loss` (atol 1e-3) ở cả 3 câu.
- **Smoke run:** 2 file `results/`; `part04_loss.log` có START, dòng mỗi mẫu, END + tổng kết; `run.log` có `[part04_loss] OK`.
- **Normal run:** slide 8 có bảng cặp; slide 9 có bảng loss từng token, loss trung bình, perplexity, mốc đoán bừa; ghi `run_id`.

## Kết quả audit (chưa audit)  [Toàn điền sau normal run]
- chưa audit
- Đã kiểm trước smoke run (17/11): chạy cục bộ trên CPU bằng model Qwen2 tí hon trọng số ngẫu nhiên và tokenizer thật; cell 4.1–4.5 chạy hết, loss tự tính khớp `outputs.loss` (lệch khoảng 1e-6). Số của lần chạy này **không** dùng cho slide.

## Slide
- Slide 8: **Next-token prediction và dữ liệu huấn luyện**.
  - Bảng cặp ngữ cảnh → token mục tiêu (token thật của Qwen).
  - Dòng dữ liệu: GPT-3 và Qwen2.5 học trên bao nhiêu token, nguồn nào (research §3).
  - Chân: "Notebook phần 4, cell 4.1 · run_id …".
- Slide 9: **Cross-entropy loss**.
  - Công thức L = −(1/N) Σ ln P(token đúng).
  - Bảng P và loss từng token; loss trung bình, perplexity, so với mốc đoán bừa 11.93.
  - Một dòng: loss tự tính = `outputs.loss`.
  - Chân: "Notebook phần 4, cell 4.2–4.5 · run_id …".

## Q&A dự kiến (khoảng 5 câu)
| Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|
| Nhãn lấy từ đâu khi pretraining? | Chính token tiếp theo trong văn bản, nên gọi là tự giám sát | research §1.2 |
| Câu n token cho mấy cặp? Có phải chạy n − 1 lần không? | n − 1 cặp; chạy một lần, nhờ causal mask mỗi vị trí cho một dự đoán | research §2.2; cell 4.2 |
| `labels=input_ids` có bị lệch không? | Không; model tự dịch nhãn sang trái 1 vị trí. Cell 4.3 kiểm bằng cách tự tính và so | HF Perplexity; cell 4.3 |
| Perplexity là gì, bao nhiêu là tốt? | exp(loss), như số lựa chọn đều mà model đang phân vân; so với mốc đoán bừa 151,936 | HF Perplexity; cell 4.4 |
| Vì sao không tự train để minh hoạ? | Cô góp ý không cần fine-tune; tính loss của model đã train dùng đúng công thức lúc huấn luyện | Góp ý 11/11 |

## Links
- Description cả bài: [../presentation_description_v2.md](../presentation_description_v2.md)
- Research: [../research/noi_dung_muc04_nguyen_ly_hoc_v2.md](../research/noi_dung_muc04_nguyen_ly_hoc_v2.md)
- Detail plan notebook: [../plans/detail_plan_notebook_v1.md](../plans/detail_plan_notebook_v1.md) phần 4
- Tiến độ và bàn giao code: [../../progress_status.md](../../progress_status.md)
- Notebook: `notebook/llm_presentation.ipynb`, phần 4 (cell 4.1–4.5)

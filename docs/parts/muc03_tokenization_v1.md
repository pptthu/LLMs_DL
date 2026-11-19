# Mục 03 — Tokenization: token, token ID và embedding (v1)

- **Created**: 2026-11-17T16:30:00+07:00
- **Last Updated**: 2026-11-17T16:30:00+07:00 (Toàn, AI hỗ trợ)
- **Người phụ trách mục** (dàn ý, lý thuyết, slide, thuyết trình): Muội | **Code**: Toàn
- **Slide**: 7 | **Notebook**: phần 3 (cell 3.1–3.5) | **Thời lượng**: khoảng 1.5 phút
- **Status**: In Progress | **Duyệt**: chưa
- Ghi chú quy trình: **AI viết nháp theo yêu cầu của Toàn (17/11)**, sau khi đã có code phần 3. Cách làm "AI viết nháp → Muội sửa, ký → Toàn duyệt" đang chờ Thư chốt (`plans/tra_loi_toan_cho_thu_v1.md` §I câu 4) [cần chốt].

---

## Đề của cô  [Người viết]
- Nội dung: Cách văn bản được chia thành token; token ID và embedding.
- Minh hoạ: Tokenize một câu tiếng Việt và một câu tiếng Anh.

## Goals / Purpose  [Người viết]
- `outputs/results/token_counts.csv`: số ký tự, số byte UTF-8, số token của 2 câu ("Hôm nay trời rất đẹp.", "The weather is very nice today.") × 2 tokenizer (GPT-2, Qwen2.5).
- Output cell 3.1: bảng token, token ID của 2 câu với Qwen và của câu tiếng Việt với GPT-2. Dùng cho slide 7.
- Output cell 3.3, 3.4: kích thước từ điển của 2 tokenizer; bảng embedding 151,936 × 1,536 và shape embedding của câu ví dụ.
- Không làm:
  - Tự huấn luyện tokenizer BPE: chỉ giải thích ý tưởng trên slide, dùng tokenizer có sẵn.
  - Xem embedding có "gần nghĩa thì gần nhau" không: không đo; chỉ nói ở mức lý thuyết [Slide Ch6 s17].
  - Đo tốc độ sinh theo số token: thuộc mục 6.
  - Giới hạn ngữ cảnh 32,768 token: thuộc mục 9; mục 3 chỉ nhắc một câu.

## Backbone  [Người viết]
- Tokenizer `Qwen/Qwen2.5-1.5B-Instruct` (`tok`, cell 0.4) và bảng embedding của model chính (model card: `docs/models/qwen2.5-1.5b-instruct_v2.md`).
- Tokenizer `openai-community/gpt2` (`gpt2_tok`, cell 0.6; model card: `docs/models/gpt2-tokenizer_v2.md`). Chỉ tải tokenizer, không tải model GPT-2.

## Input / Output  [Người viết dàn ý, AI chi tiết hoá]
- **Input:** `tok`, `gpt2_tok`, `model` (phần 0); 2 câu viết thẳng trong cell 3.1 (`SENT_VI`, `SENT_EN`), dùng lại ở phần 4.
- **Output:** `outputs/results/token_counts.csv` (smoke: `outputs/_smoke/results/`), cột `sentence, lang, tokenizer, n_chars, n_bytes, n_tokens, tokens` (`tokens` nối bằng `|`).
- **Log:** `outputs/logs/<ts>_<mode>/part03_tokenization.log` và `part03_tokenization_records.csv`, cột `idx, id, sentence, lang, tokenizer, n_tokens, tokens, time_s` (RUN_LOG_TEMPLATE_v2 §3, phần 3).

## Lý thuyết cần trình bày (5W1H)  [Người viết dàn ý, AI chi tiết hoá]
| | |
|---|---|
| What | Chia văn bản thành token (subword), gán mỗi token một số nguyên (token ID), tra hàng tương ứng trong bảng embedding |
| Why | Model chỉ tính trên số; subword cân bằng giữa kích thước từ điển và độ dài chuỗi, xử lý được từ hiếm |
| When | BPE cho dịch máy: 2016; byte-level BPE: GPT-2, 2019; Qwen2.5: 2024 |
| Where | Bước đầu tiên trước khi vào model; ảnh hưởng số token, tốc độ sinh, giới hạn ngữ cảnh |
| Who | Sennrich et al. (BPE cho NMT); OpenAI (GPT-2); Qwen Team |
| How | `tokenizer(text, add_special_tokens=False)["input_ids"]` → `model.get_input_embeddings()(ids)` → tensor (1, n, 1536) |
Nguồn: `research/noi_dung_muc03_tokenization_v2.md` §1–§4 [Bổ sung: HF *Summary of the tokenizers*; Radford et al. 2019]; Slide Ch6 s17.

## How to do it (general plan)  [Người viết]
1. Cell 3.1: `tokenize_text()` in bảng token cho 2 câu với Qwen, câu Việt với GPT-2.
2. Cell 3.2: `compare_token_counts()` cho 2 câu × 2 tokenizer, ghi `token_counts.csv`.
3. Cell 3.3: in 3 con số kích thước từ điển của Qwen và 1 của GPT-2.
4. Cell 3.4: shape bảng embedding, shape embedding của câu; kiểm embedding = tra hàng.
5. Cell 3.5: decode lại câu gốc với cả 2 tokenizer; dòng END.
6. Smoke run: kiểm `token_counts.csv` 4 dòng, log đủ 3 phần.
7. Normal run: điền slide 7 từ `token_counts.csv`, ghi `run_id`.

## Detailed plan  [AI chi tiết hoá]
- **Đếm token:** `add_special_tokens=False` cho cả 2 tokenizer, để so công bằng (`research/noi_dung_muc03_tokenization_v2.md` §5).
- **3 con số từ điển của Qwen** (đọc bằng code ở cell 3.3, không chép tay):
  - `tok.vocab_size`: token thường;
  - `len(tok)`: thêm token đặc biệt (`<|im_start|>`, `<|im_end|>`…);
  - `model.config.vocab_size`: số hàng của bảng embedding, lớn hơn `len(tok)`; hàng dư không ứng với token nào. Lý do có hàng dư: [cần kiểm nguồn] trước khi nói trên slide.
- **Hiển thị token:** cột `piece` (token thô, "Ġ" là dấu cách, chữ có dấu thành ký tự byte lạ) và cột `text` (decode riêng từng token; "�" là token chỉ chứa một phần byte của một chữ).
- **Cell / hàm:**
  - `tokenize_text(tokenizer, text) -> pd.DataFrame` (`position, token_id, piece, text`) (3.1).
  - `compare_token_counts(tokenizers: dict, sentences: dict) -> pd.DataFrame` (3.2).
- **Gotchas:**
  - `len(tokenizer(câu)["input_ids"])` có thể đã gồm token đặc biệt; luôn truyền `add_special_tokens=False`.
  - Qwen vẫn tách "Hôm" thành "H" + "ôm" (kiểm 17/11): tokenizer học theo tần suất, không theo từ điển tiếng Việt.
  - Không dùng tokenizer GPT-2 để đưa vào model Qwen: token ID hai bên khác nhau. GPT-2 ở đây chỉ để so số token.

## Đánh giá  [Người viết]
- **Đánh giá cái gì:** số token của câu Việt và câu Anh với 2 tokenizer; tokenizer có mã hoá hai chiều đúng không; embedding đúng là tra hàng không.
- **Đo bằng gì:** `len(input_ids)` với `add_special_tokens=False`; `decode(encode(câu)) == câu`; `torch.equal(embedding(ids), weight[ids])`.
- **Thế nào là đạt:** có đủ bảng 2 câu × 2 tokenizer; decode đúng 4/4; embedding đúng là các hàng của bảng. Không đặt điều kiện "câu Việt phải tốn token hơn": ghi đúng số đo.
- **Output:** `outputs/results/token_counts.csv`, cột `n_tokens`.

## Success criteria  [Người viết]
- **Assert trong notebook:**
  - 3.1: ghép chữ các token Qwen ra đúng câu Việt.
  - 3.2: 4 dòng, `n_tokens > 0`.
  - 3.3: GPT-2 = 50,257; `tok.vocab_size ≤ len(tok) ≤ model.config.vocab_size`.
  - 3.4: shape bảng = (vocab, hidden) theo `ARCH`; embedding = tra hàng; dùng chung với LM head đúng như `tie_word_embeddings`.
  - 3.5: decode đúng câu gốc 4/4.
- **Smoke run:** `outputs/_smoke/results/token_counts.csv` 4 dòng; `part03_tokenization.log` có START, 4 dòng mỗi mẫu, END + tổng kết; `run.log` có `[part03_tokenization] OK`.
- **Normal run:** slide 7 lấy số token và bảng token từ file kết quả, ghi `run_id`.

## Kết quả audit (chưa audit)  [Toàn điền sau normal run]
- chưa audit
- Đã kiểm trước smoke run (17/11): chạy cục bộ trên CPU với tokenizer thật (Qwen, GPT-2) và model Qwen2 tí hon; cell 3.1–3.5 chạy hết, mọi `assert` đạt. Số của lần chạy này **không** dùng cho slide.

## Slide
- Slide 7: **Tokenization, token ID, embedding**.
  - Trái: câu Việt và câu Anh sau khi tokenize bằng Qwen (token + ID).
  - Giữa: bảng 2 câu × 2 tokenizer, số token từ `token_counts.csv`.
  - Phải: luồng token ID → bảng embedding 151,936 × 1,536 → tensor (1, n, 1536).
  - Chân: "Notebook phần 3, cell 3.1–3.4 · run_id …".

## Q&A dự kiến (khoảng 5 câu)
| Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|
| Vì sao không chia theo từ? | Từ điển phình to, bảng embedding khổng lồ; từ chưa gặp thành `<unk>` | HF tokenizer summary |
| BPE học thế nào? | Bắt đầu từ byte, lặp lại việc gộp cặp liền kề phổ biến nhất cho tới đủ kích thước từ điển | HF tokenizer summary |
| Vì sao câu tiếng Việt tốn token hơn với GPT-2? | Chữ có dấu chiếm 2–3 byte UTF-8; GPT-2 học chủ yếu từ tiếng Anh nên ít gộp cặp byte tiếng Việt. Nói theo số đo ở cell 3.2 | research §3.3; cell 3.2 |
| Dùng tokenizer GPT-2 cho Qwen được không? | Không; token ID hai bên khác nhau, model học với tokenizer nào phải dùng tokenizer đó | research §3.1 |
| Bảng embedding của Qwen bao nhiêu tham số? | 151,936 × 1,536 ≈ 233 triệu, dùng chung với LM head (cell 3.4 kiểm bằng địa chỉ bộ nhớ) | cell 3.4; `config.json` |

## Links
- Description cả bài: [../presentation_description_v2.md](../presentation_description_v2.md)
- Research: [../research/noi_dung_muc03_tokenization_v2.md](../research/noi_dung_muc03_tokenization_v2.md)
- Detail plan notebook: [../plans/detail_plan_notebook_v1.md](../plans/detail_plan_notebook_v1.md) phần 3
- Tiến độ và bàn giao code: [../../progress_status.md](../../progress_status.md)
- Notebook: `notebook/llm_presentation.ipynb`, phần 3 (cell 3.1–3.5)

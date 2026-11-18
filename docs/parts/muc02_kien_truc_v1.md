# Mục 02 — Nền tảng kiến trúc: Transformer, Decoder-only, causal attention (v1)

- **Created**: 2026-11-17T16:30:00+07:00
- **Last Updated**: 2026-11-17T16:30:00+07:00 (Toàn, AI hỗ trợ)
- **Người phụ trách mục** (dàn ý, lý thuyết, slide, thuyết trình): Muội | **Code**: Toàn
- **Slide**: 5, 6, 6b | **Notebook**: phần 2 (cell 2.1–2.10) | **Thời lượng**: khoảng 3.5 phút (1.5' + 1' + 1')
- **Status**: In Progress | **Duyệt**: chưa
- Ghi chú quy trình: `rules/RULES_v2.md` §1 quy định người phụ trách tự viết dàn ý. Bản này **AI viết nháp theo yêu cầu của Toàn (17/11)**, viết sau khi đã có code phần 2. Cách làm "AI viết nháp → Muội sửa, ký → Toàn duyệt" đang chờ Thư chốt (`plans/tra_loi_toan_cho_thu_v1.md` §I câu 4) [cần chốt].
- Muội cần làm: đọc từng mục, sửa chỗ chưa đúng ý mình, đặc biệt **Goals**, **Đánh giá** và **Slide**; ký tên ở "Last Updated".

---

## Đề của cô  [Người viết]
- Nội dung: Vai trò của Transformer; tập trung vào Decoder-only và causal attention.
- Minh hoạ: Sơ đồ từ token đầu vào đến xác suất token tiếp theo.
- Thêm (quyết định của Toàn, 16/11, điểm cộng): so sánh LLM với RNN/LSTM, encoder-only (BERT), VLM, **phải có demo**.

## Goals / Purpose  [Người viết]
- `outputs/results/model_config.json`: thông số kiến trúc Qwen2.5-1.5B-Instruct đọc từ `model.config`, kèm số tham số theo thành phần (embedding, attention, FFN, norm). Dùng cho sơ đồ slide 5.
- `outputs/figures/causal_mask.png` (ma trận quy tắc) và `outputs/figures/attention_qwen.png` (trọng số attention **thật**, layer 0, head 0) trên câu "Hôm nay trời rất đẹp.", chia theo token thật của Qwen. Dùng cho slide 6.
- `outputs/results/top5_next_tokens.csv`: 5 token tiếp theo có xác suất cao nhất sau "Hôm nay trời rất". Dùng cho slide 6.
- Slide 6b gồm:
  - `compare_bert_qwen.csv`: mBERT điền `[MASK]` (thấy hai phía) so với Qwen (chỉ thấy bên trái);
  - `compare_causal_swap.csv`: đổi token cuối "đẹp" → "tối", hidden state của Qwen ở các vị trí trước không đổi, của mBERT thì đổi;
  - `compare_vlm.csv`: cấu trúc Qwen2-VL-2B (bộ mã hoá ảnh + LLM) [cần chốt];
  - `compare_rnn_transformer.csv`: thời gian LSTM so với một lớp Transformer [cần chốt].
- Không làm:
  - Chia câu thành token, token ID, embedding: thuộc mục 3. Ở đây chỉ dùng kết quả chia token.
  - Tính loss, cặp ngữ cảnh → token mục tiêu: thuộc mục 4.
  - Sinh văn bản nhiều bước, greedy / temperature: thuộc mục 6. Ở đây chỉ lấy xác suất **một** bước.
  - Chạy VLM trên ảnh thật: chưa làm. Chỉ đọc cấu trúc (phương án (a) của Toàn); phương án (c) chỉ làm nếu Thư chọn và còn thời gian.
  - Train LSTM hay Transformer: không train (góp ý của cô 11/11). Demo LSTM chỉ đo cách tính, trọng số ngẫu nhiên.

## Backbone  [Người viết]
- `Qwen/Qwen2.5-1.5B-Instruct`, fp16, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (model card: `docs/models/qwen2.5-1.5b-instruct_v2.md`, có sơ đồ kiến trúc). Dùng lại `model`, `tok` của cell 0.4. Cell 2.3 nạp thêm một bản cùng model với `attn_implementation="eager"`, xong thì xoá.
- `google-bert/bert-base-multilingual-cased` (mBERT, encoder-only; model card: `docs/models/bert-base-multilingual-cased_v1.md`).
- `Qwen/Qwen2-VL-2B-Instruct`: **chỉ config**, dựng khung trên thiết bị `meta`, không tải trọng số (model card: `docs/models/qwen2-vl-2b-instruct_v1.md`).
- `torch.nn.LSTM`, `torch.nn.TransformerEncoderLayer`: trọng số ngẫu nhiên, không phải model đã train.

## Input / Output  [Người viết dàn ý, AI chi tiết hoá]
- **Input:**
  - `model`, `tok`, `ARCH`: lấy từ phần 0 (cell 0.4, 0.5).
  - mBERT: Hugging Face Hub, tải bằng `pipeline("fill-mask", model=...)` ở cell 2.5.
  - Config Qwen2-VL-2B: Hugging Face Hub, đọc bằng `AutoConfig.from_pretrained(...)` ở cell 2.8.
  - Câu ví dụ: viết thẳng trong cell (`MASK_TEXT`, `TOPK_TEXT`, `DEMO_A`, `SWAP_TEXTS`). Không dùng `data/compare_prompts.json` và `data/vlm_image.png` vì không chạy VLM trên ảnh.
- **Output** (normal run: `outputs/`; smoke run: `outputs/_smoke/`):
  - `results/model_config.json`: `model, revision, architectures`, các khoá của `ARCH`, `head_dim, q_heads_per_kv_head, rope_theta, n_params_by_group, n_params_without_embedding`.
  - `figures/causal_mask.png`, `figures/attention_qwen.png`.
  - `results/top5_next_tokens.csv`: `text, n_input_tokens, rank, token_id, token, prob`.
  - `results/compare_bert_qwen.csv`: `model, input_seen, rank, token, prob` (thêm dòng `rank = "đáp án"`).
  - `results/compare_causal_swap.csv`: `model, position, token, max_abs_diff, is_changed_token`.
  - `results/compare_vlm.csv`: `item, qwen2.5_1.5b_instruct, qwen2_vl_2b`.
  - `results/compare_rnn_transformer.csv`: `model, seq_len, n_params, sequential_steps, ms_mean`.
- **Log:** `outputs/logs/<ts>_<mode>/part02_arch.log` và `part02_arch_records.csv`, cột `idx, id, text, position, token, top_k_rank, top_k_token, top_k_prob, time_s` (RUN_LOG_TEMPLATE_v2 §3, phần 2).

## Lý thuyết cần trình bày (5W1H)  [Người viết dàn ý, AI chi tiết hoá]
| | |
|---|---|
| What | Transformer decoder-only với masked (causal) self-attention: mỗi token chỉ nhìn chính nó và các token trước |
| Why | RNN nén cả câu vào vector cố định, chạy tuần tự, khó nhớ xa. Mask để lúc huấn luyện model không "chép đáp án", và để cách chạy lúc huấn luyện giống lúc sinh |
| When | 2017 (*Attention Is All You Need*); decoder-only phổ biến từ GPT |
| Where | Mọi decoder block của GPT, Llama, Qwen; Qwen2.5-1.5B có 28 block |
| Who | Vaswani et al. (Google), 2017 |
| How | Q·Kᵀ/√d_k + mask (−∞ ở vị trí tương lai) → softmax → ·V; nhiều head (Qwen: 12 head Q, 2 head KV, GQA); residual + RMSNorm; vị trí bằng RoPE; token cuối → LM head → logits trên 151,936 token → softmax |
Nguồn: `research/noi_dung_muc02_kien_truc_v2.md` §1–§5, §11 [Bổ sung: Vaswani et al. 2017; HF LLM Course; model card Qwen2.5; Devlin et al. 2019; Wang et al. 2024 (Qwen2-VL)]; Slide Ch6 s11–s16.

## How to do it (general plan)  [Người viết]
1. Cell 2.1: ghi `model_config.json` từ `ARCH`, đếm tham số theo nhóm tên tham số.
2. Cell 2.2: chia câu ví dụ bằng `tok`, dựng `causal_mask(n)`, vẽ `causal_mask.png`.
3. Cell 2.3: nạp bản eager, `output_attentions=True`, vẽ `attention_qwen.png` (layer 0, head 0), rồi xoá bản eager.
4. Cell 2.4: `top_k_next_tokens(model, tok, "Hôm nay trời rất", k=5)`, ghi `top5_next_tokens.csv`.
5. Cell 2.5–2.6: mBERT điền `[MASK]` trong "Thủ đô của [MASK] là Paris."; Qwen chỉ nhận "Thủ đô của"; ghi `compare_bert_qwen.csv`.
6. Cell 2.7: đổi token cuối "đẹp" → "tối", so hidden state lớp cuối của cả hai model; ghi `compare_causal_swap.csv`.
7. Cell 2.8: dựng khung Qwen2-VL-2B trên `meta`, đếm tham số theo thành phần; ghi `compare_vlm.csv` [cần chốt].
8. Cell 2.9: đo thời gian LSTM so với một lớp Transformer ở 128 / 512 / 1,024 token; ghi `compare_rnn_transformer.csv` [cần chốt].
9. Cell 2.10: xoá mBERT, ghi `_records.csv`, dòng END.
10. Smoke run (`SMOKE = True`): kiểm 6 file `results/`, 2 hình, log có START / mỗi mẫu / END.
11. Normal run (Save & Run All): điền slide 5, 6, 6b từ file kết quả, ghi `run_id`.

## Detailed plan  [AI chi tiết hoá]
- **Câu ví dụ:** "Hôm nay trời rất đẹp." (8 token Qwen: `H`, `ôm`, ` n`, `ay`, ` trời`, ` rất`, ` đẹp`, `.`; đã kiểm bằng tokenizer thật 17/11). Qwen tách "Hôm" thành 2 token, nên hình có 8 hàng chứ không phải 5.
- **Văn bản thô, không chat template** ở 2.4, 2.6, 2.7: để thấy đúng xác suất token tiếp theo của chuỗi, không lẫn system prompt.
- **Layer, head ở 2.3:** layer 0, head 0 [đề xuất]; đổi được bằng `ATTN_LAYER`, `ATTN_HEAD`.
- **Demo A, đáp án 1 token:** "Pháp" là 1 token của mBERT (`assert` ở 2.5). Cặp đổi token cuối chọn **"đẹp" / "tối"** vì cả hai là 1 token ở cả mBERT và Qwen. "xấu" bị mBERT tách thành `x` + `##ấu` nên không dùng được (kiểm 17/11).
- **So hidden state ở 2.7:** `torch.allclose(..., atol=1e-3, rtol=1e-3)` trên các vị trí **trước** token bị đổi. mBERT giữ `[CLS]`, `[SEP]` vì đó là cách mBERT được dùng.
- **VLM ở 2.8:** `with torch.device("meta"): Qwen2VLForConditionalGeneration(cfg)`. Chạy thử cục bộ 17/11 với transformers 5.20.2: visual 665,271,296 + LLM 1,543,714,304 = 2,208,985,600 tham số, khớp số Thư ghi (`plans/gop_y_thu_cho_toan_v1.md` §4.2). Số này đếm từ config, không phải output model.
- **RNN ở 2.9:** `RNN_D = 512`, `nhead = 8`, `dim_feedforward = 2048`, `RNN_REPEAT = 3` (smoke) / 20 (normal), khởi động 2 lần trước khi đo, `cuda_sync()` trước và sau khi bấm giờ.
- **Cell / hàm:**
  - `cuda_sync()`, `vram_used_gb()`, `free_vram()` (2.1): dùng lại ở phần 3–5.
  - `causal_mask(n) -> torch.Tensor` (bool, tam giác dưới), `token_labels(tok, ids) -> list[str]` (2.2).
  - `top_k_next_tokens(model, tok, text, k=5) -> tuple[pd.DataFrame, list[int]]` (2.4).
  - `last_hidden(mdl, tokenizer, text, add_special_tokens) -> tuple[list[int], torch.Tensor]` (2.7).
  - `time_ms(fn, repeat) -> float` (2.9).
- **Gotchas:**
  - SDPA (mặc định) không trả trọng số attention; phải nạp `attn_implementation="eager"` (`research/research_presentation_llm_v1.md` §1).
  - Nạp bản eager tốn thêm khoảng 3.1 GB VRAM; xoá ngay sau 2.3 để phần 5 còn chỗ nạp bản base.
  - Logits có 151,936 cột, nhiều hơn `len(tok)` = 151,665; các hàng dư vẫn nhận xác suất rất nhỏ, không ảnh hưởng top-5.
  - LSTM trên GPU dùng cuDNN, đã tối ưu; kết quả đo có thể không cho thấy LSTM chậm hơn ở chuỗi ngắn. Ghi đúng như đo.
  - Phần LLM của Qwen2-VL là **thế hệ Qwen2**, khác trọng số Qwen2.5 dù cùng cấu hình (`plans/gop_y_thu_cho_toan_v1.md` §4.3).

## Đánh giá  [Người viết]
- **Đánh giá cái gì:** causal mask có tác dụng trong model thật không; output là phân phối xác suất; khác biệt giữa nhìn hai chiều và chỉ nhìn quá khứ.
- **Đo bằng gì:**
  - Mask: `assert` tam giác dưới; trọng số attention thật phần trên đường chéo = 0, mỗi hàng cộng lại ≈ 1 (`atol=1e-2`).
  - Top-5: xếp giảm dần, tổng ≤ 1.
  - Demo A: top-5 + xác suất của mBERT và Qwen; xác suất mỗi model gán cho đáp án "Pháp".
  - Đổi token cuối: chênh lệch lớn nhất (`max_abs_diff`) của hidden state ở các vị trí trước token bị đổi.
- **Thế nào là đạt:**
  - Mọi `assert` của 2.1–2.9 đạt; Qwen "không đổi" ở phép thử đổi token cuối (tính chất toán học của causal mask, nên có `assert`).
  - mBERT: ghi đúng số đo, **không** đặt điều kiện. Thứ hạng của "Pháp" cũng ghi đúng như thật.
  - Có output thật cho mọi demo.
- **Output:** 6 file `results/` và 2 hình ở mục Input / Output.

## Success criteria  [Người viết]
- **Assert trong notebook:**
  - 2.1: tổng các nhóm = `ARCH["n_params"]`; embedding = vocab × hidden.
  - 2.2: mask tam giác dưới.
  - 2.3: đủ 28 layer; phần trên đường chéo = 0; mỗi hàng ≈ 1.
  - 2.4: 5 dòng, giảm dần, tổng ≤ 1.
  - 2.5: đáp án là 1 token mBERT.
  - 2.6: 12 dòng.
  - 2.7: hai câu khác đúng 1 token; Qwen không đổi.
  - 2.8: tổng 2,208,985,600; số chiều token ảnh = hidden LLM.
  - 2.9: đủ 6 dòng, thời gian > 0.
  - 2.10: đủ 6 file và 2 hình.
- **Smoke run:** `outputs/_smoke/results/` có 6 file, `outputs/_smoke/figures/` có 2 hình; `part02_arch.log` có START, dòng mỗi mẫu, END + tổng kết; `run.log` có dòng `[part02_arch] OK`.
- **Normal run:** slide 5 có bảng thông số từ `model_config.json`; slide 6 có `attention_qwen.png` và bảng top-5; slide 6b có số từ `compare_bert_qwen.csv` và `compare_causal_swap.csv`; mọi số ghi `run_id`.

## Kết quả audit (chưa audit)  [Toàn điền sau normal run]
- chưa audit
- Đã kiểm trước smoke run (17/11): chạy cục bộ trên CPU bằng model Qwen2 tí hon trọng số ngẫu nhiên, cùng tokenizer thật, mBERT thật, config VLM thật. Cell 2.1–2.10 chạy hết, mọi `assert` đạt. Số của lần chạy này **không** dùng cho slide.

## Slide
- Slide 5: **Transformer và Decoder-only**.
  - Trái 3 dòng: Transformer bỏ hồi quy, chỉ dùng attention; ba họ (encoder-only, decoder-only, encoder-decoder); LLM hiện nay là decoder-only.
  - Phải: sơ đồ token → embedding 1,536 chiều → 28 decoder block → LM head → logits 151,936 → softmax, số lấy từ `model_config.json`.
  - Chân: "Notebook phần 2, cell 2.1".
- Slide 6: **Causal attention: chỉ nhìn quá khứ**.
  - Công thức `softmax(QKᵀ/√d_k + M)·V`.
  - `attention_qwen.png`; bảng top-5 sau "Hôm nay trời rất".
  - Chân: "Notebook phần 2, cell 2.2–2.4 · run_id …".
- Slide 6b: **LLM so với các mô hình khác**.
  - Bảng so sánh RNN/LSTM, BERT, LLM, VLM (`research/noi_dung_muc02_kien_truc_v2.md` §11.1).
  - Demo A: top-1 của mBERT và Qwen; một dòng "đổi token cuối: Qwen 0, mBERT …".
  - Một dòng VLM = bộ mã hoá ảnh + LLM [cần chốt]; một dòng thời gian LSTM / Transformer [cần chốt].
  - Chân: "Notebook phần 2, cell 2.5–2.9 · run_id …".

## Q&A dự kiến (khoảng 5 câu)
| Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|
| Vì sao gán −∞ mà không gán 0? | e⁰ = 1 vẫn cho trọng số dương; chỉ −∞ cho trọng số đúng bằng 0. Hình 2.3 cho thấy phần trên đường chéo bằng 0 | `research/noi_dung_muc02_kien_truc_v2.md` §4.3; cell 2.3 |
| 1.5 tỉ tham số nằm ở đâu? | Đọc `n_params_by_group` trong `model_config.json`: FFN chiếm phần lớn mỗi block, embedding khoảng 15% | cell 2.1; research §5.2 |
| Làm sao biết Qwen chỉ nhìn quá khứ? | Đổi token cuối thì hidden state các vị trí trước của Qwen không đổi; của mBERT thì đổi | cell 2.7 |
| GQA là gì? | 12 head Q dùng chung 2 cặp K, V (6 head Q một cặp); giảm bộ nhớ KV cache khi sinh | model card; cell 2.1 (`q_heads_per_kv_head`) |
| VLM khác LLM ở đâu? | VLM thêm bộ mã hoá ảnh, đưa ảnh thành token có cùng số chiều với LLM; phần sinh chữ vẫn là LLM decoder-only | cell 2.8; Wang et al. 2024 |
| Vì sao LLM không dùng LSTM? | LSTM phải chạy tuần tự từng bước, khó train song song trên dữ liệu rất lớn; demo 2.9 chỉ minh hoạ cách tính, không nói chất lượng | Slide Ch6; Vaswani et al. 2017; cell 2.9 |

## Links
- Description cả bài: [../presentation_description_v2.md](../presentation_description_v2.md)
- Research: [../research/noi_dung_muc02_kien_truc_v2.md](../research/noi_dung_muc02_kien_truc_v2.md)
- Góp ý và trả lời về demo VLM / RNN: [../plans/gop_y_thu_cho_toan_v1.md](../plans/gop_y_thu_cho_toan_v1.md) §4, [../plans/tra_loi_toan_cho_thu_v1.md](../plans/tra_loi_toan_cho_thu_v1.md) §D
- Detail plan notebook: [../plans/detail_plan_notebook_v1.md](../plans/detail_plan_notebook_v1.md) phần 2 (số cell đã khác: code hiện tại có 2.1–2.10)
- Tiến độ và bàn giao code: [../../progress_status.md](../../progress_status.md)
- Notebook: `notebook/llm_presentation.ipynb`, phần 2 (cell 2.1–2.10)

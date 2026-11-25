# Lý thuyết bộ metric đánh giá (slide 21–23 và các mục dùng metric)

- **Created**: 2026-11-25 (Toàn, AI hỗ trợ). Dùng để ôn và trả lời Q&A cho phần đánh giá; lời nói trên slide ở `docs/scripts/kich_ban_toan_v5.md`.
- **Phạm vi:** 18 metric trong danh sách tham khảo của cô + các metric nhóm thêm (CI 95%, McNemar, confusion matrix, top-k accuracy, distinct-n, recall@3, chấm 3 mức). Mỗi metric: là gì · công thức · ví dụ tính tay trên số thật · cài đặt trong notebook · kết quả · giới hạn.
- **Số liệu:** run chính thức `20261125_214650_normal`, đọc từ `outputs/results/`. Ví dụ tính tay đều kiểm lại được bằng file kết quả.
- **Ký hiệu nguồn:** `[Slide ChX]` = slide môn học; `[#n]` = mục n trong `docs/references.md`; `[Bổ sung]` = kiến thức ngoài, đã ghi nguồn; `[cần kiểm]` = chưa mở lại nguồn gốc, phải kiểm trước khi trích.

---

## 0. Bản đồ: metric nào đo ở đâu

| Nhóm | Metric | Mục / cell | Input → output, so với gì | n | File kết quả |
|---|---|---|---|---|---|
| Dự đoán ngôn ngữ | Cross-entropy, Perplexity | 4 / 4.2–4.5 | câu → P(token đúng) mỗi vị trí | 3 câu | `loss_summary.csv`, `per_token_loss.csv` |
| | *Top-1 / top-5 accuracy (thêm)* | 4 / 4.6 | như trên | 3 câu | `next_token_accuracy.csv` |
| Độ đúng, trùng khớp | Accuracy (+ CI 95%, McNemar, confusion matrix) | 7, 10 / 7.3–7.5, 10.1–10.2 | câu phản hồi → nhãn, so nhãn gốc | 2,999 | `icl_metrics.json` |
| | Exact Match, Token F1 | 8 / 10.3b | câu hỏi → câu trả lời, so đáp án | 15 | `qa_em_f1.csv`, `lexical_metrics.csv` |
| | BLEU | 1 / 10.3b | câu dịch, so câu tham chiếu | 1 | `lexical_metrics.csv` |
| | ROUGE-1/2/L | 1 / 10.3b | bản tóm tắt, so bản mẫu | 1 | `lexical_metrics.csv` |
| Ngữ nghĩa (LLM giám khảo) | Answer Relevancy, Faithfulness, Hallucination (DeepEval) | 8 / 10.3c | câu trả lời so câu hỏi / 3 đoạn tìm được / đoạn gốc | 18 (Faithfulness 17) | `deepeval_*.csv` |
| | G-Eval (5 tiêu chí) | 1, 5, 6, 8 / 10.3d | output so prompt / đáp án | 63 output (4 + 8 + 15 + 36) | `geval_*.csv` |
| Sinh mã | Unit-test pass rate, Pass@k | 1 / 1.6 | prompt → hàm `is_even`, chạy 7 test | 1 bài, 10 mẫu | `code_passk_*.csv` |
| *Đa dạng (thêm)* | distinct-1/2, tỉ lệ lặp | 6 / 6.5 | 3 output mỗi cấu hình | 15 output | `decoding_diversity.csv` |
| Hiệu quả thực thi | TTFT, Latency, Throughput, Memory, Cost/request | 6 / 6.6; 9 / 9.5 | 3 prompt, trung vị 5 lần | 3 prompt | `efficiency.csv`, `context_cost.csv` |

Ba nguyên tắc chung cho mọi metric trong bài:
1. **Greedy** khi chấm → chạy lại ra y hệt, điểm không dao động vì lấy mẫu.
2. **Luật chấm và câu tham chiếu viết trước khi chạy** → không chấm theo cảm tính sau khi thấy output (`RULES_v3.md` §2.4).
3. **Ghi cỡ mẫu n cạnh mọi số**; n = 1 thì chỉ là minh họa công thức.

---

## 1. Dự đoán ngôn ngữ

### 1.1 Cross-entropy (CE)

**Là gì.** Mức "ngạc nhiên" trung bình của model trước token thật. Đây chính là loss dùng khi pretraining và SFT [Slide Ch2: softmax, cross-entropy].

**Công thức.** Với câu có token x₁…xₙ:

  𝓛 = −(1/N) Σₜ ln P(xₜ₊₁ | x₁…xₜ),  N = n − 1 vị trí có ngữ cảnh.

- P = 0.9 → −ln 0.9 = 0.105 (đoán chắc và đúng, loss nhỏ).
- P = 0.01 → −ln 0.01 = 4.61 (đoán sai hướng, loss lớn).
- **Mốc đoán bừa:** chia đều cho 151,936 token → 𝓛 = ln 151,936 = **11.93**.

**Vì sao là −ln P.** Cross-entropy giữa phân phối thật (one-hot ở token đúng) và phân phối model chỉ còn lại đúng một số hạng −ln P(token đúng). Tối thiểu CE tương đương tối đa likelihood của dữ liệu.

**Cài đặt (cell 4.2).** Chạy model một lần trên cả câu; `log_softmax` logit ở vị trí 0…n−2 (bỏ vị trí cuối vì không có token để đoán), `gather` đúng token ở vị trí 1…n−1 (bỏ token đầu vì không có ngữ cảnh). Tính ở float32. Kiểm: khớp `outputs.loss` của transformers, chênh 6e-08 trên 3/3 câu.

**Kết quả.** Câu đúng 2.985 · câu xáo trộn 6.701 · câu Anh 3.129 · đoán bừa 11.93.

**Giới hạn.** Phụ thuộc tokenizer: hai model khác tokenizer thì CE trên mỗi token không so trực tiếp được (khác số token cho cùng câu). Câu ngắn (7 vị trí) nên chỉ minh họa, không phải đánh giá model.

### 1.2 Perplexity (PPL)

**Công thức.** PPL = e^𝓛. **Ý nghĩa:** model đang phân vân như thể phải chọn đều trong khoảng PPL token. Đoán bừa → PPL = 151,936.

**Kiểm tay.** e^2.985 = 19.8 · e^6.701 = 813.6 ✓ (`loss_summary.csv`).

**Nguồn.** HF *Perplexity of fixed-length models* [references, mục HF docs].

### 1.3 Top-1 / top-5 accuracy của token tiếp theo (thêm, cell 4.6)

- **Top-1:** % vị trí mà token có xác suất cao nhất trùng token thật.
- **Top-5:** % vị trí token thật nằm trong 5 token cao nhất. Luôn top-1 ≤ top-5 (assert).
- **Kết quả:** câu đúng 0.43 (3/7) / 0.71 (5/7); xáo trộn 0.12 (1/8) / 0.25 (2/8); câu Anh 0.33 / 0.67.
- **Vì sao thêm:** CE trộn "đúng hay sai" với "chắc đến đâu"; top-k accuracy tách riêng phần "có đoán trúng không", dễ giải thích hơn cho người nghe.

---

## 2. Độ đúng, trùng khớp

### 2.1 Accuracy và khoảng tin cậy 95%

**Accuracy** = số câu đúng / tổng câu. Mục 7: zero-shot 2,695/2,999 = **89.86%**, few-shot 2,768/2,999 = **92.30%**.

**CI 95% (Wald)** [Bổ sung: xấp xỉ chuẩn của phân phối nhị thức]:

  p̂ ± 1.96 · √(p̂(1 − p̂)/n)

Kiểm tay zero-shot: √(0.8986 × 0.1014 / 2,999) = 0.00551 → × 1.96 = **±1.08 điểm %** ✓.

**Giới hạn của CI riêng lẻ.** Hai CI [88.8, 90.9] và [91.4, 93.3] không chồng nhau, nhưng so hai CI độc lập là **sai cách** khi hai cách chấm trên **cùng câu** → cần kiểm định ghép cặp (McNemar, §2.3).

### 2.2 Confusion matrix

Bảng 2 × 2: hàng = nhãn thật, cột = nhãn model đoán. Từ đó đọc **lỗi theo hướng**:

| | Zero-shot | Few-shot |
|---|---|---|
| Tiêu cực → đoán tích cực | 248 | 145 |
| Tích cực → đoán tiêu cực | 56 | 86 |
| Tổng sai | 304 | 231 |

Accuracy chỉ cho biết tổng sai giảm; confusion matrix cho biết few-shot **đổi kiểu lỗi** (nghiêng về "tiêu cực").

### 2.3 Kiểm định McNemar exact (thêm, cell 10.2)

**Khi nào dùng.** So hai phương pháp phân loại trên **cùng một tập test**, kết quả mỗi câu là đúng/sai → dữ liệu **nhị phân, ghép cặp** [#22 Dietterich 1998].

**Bảng theo cặp:**

| | Few-shot đúng | Few-shot sai |
|---|---|---|
| Zero-shot đúng | a = 2,648 | b = 47 |
| Zero-shot sai | c = 120 | d = 184 |

**Lập luận.** Ô a, d (hai cách cùng kết quả) không nói gì về cách nào tốt hơn. Chỉ b + c = 167 câu bất đồng mang thông tin. Giả thuyết H₀ "hai cách như nhau" ⇒ mỗi câu bất đồng nghiêng về bên nào cũng 50/50 ⇒ b ~ Bin(167, 0.5), kỳ vọng b ≈ c ≈ 83.5.

**p-value (exact, hai phía):** p = 2 · P(X ≤ 47), X ~ Bin(167, 0.5) = **1.5 × 10⁻⁸** < 0.05 → bác bỏ H₀: few-shot tốt hơn có ý nghĩa thống kê.

**Cài đặt.** `statsmodels.stats.contingency_tables.mcnemar(table, exact=True)` [references: statsmodels mcnemar]. Tự kiểm: b = c → p = 1.

**Vì sao không dùng t-test hay so 2 CI.** Kết quả không liên tục và hai mẫu không độc lập; McNemar khai thác đúng cấu trúc ghép cặp nên mạnh hơn.

**Giới hạn.** Chỉ nói "khác biệt này khó do may" **trên bộ dữ liệu, model và bộ ví dụ này**; không nói few-shot luôn tốt hơn (đổi ví dụ/thứ tự có thể khác — Zhao et al. 2021).

### 2.4 Exact Match (EM)

**Là gì.** 1 nếu output sau chuẩn hóa **bằng đúng** một đáp án, ngược lại 0; lấy trung bình trên các câu [#26 SQuAD].

**Chuẩn hóa trong notebook (`norm_answer`).** Chữ thường; dấu câu thành khoảng trắng (không xoá, để "Charles-de-Gaulle" khớp "Charles de Gaulle"); gộp khoảng trắng. Khác script SQuAD: **không bỏ mạo từ a/an/the** vì dữ liệu tiếng Việt.

**Ví dụ thật.** "20.4%" = đáp án → 1. "Hoàng đế phải thông qua Hội đồng để ban hành luật." so với "phải thông qua Hội đồng" → **0**, dù đúng ý.

**Kết quả.** 0 → 0.2 (không RAG → RAG), n = 15 câu có đáp án.

**Giới hạn.** Quá chặt với model sinh câu đầy đủ; phù hợp khi output là span ngắn (như SQuAD gốc). Vì vậy kết quả chính của mục 8 là **chấm 3 mức** (§2.8), EM chỉ bổ sung.

### 2.5 Token F1

**Là gì.** Độ trùng **túi từ** giữa output và đáp án [#26].

  P = số từ trùng / số từ output · R = số từ trùng / số từ đáp án · F1 = 2PR / (P + R)

"Số từ trùng" đếm theo multiset (`Counter(p) & Counter(g)`); nhiều đáp án thì lấy **max**.

**Ví dụ tay (slide 22).** Output 11 từ chứa đủ 5 từ "phải thông qua hội đồng" → P = 5/11, R = 5/5 = 1 → F1 = 2 · (5/11) · 1 / (5/11 + 1) = 10/16 = **0.625**. Tự kiểm trong cell: "thủ đô là Tskhinvali" so "Tskhinvali" → P = 1/4, R = 1 → F1 = 0.4.

**Kết quả.** 0.007 → 0.395, n = 15.

**Giới hạn.** Không quan tâm thứ tự từ, không hiểu nghĩa: "Không phải Hội đồng" vẫn trùng nhiều từ. Đáp án ngắn + output dài → P thấp, kéo F1 xuống dù đúng.

### 2.6 BLEU

**Là gì.** Độ trùng n-gram (n = 1…4) của bản dịch với câu tham chiếu, thang 0–100 [#36 Papineni 2002]. Notebook dùng `sacrebleu.sentence_bleu` [#38 Post 2018] để tách từ và tính theo chuẩn.

**Công thức.**

  BLEU = BP · exp( ¼ Σₙ₌₁⁴ ln pₙ ),  pₙ = số n-gram trùng (cắt theo số lần xuất hiện trong tham chiếu) / số n-gram của output

  BP = 1 nếu output dài ≥ tham chiếu, ngược lại e^(1 − r/c) (phạt câu ngắn).

**Ví dụ tay (slide 4, 22).** Output "Today the weather is very beautiful ." (7 token sau khi tách dấu chấm), tham chiếu "The weather is very nice today ." (7 token). BLEU **phân biệt hoa thường**: "Today" ≠ "today", "the" ≠ "The".
- p₁ = 4/7 (weather, is, very, .) · p₂ = 2/6 (weather is, is very) · p₃ = 1/5 (weather is very) · p₄ = 0/4.
- p₄ = 0 làm ln không xác định → `sentence_bleu` mặc định **làm trơn kiểu "exp"**: n-gram đầu tiên có 0 khớp lấy 1/(2 × 4) = 1/8.
- BP = 1 (cùng độ dài).
- BLEU = (4/7 · 2/6 · 1/5 · 1/8)^(1/4) = (0.00476)^(0.25) = 0.263 → **26.3** ✓.

**Giới hạn.** Chỉ đếm chữ trùng: "beautiful" và "nice" cùng nghĩa vẫn không được điểm. BLEU thiết kế cho **cấp corpus** (nhiều câu); một câu như bài này chỉ minh họa (n = 1).

### 2.7 ROUGE-1 / ROUGE-2 / ROUGE-L

**Là gì.** Bộ metric cho tóm tắt, thiên về **recall**: bản tóm tắt mẫu có bao nhiêu phần xuất hiện trong output [#37 Lin 2004].
- ROUGE-1: trùng từ đơn; ROUGE-2: trùng cặp từ liền nhau.
- ROUGE-L: dựa trên **chuỗi con chung dài nhất (LCS)** — các từ trùng phải đúng thứ tự nhưng không cần liền nhau.
- Notebook báo **F1** của mỗi loại (cân bằng P và R).

**Ví dụ tay (slide 22).** Output 90 từ, tham chiếu 41 từ; ROUGE-1 P = 0.40, R = 0.88 → F1 = 2 · 0.40 · 0.88 / 1.28 = **0.55** ✓. Đọc: R cao = giữ gần đủ ý của bản mẫu; P thấp = output dài, rút gọn ít.

**Chi tiết cài đặt quan trọng.** Tokenizer mặc định của `rouge-score` chỉ giữ chữ a–z, 0–9 nên **xoá chữ có dấu tiếng Việt**. Notebook thay bằng `WsTokenizer` (tách theo khoảng trắng sau `norm_answer`), có assert `"Học sâu, GPU!" → ["học", "sâu", "gpu"]`. Không sửa chỗ này thì ROUGE tiếng Việt ra số sai mà không báo lỗi.

**Kết quả.** 0.55 / 0.34 / 0.50, n = 1 bản (tham chiếu `REF_SUMMARY` nhóm tự viết, duyệt trước normal run).

**Giới hạn.** Như BLEU: không hiểu nghĩa, phụ thuộc mạnh vào một bản tham chiếu.

### 2.8 Chấm 3 mức Đúng / Từ chối / Sai và recall@3 (thêm, mục 8)

- **Đúng** = output chứa một đáp án (sau chuẩn hóa); **Từ chối** = chứa "Không có thông tin"; còn lại **Sai**. Với câu kho **không có** đáp án: Từ chối là hành vi đúng, trả lời bất kỳ là bịa.
- Vì sao 3 mức: accuracy 2 mức gộp "biết từ chối" với "bịa" → không đo được hallucination.
- Kết quả: không RAG 0 / 14 / 1, có RAG **10 / 2 / 3** (15 câu); câu không có đáp án: RAG bịa 2/3. Toàn bộ đã đọc lại; 2 nhãn sửa tay có lý do.
- **recall@3** = % câu mà đoạn chứa đáp án nằm trong top-3 TF-IDF = 13/18 = 0.722. Đo riêng **bước tìm kiếm**, tách khỏi bước sinh.

---

## 3. LLM làm giám khảo (LLM-as-a-judge)

### 3.1 Ý tưởng chung

Metric so khớp chữ không hiểu nghĩa. Cách thay thế: cho **một LLM khác** đọc input, output (và đáp án / tài liệu) rồi chấm theo hướng dẫn [#31 Zheng 2023]. Ưu: hiểu nghĩa, chấm được tiêu chí mềm (mạch lạc, làm theo chỉ dẫn). Nhược đã biết [#31]: giám khảo có thể sai; **thiên vị model cùng họ hoặc chính mình (self-preference)**, thiên vị câu dài (verbosity), thiên vị vị trí khi so cặp.

**Thiết lập trong bài (cell 10.3c).**
- Giám khảo `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4 (khoảng 5.5 GB), chạy local trên GPU thứ hai; **lớn hơn và khác** model được chấm (1.5B) [#31: giám khảo nên mạnh hơn].
- Giám khảo sinh **greedy** → chạy lại ra cùng điểm.
- Lớp `LocalJudge` bọc model cho DeepEval: lấy đoạn JSON từ `{` đầu tới `}` cuối rồi kiểm schema; JSON hỏng thì **ghi lỗi, không bỏ im lặng**; assert lỗi ≤ 10%.
- `JUDGE_MAX_NEW_TOKENS = 1024` vì smoke run 24/11 với 512 làm JSON bị cắt.

**Giới hạn còn lại:** giám khảo cùng họ Qwen với model được chấm → có thể thiên vị (ghi trên slide 21).

### 3.2 DeepEval: Answer Relevancy

[Bổ sung, theo docs DeepEval; `[cần kiểm]` mở lại trang docs *Answer Relevancy* và thêm vào `references.md`.]
- Giám khảo tách output thành các **câu khẳng định (statements)**, rồi xét từng câu có **liên quan tới câu hỏi** không.
- Điểm = số câu liên quan / tổng số câu. Cao = tốt, đạt khi ≥ 0.5.
- **Không đo đúng/sai:** "…hoàng đế phải **ký tên**…" (Sai) vẫn được 1.0 vì bám đúng câu hỏi.
- Kết quả: 0.056 → 0.753 (18 câu). Không RAG thấp vì model gần như luôn trả "Không có thông tin".

### 3.3 DeepEval: Faithfulness

[Bổ sung, theo docs DeepEval; `[cần kiểm]` như trên.]
- Giám khảo rút các **sự thật (truths)** từ 3 đoạn RAG tìm được (`truths_extraction_limit=10` để JSON không quá dài), rút các **khẳng định (claims)** từ output, rồi xét từng claim có **mâu thuẫn** truths không.
- Điểm = số claim không mâu thuẫn / tổng claim. Đo "output có bám tài liệu được đưa vào không" — **không** đo tài liệu có đúng không.
- Chỉ áp cho chế độ RAG. Kết quả **0.627**, n = 17 (câu `0019-0036-0002` giám khảo trả JSON lỗi). Ví dụ bắt đúng: "Nam Ossetia thuộc chủ quyền của Nga" trái tài liệu (Gruzia) → 0.0.

### 3.4 DeepEval: Hallucination

- Test case nhận **đoạn gốc chứa đáp án** (`context`), không phải 3 đoạn tìm được.
- Theo docs DeepEval [references: DeepEval docs *Hallucination*]: bản 4.x tính **tỉ lệ đoạn context mà output khớp, không mâu thuẫn**, "higher is better". Bản cũ tính ngược (tỉ lệ mâu thuẫn, thấp = tốt) → notebook **ghim `deepeval==4.3.5`**. Smoke run 24/11 phát hiện chiều ngược này.
- Kết quả: 0.000 → 0.556 (18 câu). Giám khảo sai rõ ở câu "20.4%" (Đúng) → 0.0.
- **Đừng nhầm** với hallucination ở mục 9 (đo bằng chấm tay theo nguồn); đây là ước lượng tự động.

### 3.5 G-Eval

**Ý tưởng gốc [#39 Liu et al. 2023].** Giám khảo nhận tiêu chí → tự sinh **các bước chấm (chain-of-thought)** → điền điểm theo form. Bài báo lấy **trung bình có trọng số theo xác suất** các mức điểm (Σ p(s)·s) để điểm mịn hơn, tương quan với người chấm cao hơn metric chữ.

**Cách nhóm làm (cell 10.3d).**
- **Viết sẵn các bước chấm** cho 5 tiêu chí (`GEVAL_SPECS`) thay vì để giám khảo tự sinh → ít lần gọi, chạy lại ra như nhau, người đọc kiểm được rubric.
- Giám khảo cho 0–10, DeepEval chia 10 → 0–1, ≥ 0.5 là đạt. Model local không trả log-prob qua lớp bọc này nên **không có bước trung bình theo xác suất** như bài báo `[cần kiểm]` trong mã DeepEval 4.3.5.

| Tiêu chí | Giám khảo kiểm gì | Trường dữ liệu | Mục |
|---|---|---|---|
| Correctness | Sự kiện trong output có mâu thuẫn đáp án không | input, output, đáp án | 1, 8 |
| Semantic equivalence | Cùng nghĩa với đáp án, bỏ qua cách viết | output, đáp án | 8 |
| Completeness | Đủ các ý chính của đáp án chưa | output, đáp án | 8 |
| Coherence | Mạch lạc, đúng chủ đề, không lặp | input, output | 1, 5, 6 |
| Instruction following | Liệt kê từng yêu cầu trong prompt, output thỏa bao nhiêu; viết tiếp thay vì trả lời = không làm theo | input, output | 1, 5, 6 |

Câu mục 8 không có đáp án: đáp án tham chiếu là "câu trả lời đúng là từ chối".

**Kết quả chính.** Mục 5: Instruction following base 0.00 / Instruct 0.82, Coherence 0.05 / 0.95 · mục 6: Coherence greedy 0.70, T = 1.5 0.00 · mục 8: Correctness 0.22 → 0.38. 324 lần gọi giám khảo (cộng với 10.3c), 13.7 phút, 1 câu JSON lỗi.

### 3.6 Giám khảo có đáng tin không: đối chiếu với nhãn tay

So **36 output mục 8** (18 câu × 2 chế độ), "giám khảo nói đạt" (điểm ≥ 0.5) với "nhãn tay là đúng":

| Metric | Định nghĩa "nhãn tay đúng" | Khớp |
|---|---|---|
| DeepEval Hallucination | nhãn = Đúng | **32/36 (89%)** |
| G-Eval Correctness | nhãn = Đúng, **hoặc** câu kho không có đáp án mà model Từ chối (đúng hành vi) | **31/36 (86%)** — số trên slide 21 |
| G-Eval Correctness | chỉ nhãn = Đúng | 27/36 (75%) |

Hai dòng G-Eval khác nhau vì với câu không có đáp án, đáp án tham chiếu của G-Eval là "từ chối"; nên định nghĩa thứ hai (31/36) **nhất quán với cách G-Eval được chấm**. Khi bị hỏi, nói rõ định nghĩa này. 5 ca lệch của G-Eval: 3 câu RAG nhãn Đúng bị chấm 0.0 (`0004-0049-0002`, `0012-0034-0001`, `0013-0006-0001`) và 2 câu không RAG nhãn Từ chối (câu có đáp án) được chấm 0.8–1.0 (`0019-0036-0002`, `0001-0051-0006`).

---

## 4. Sinh mã

### 4.1 Unit-test pass rate

- = số test qua / tổng test. 7 test **ghi trước khi chạy**: 4 input của prompt + số âm chẵn (−4), số lẻ (7), số rất lớn (10⁹).
- Cách chạy: tách khối ```python``` bằng regex → `exec` → gọi `is_even` trên từng test; lỗi cú pháp/runtime tính 0.
- **Lưu ý an toàn:** `exec` không sandbox; chấp nhận được với hàm 1 dòng do model 1.5B sinh, code lạ phải chạy trong subprocess có timeout (ghi trong comment của cell 1.6).
- Kết quả: greedy 7/7; trung bình 10 mẫu 1.0.

### 4.2 Pass@k

**Là gì.** Xác suất **ít nhất 1 trong k lời giải** sinh ra là đúng (qua cả 7 test) [#19 Chen et al. 2021, HumanEval].

**Vì sao không lấy đúng k mẫu rồi đếm.** Phương sai lớn. Chen et al. sinh **n ≥ k** mẫu, đếm **c** mẫu đúng, rồi dùng **ước lượng không chệch**:

  Pass@k = 1 − C(n − c, k) / C(n, k)

Diễn giải: C(n − c, k)/C(n, k) là xác suất bốc k mẫu (không hoàn lại) từ n mà **trúng toàn mẫu sai**; lấy 1 trừ đi.

**Ví dụ tay (assert trong cell 1.6).** n = 10, c = 3:
- Pass@1 = 1 − C(7,1)/C(10,1) = 1 − 7/10 = 0.3 (= c/n).
- Pass@5 = 1 − C(7,5)/C(10,5) = 1 − 21/252 = 0.917.
- Nếu n − c < k (ít mẫu sai hơn k) thì chắc chắn có mẫu đúng → Pass@k = 1.

**Cấu hình lấy mẫu.** T = 0.8, top-p 0.95, seed 42. Chen et al. dùng top-p 0.95 và T cao (0.8) khi k lớn, T thấp (0.2) cho Pass@1 [#19]; nhóm dùng một cấu hình cho cả k = 1 và k = 5.

**Kết quả.** c = 10/10 → Pass@1 = Pass@5 = **1.0**. **Bài quá dễ nên bão hòa**: chỉ minh họa công thức; đo năng lực code thật cần bộ nhiều bài như HumanEval (164 bài).

---

## 5. Độ đa dạng khi sinh (thêm, cell 6.5)

- **distinct-n** [#40 Li et al. 2016] = số n-gram **khác nhau** / tổng số n-gram, gộp 3 output của một cấu hình. Cao = đa dạng. Assert: "a b a b" → distinct-1 = 2/4, distinct-2 = 2/3.
- **Tỉ lệ lặp** = 1 − (bigram khác nhau / tổng bigram) trong **một** output; 0 = không lặp vòng.
- Kết quả distinct-2: greedy 0.33 → T = 0.5 0.65 → top-p 0.5 0.76 → top-p 0.95 0.84 → T = 1.5 0.99.
- **Giới hạn:** đa dạng cao không có nghĩa là tốt — T = 1.5 đa dạng nhất nhưng vô nghĩa (G-Eval Coherence 0.00). Phải đọc cùng metric chất lượng.

---

## 6. Hiệu quả thực thi (cell 6.6, 9.5)

### 6.1 Hai pha khi sinh: prefill và decode

- **Prefill:** đọc cả prompt trong **một lượt forward**, mọi vị trí tính song song (nhờ causal mask), tạo KV cache và token đầu tiên.
- **Decode:** mỗi token mới cần **một lượt forward**, chỉ tính cho 1 vị trí, đọc lại KV cache của các token trước. Tuần tự, không song song được theo thời gian.

### 6.2 Định nghĩa và cách đo

| Metric | Định nghĩa trong notebook | Kết quả (3 prompt) |
|---|---|---|
| TTFT | thời gian `generate(max_new_tokens=1)` ≈ prefill + 1 token | 0.045 / 0.048 / 0.046 s |
| Latency | thời gian sinh xong cả câu | 2.31 / 3.46 / 8.44 s (60 / 90 / 219 token ra) |
| Throughput | token sinh ÷ latency | ≈ 26.0 token/s |
| Memory | VRAM đỉnh `max_memory_allocated` (GiB); RAM tiến trình (RSS) | 2.90 GiB; RAM 2.35 GiB |
| Cost / request | GPU-giây = latency × 1 GPU (× giá thuê nếu có) | 2.3 / 3.5 / 8.4 GPU-giây |

**Đo đúng cách:** chạy nóng 1 lần (lần đầu chậm vì khởi tạo kernel) · đo 5 lần lấy **trung vị** · `torch.cuda.synchronize()` trước khi bấm giờ vì GPU chạy **bất đồng bộ** với Python · greedy, batch 1, cùng `max_new_tokens`.

**Đọc kết quả.**
- TTFT gần như không đổi dù prompt 42 hay 165 token: prefill song song, GPU còn dư sức ở độ dài này.
- Latency tỉ lệ với **số token ra**, không phải số token vào → throughput ổn định 26 token/s.
- `GPU_USD_PER_HOUR = None`: Kaggle miễn phí nên báo bằng GPU-giây; muốn quy ra tiền thì nhân giá thuê T4 có nguồn.

### 6.3 Bộ nhớ: tính tay từ cấu hình model

Đơn vị: 1 GiB = 1024³ byte; fp16 = 2 byte. Cấu hình (`model_config.json`): 28 layer, 12 head Q, 2 head K/V, head_dim 128, hidden 1,536, FFN 8,960, vocab 151,936.

**(a) Trọng số.** 1,543,714,304 × 2 = 3,087,428,608 byte = **2.88 GiB** (đo 2.89) — "3.1 GB" trên slide 2 là đơn vị thập phân (10⁹), cùng một lượng.

**(b) KV cache.** Mỗi token lưu K và V ở mọi layer:

  2 (K, V) × 28 × 2 head K/V × 128 × 2 byte = **28,672 byte/token**

- 300 token → 8.6 MB: vì vậy VRAM đỉnh khi sinh (2.90) chỉ hơn trọng số 0.01 GiB.
- 8,192 token → 0.22 GiB · 32,768 token → 0.88 GiB.
- **GQA giúp gì:** nếu mỗi head Q có K/V riêng (12 head) thì KV cache gấp 6 lần.

**(c) Ma trận attention lúc prefill (tăng theo n²).** Số đo `context_cost.csv` trừ trọng số 2.89 GiB:

| n token | Phần thêm đo được (GiB) | Tăng so với độ dài trước | Dự đoán 7.30·(n/8,192)² + KV |
|---|---|---|---|
| 512 | 0.06 | | 0.04 |
| 1,024 | 0.17 | ×2.8 | 0.14 |
| 2,048 | 0.56 | ×3.3 | 0.51 |
| 4,096 | 2.01 | ×3.6 | 1.94 |
| 8,192 | 7.52 | ×3.7 | 7.52 |

- Gấp đôi độ dài → gần ×4 bộ nhớ: dấu hiệu **bậc hai**; KV cache (bậc một) chỉ cho ×2.
- Ma trận điểm attention một layer: 12 × 8,192² = 805,306,368 phần tử → 1.5 GiB ở fp16, 3.0 GiB ở fp32. Phần còn lại 7.30 GiB ≈ **9.7 byte mỗi phần tử**, khớp giả thuyết "điểm fp16 + bản fp32 cho softmax + kết quả fp32" ≈ 10 byte.
- **Loại trừ "logits toàn chuỗi":** 8,192 × 151,936 × 4 byte = 4.64 GiB, bậc một → ở 4,096 phải ≥ 2.32 GiB, nhưng đo chỉ 2.01 → không khớp.
- **Kết luận:** 10.41 GiB ≈ 2.9 trọng số + 0.2 KV cache + 7.3 ma trận attention n × n.
- `[cần kiểm]` Vì sao SDPA tạo cả ma trận n × n: giả thuyết là T4 (sm75) không có FlashAttention nên PyTorch chọn kernel "math". Đây là **suy luận từ đường cong n²**, chưa đo trực tiếp; kiểm bằng cách chạy cell 9.5 trong `torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`.

**(d) Bài học từ lỗi đo 25/11.** Run `082211` báo VRAM đỉnh 6.2 GB vì biến `inputs` của cell 5.3 còn giữ model base trên GPU. Đo bộ nhớ phải kiểm "trước khi đo còn gì trên GPU" (dòng "VRAM sau khi xoá base" giờ in 2.88 GiB).

---

## 7. Tóm tắt giới hạn của cả bộ metric

| Giới hạn | Ảnh hưởng | Nhóm xử lý thế nào |
|---|---|---|
| BLEU, ROUGE, Pass@k chỉ n = 1 bài | Không phải số đánh giá | Ghi n cạnh số; nói rõ "chỉ minh họa công thức" |
| EM, F1, BLEU, ROUGE chỉ đếm chữ | Đúng ý vẫn điểm thấp | Kết quả chính dùng chấm 3 mức có đọc tay |
| Giám khảo LLM có thể sai, cùng họ Qwen | Điểm DeepEval/G-Eval lệch | Đối chiếu nhãn tay (31–32/36); ghi ca sai lên slide |
| 1 model, 1 bộ dữ liệu, 1 bộ ví dụ | Không suy rộng | Ghi giới hạn ở slide 20 |
| Hiệu quả đo trên T4, batch 1 | Không đại diện máy khác | Ghi rõ điều kiện đo |

## 8. Câu hỏi Q&A về metric

1. **Perplexity 19.8 nghĩa là gì?** Trung bình model phân vân như chọn đều giữa khoảng 20 token; đoán bừa là 151,936.
2. **Vì sao EM = 0.2 mà nói RAG tốt lên?** EM chỉ cho điểm khi khớp y đáp án; câu đúng ý nhưng đầy đủ hơn bị 0. Chấm 3 mức: 0 → 10/15.
3. **BLEU 26.3 có phải bản dịch kém?** Không: đúng nghĩa; BLEU thấp vì "beautiful" ≠ "nice" và phân biệt hoa thường. Đó là giới hạn của BLEU.
4. **Pass@k khác pass rate thế nào?** Pass rate = % test qua của một lời giải; Pass@k = xác suất có ít nhất 1 lời giải đúng hoàn toàn trong k lần thử.
5. **Answer Relevancy cao có nghĩa là đúng?** Không, chỉ là bám câu hỏi ("ký tên" sai vẫn 1.0).
6. **Faithfulness khác Hallucination?** Faithfulness so output với 3 đoạn **tìm được**; Hallucination (cách nhóm dùng) so với **đoạn gốc chứa đáp án**.
7. **G-Eval khác DeepEval metric khác?** G-Eval là khung chấm theo rubric tự định nghĩa (nhóm viết bước chấm); ba metric kia có thuật toán cố định.
8. **Sao không dùng t-test?** Kết quả đúng/sai, ghép cặp trên cùng câu → McNemar.
9. **TTFT khác latency?** TTFT = tới token đầu (prefill); latency = tới token cuối.
10. **Vì sao ngữ cảnh 8,192 token tốn 10.4 GiB?** §6.3: phần lớn là ma trận attention n × n lúc prefill, không phải KV cache.

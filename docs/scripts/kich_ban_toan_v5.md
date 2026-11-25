# Kịch bản thuyết trình chính thức — 25 slide, 10 mục (v5)

- **Created**: 2026-11-25 (Toàn, AI hỗ trợ). Thay `kich_ban_toan_v4.md` (v4 viết cho bộ 18 slide cũ, giữ nguyên làm lịch sử).
- **Khác v4:** theo bộ 25 slide hiện tại (`slides/infographic/llm_slides.html`), mỗi slide một đoạn lời nói liền mạch; thêm slide 3 (so sánh LM, LLM, chatbot), slide 21–23 (18 metric, so khớp chữ + DeepEval, G-Eval + hiệu quả thực thi); số liệu theo lần chạy chính thức **`20261125_214650_normal`** (Kaggle T4 x2, 154 cell, 0 lỗi, 41.2 phút).
- Lý thuyết chi tiết của bộ metric (để trả lời Q&A): `docs/research/noi_dung_bo_metric_danh_gia_v1.md`.

**Ký hiệu**

| Ký hiệu | Nghĩa |
|---|---|
| "Chữ trong ngoặc kép" | Lời nói, nói thành tiếng |
| [ ... ] | Thao tác trên màn hình, không đọc |
| ⏩ | Câu bỏ được khi thiếu giờ (không mất ý chính) |
| **In đậm** | Nhấn giọng |

**Thời gian.** Giới hạn **20 phút, gồm cả lúc mở notebook**. Bỏ hết câu ⏩: khoảng 18 phút nói + 1 phút notebook. Nói đủ cả ⏩: khoảng 22 phút. Khi tập, bấm giờ cả bài; quá 19 phút thì bỏ ⏩ từ slide 13 trở đi trước.

**Mở sẵn:** PDF slide (toàn màn hình) + notebook `notebook/llm_presentation.ipynb` bản đã chạy. Chỉ mở notebook 3 lần, mỗi lần khoảng 20 giây: slide 7 (cell 2.4), slide 12 (cell 5.3), slide 20 (cell 10.2).

---

## Slide 1 — Mở đầu (0.3 phút)

"Em chào cô và các bạn. Em là Toàn, đại diện Nhóm 1 trình bày đề tài **Tìm hiểu mô hình ngôn ngữ lớn — Large Language Models**.

Bài trả lời ba câu hỏi: LLM **học** và **sinh văn bản** như thế nào; LLM đi qua **những giai đoạn huấn luyện** nào; và nó **giỏi gì, hạn chế ở đâu**. Mỗi mục đều có **demo trên model thật** là Qwen2.5-1.5B-Instruct, với dữ liệu tiếng Việt UIT-VSFC và UIT-ViQuAD. Mọi số trên slide đọc từ file kết quả của **một lần chạy chính thức** trên Kaggle: 154 cell, không lỗi."

---

## Slide 2 — Model, môi trường, thư viện (0.6 phút)

"Model chính là **Qwen2.5-1.5B-Instruct**, kiến trúc **decoder-only Transformer**: 28 block, mỗi token là vector 1,536 chiều, từ điển 151,936 token, ngữ cảnh tối đa 32,768 token. Tổng **1,543,714,304 tham số** — em không chép từ web mà **đếm bằng code** rồi `assert`, như khung code bên dưới.

Hai model phụ: bản **base** để so ở mục 5, và tokenizer **GPT-2** để so ở mục 3.

Chạy trên GPU **T4** của Kaggle, model ở **fp16**, 2 byte một tham số, nên riêng trọng số khoảng **3.1 GB**. Nhóm **chỉ suy luận, không fine-tune**.

⏩ Bản Instruct mặc định **lấy mẫu ngẫu nhiên**, nên mọi lần gọi `generate` em đều truyền tham số tường minh, và khi đánh giá thì dùng **greedy** để chạy lại ra y hệt.

⏩ Bảng bên phải là thư viện và phiên bản, đọc bằng code và lưu vào `env.json`."

---

## Slide 3 — Mục 1 (1/2): Language model, LLM, chatbot (1.0 phút)

"Mục 1. Em đi từ khái niệm nhỏ nhất.

**Language model** là mô hình **gán xác suất cho chuỗi chữ**. Công thức đọc là: xác suất cả chuỗi **bằng tích** xác suất từng token khi biết các token đứng trước. Nghĩa là model chỉ cần làm đi làm lại **một việc: đoán token tiếp theo**.

**LLM** vẫn là language model, chỉ khác ở chữ **'lớn'**: Transformer, hàng tỉ tham số, học trên hàng nghìn tỉ token — Qwen2.5 khoảng 18 nghìn tỉ.

**Chatbot** là **ứng dụng** bọc ngoài LLM: LLM đã qua post-training, cộng **chat template**, giao diện, lớp an toàn. Khung đen là chat template thật: câu hỏi nằm giữa các thẻ `im_start`, `im_end`, cuối cùng là `assistant` để trống — với model, 'trả lời' chính là **viết tiếp sau chữ assistant**.

Sơ đồ ba vòng cho thấy đây là **ba khái niệm lồng nhau**, không phải ba thứ tách rời. ⏩ Bảng bên dưới so sánh theo input, output, cách huấn luyện và cách đánh giá; dòng cuối là các metric em sẽ dùng xuyên bài.

Khung 'Cùng 1 model, khác cách dùng': cùng câu 'Thủ đô của Pháp là gì', bản base **viết tiếp như trang báo**, bản Instruct qua chat template trả lời **'Paris'**. Bên dưới, mọi chatbot vẫn chỉ đoán token tiếp theo."

---

## Slide 4 — Mục 1 (2/2): Một LLM làm 4 tác vụ (1.0 phút)

"Cùng một model, sinh greedy, **chỉ đổi prompt**, em cho làm 4 tác vụ. Mỗi dòng đọc theo thứ tự: input, output thật, model làm thế nào, tiêu chí, kết quả.

**Hỏi đáp**: thủ đô Nhật Bản → 'Tokyo', đạt, G-Eval Correctness 1.0.

**Tóm tắt** đoạn 5 câu thành 1 câu: giữ đủ 5 ý. Đo bằng **ROUGE** — mức trùng từ, cặp từ và chuỗi con chung dài nhất với bản tóm tắt mẫu: 0.55, 0.34, 0.50.

**Dịch** 'Hôm nay trời rất đẹp': ra 'Today the weather is very beautiful' — **đúng nghĩa**, nhưng **BLEU chỉ 26.3** vì BLEU đếm chữ trùng với câu tham chiếu, mà câu tham chiếu dùng chữ 'nice'.

**Sinh mã** hàm `is_even`: em không đọc bằng mắt mà **chạy thật** 7 unit test — qua 7 trên 7. Thêm 10 lời giải lấy mẫu, cả 10 đều qua, nên **Pass@1 = Pass@5 = 1.0**.

⏩ Dải sơ đồ giữa slide là đường đi của phần sinh mã: prompt → generate → tách khối code bằng regex → chạy → unit test. Công thức Pass@k là của Chen và cộng sự 2021.

Lưu ý: đây là bài dễ nên điểm bão hòa, **chỉ để minh họa công thức**; đo thật cần bộ như HumanEval. Ý chính: bốn tác vụ khác nhau nhưng **đều là sinh token tiếp theo**."

---

## Slide 5 — Mục 2 (1/3): Transformer, encoder và decoder (0.9 phút)

"Mục 2: kiến trúc. **Transformer** xử lý chuỗi bằng **self-attention**: mỗi token 'nhìn' các token khác để lấy thông tin. Hình bên trái so với RNN: RNN đọc lần lượt, token đầu phải đi qua nhiều bước mới tới cuối nên dễ quên; Transformer **nối mọi cặp token trong một bước**, tính bằng nhân ma trận nên chạy **song song trên GPU** — nhờ vậy mới mở rộng được lên hàng tỉ tham số.

Bảng thành phần: embedding đổi token thành vector; **RoPE** cho biết vị trí; self-attention trộn thông tin; **FFN** xử lý riêng từng token; residual và norm giúp mạng sâu học ổn định; output head ra xác suất.

Transformer gốc có hai chồng: **encoder** đọc hiểu, **decoder** sinh. Ba ma trận mask nhỏ: ô xanh là được nhìn, hàng là token đang xét, cột là token được nhìn. Encoder nhìn **hai chiều**, decoder có **causal mask** chỉ nhìn quá khứ.

LLM chọn **decoder-only** vì một mục tiêu duy nhất — đoán token tiếp theo — dùng được cho mọi văn bản không nhãn, và cùng một cơ chế vừa để học vừa để sinh."

---

## Slide 6 — Mục 2 (2/3): Decoder-only và causal attention (1.1 phút)

"Bảng trái là đường đi của câu 'Hôm nay trời rất' — 6 token — qua Qwen. Đọc từng dòng: token ID kích thước 1 × 6; **embedding** tra hàng trong ma trận 151,936 × 1,536, ra 1 × 6 × 1,536. Rồi **28 decoder block** giống nhau, kích thước giữ nguyên; mỗi block có **causal attention** — 12 head Q nhưng chỉ 2 head K/V, gọi là GQA, để tiết kiệm bộ nhớ — và **FFN SwiGLU** chiếm **75% tham số**. Cuối cùng chỉ lấy **vector vị trí cuối**, nhân với ma trận embedding để ra 151,936 điểm, softmax ra xác suất: ' đẹp' 0.788.

Bên phải là công thức: **softmax của QKᵀ chia căn d, cộng M, nhân V**. Q là token đang tìm gì, K là token khác có gì, V là nội dung được lấy, **M bằng âm vô cùng ở ô tương lai**.

Ví dụ tính tay 3 token: ma trận sau softmax là 3 × 3, **hàng là token đang xét, cột là token được nhìn**, mỗi hàng cộng bằng 1. Hàng 'Hôm' chỉ nhìn chính nó: 1, 0, 0. Hàng 'trời' chia 0.25, 0.25, 0.50 cho ba token. Ô tương lai bằng 0 vì e mũ âm vô cùng bằng 0. Kiểm trên Qwen thật: mọi ô trên đường chéo đều **0.00**.

⏩ Mask này giúp khi học, mọi vị trí cùng đoán token kế tiếp **song song** mà không nhìn trộm đáp án."

---

## Slide 7 — Mục 2 (3/3): Từ token đến xác suất token tiếp theo (0.8 phút + notebook)

"Dải trên cùng tóm cả mục 2 thành 8 bước: văn bản → 6 token → 6 vector 1,536 số → 28 block cho các vector trao đổi qua causal attention → chỉ vector cuối được đổi thành **151,936 logit** → softmax → chọn một token → nối vào input và lặp lại.

Hai hình giữa: trái là causal mask 8 × 8 theo quy tắc, phải là attention **thật** của Qwen ở layer 0, head 0 — tam giác dưới, phía trên đường chéo đều bằng 0.

[Mở notebook, cell 2.4: chỉ bảng top-5.] Đây là output thật: sau 'Hôm nay trời rất', ' đẹp' **78.8%**, ' nắng' 6.3%, ' lạnh' 4.6%. Model **không chọn từ**, nó cho **mọi token một xác suất**. Greedy lấy ' đẹp'; lấy mẫu thì khoảng 6% lần ra ' nắng'. Mục 4 nói model **học** xác suất này thế nào, mục 6 nói **chọn** token ra sao."

---

## Slide 8 — Mục 3 (1/2): Tokenization, token ID, embedding (0.9 phút)

"Mục 3. Bảng trên so 4 cách chia câu 'Hôm nay trời rất đẹp.': theo từ 6 token nhưng từ điển phình to và gặp từ lạ thành UNK; theo ký tự 21; theo byte 29. LLM dùng **subword BPE** để cân bằng: Qwen **8 token**, GPT-2 **22 token**.

**BPE** học bảng gộp: bắt đầu từ 256 byte, đếm cặp liền nhau hay gặp nhất, gộp thành token mới, lặp tới khi đủ từ điển. Kho văn bản nhiều tiếng Việt thì ' trời' thành **1 token**; ít tiếng Việt như GPT-2 thì còn rời thành 5 token.

Bảng dưới là 5 bước của ' trời': đổi sang 7 byte UTF-8 → tách sơ bộ, dấu cách dính vào đầu từ → BPE gộp 7 byte thành 1 → tra từ điển ra ID **130868** → lấy **hàng thứ ID của ma trận embedding**, ra vector 1,536 số.

Điểm cần nhớ: **token ID chỉ là số thứ tự, không mang nghĩa; vector mới mang nghĩa**, và vector đó học trong pretraining. ⏩ Hình heatmap: 8 hàng là 8 token, 16 cột đầu của 1,536 chiều."

---

## Slide 9 — Mục 3 (2/2): Demo câu Việt và câu Anh (0.6 phút)

"Demo chạy thật. Câu Việt: Qwen **8 token**, decode ngược ra đúng câu gốc. GPT-2 **22 token**, có các ký tự '�' — mỗi cái là **một byte lẻ**, vì chữ 'ờ' 3 byte bị tách thành 3 token.

Câu Anh cùng nghĩa: **cả hai đều 7 token**. ⏩ Cùng chữ 'The' nhưng ID khác nhau — 785 với 464 — nên **tokenizer phải đi đúng cặp với model**.

Bảng đánh giá: tiếng Anh như nhau; tiếng Việt GPT-2 tốn **gấp 2.75 lần** token, fertility 4.4 token mỗi từ so với 1.6. Ít token hơn nghĩa là **rẻ hơn, nhanh hơn, tốn ít ngữ cảnh hơn** — quay lại ở mục 9."

---

## Slide 10 — Mục 4: Next-token prediction và cross-entropy (1.0 phút)

"Mục 4: LLM học thế nào. Nguyên lý: **chỉ học đoán token tiếp theo**. Nhãn chính là token kế tiếp, **có sẵn trong văn bản**, không cần người gán — nên học được trên nghìn tỉ token. Dải sơ đồ là một bước học: văn bản → token → cặp (ngữ cảnh, token tiếp) → model ra xác suất → loss → backprop sửa trọng số.

Hình trái: câu 8 token tạo **7 cặp** ngữ cảnh → mục tiêu, và nhờ causal mask cả 7 cặp tính **trong một lần chạy**.

Loss là **cross-entropy**: trung bình của **âm ln xác suất token đúng**. Đoán đúng với P 0.9 thì loss 0.1; chỉ cho P 0.01 thì loss 4.6. **Học là làm loss nhỏ lại.** Perplexity bằng e mũ loss, hiểu là model đang phân vân giữa khoảng bao nhiêu lựa chọn.

Hình phải là loss từng token trên model thật: 'Hôm n' → 'ay' gần như chắc chắn, loss 0.004; 'H' → 'ôm' loss 8.19 vì ngữ cảnh còn quá mở.

Bảng: câu đúng loss **2.985**, perplexity 19.8; câu **xáo trộn** 6.701, perplexity 813.6; mốc đoán bừa ln 151,936 = 11.93. ⏩ Top-1 accuracy — token xác suất cao nhất trùng token thật — 0.43 với câu đúng, 0.12 với câu xáo trộn. ⏩ Loss em tự tính khớp `outputs.loss` của transformers, chênh 6 phần trăm triệu. Notebook **chỉ tính loss, không huấn luyện**."

---

## Slide 11 — Mục 5 (1/2): Pretraining, SFT, alignment (1.0 phút)

"Mục 5. Câu mở đầu quan trọng: **model không biết mình đang ở giai đoạn nào**. Ở cả ba giai đoạn nó chỉ chỉnh trọng số để tăng xác suất một chuỗi token; thứ thay đổi là **dữ liệu** và **token nào được tính loss**.

Ba cột đọc cùng một khuôn: mục tiêu, dữ liệu, loss tính trên đâu, học được gì, đánh giá.

**Pretraining**: văn bản thô không nhãn, loss trên **mọi token** → ra **base model**, hỏi thì viết tiếp.

**SFT**: cặp chỉ dẫn – câu trả lời mẫu do người viết, bọc chat template; vẫn cross-entropy nhưng **chỉ tính trên phần trả lời** → học trả lời đúng định dạng, biết dừng.

**Preference alignment**: bộ ba prompt, câu tốt, câu kém; loss **so sánh xác suất** câu tốt với câu kém → hữu ích, an toàn, biết từ chối.

Hàng dưới là hai cách làm alignment. **RLHF**: người chọn câu tốt hơn → học **reward model** → tối ưu bằng PPO, có phạt **KL** để model không lệch quá xa bản SFT. **DPO** học thẳng từ cặp tốt/kém, **không cần reward model, không cần RL** — giống bài toán phân loại 'câu nào tốt hơn'. ⏩ Qwen2.5 dùng SFT, rồi DPO và GRPO."

---

## Slide 12 — Mục 5 (2/2): Demo base vs Instruct (0.8 phút + notebook)

"Nhóm **không tự fine-tune**; em thấy tác dụng của các giai đoạn bằng cách so **hai bản Qwen công bố**: base chỉ qua pretraining, Instruct qua cả ba. Hai bản **cùng kiến trúc, cùng 1.54 tỉ tham số, cùng prompt, cùng greedy** — khác biệt chỉ do post-training.

[Mở notebook, cell 5.3: chỉ output câu Pháp.] Câu 'Thủ đô của Pháp là gì': bản base viết tiếp như trang VnExpress; bản Instruct trả lời 'Paris'. Câu yêu cầu quay cóp: base viết tiếp bài báo, Instruct **từ chối**.

Luật 'đạt' ghi **trước** khi chạy. Kết quả: base **0/4**, Instruct **4/4**. Giám khảo G-Eval cũng xác nhận: Instruction following 0.00 so với 0.82.

Đọc kết quả: bản base **có kiến thức** — vẫn nhắc tới Paris — nhưng chỉ biết viết tiếp văn bản; **SFT và alignment** dạy nó trả lời đúng chỉ dẫn."

---

## Slide 13 — Mục 6 (1/2): Autoregressive, greedy, temperature, top-p (1.0 phút)

"Mục 6. Sinh văn bản là **autoregressive**: sinh một token, nối vào input, lặp lại tới khi gặp EOS hoặc đủ token. Dải sơ đồ có 6 bước; bước 3 và 4 là chỗ các cách sinh khác nhau.

Hàng token màu là 8 bước greedy thật với prompt Hà Nội, số nhỏ dưới mỗi token là xác suất lúc chọn. ⏩ Vòng lặp em tự viết ra **đúng 8/8 token** như `generate`, có assert.

Công thức **temperature**: chia logit cho T rồi softmax. **T nhỏ hơn 1** làm phân phối nhọn, gần greedy; **T lớn hơn 1** làm phẳng, đa dạng hơn nhưng dễ sai. **Top-p** giữ nhóm token nhỏ nhất có tổng xác suất ≥ p rồi mới bốc thăm.

Hình giữa: xác suất token đầu 'H' là 0.995 ở T = 0.5, 0.847 ở T = 1, còn 0.468 ở T = 1.5 — để phủ 90% xác suất cần **1, 3 rồi 135 token**.

Bảng so sánh ba cách: greedy chạy lại ra y hệt, dùng khi **đánh giá**; temperature cho viết sáng tạo; top-p tự co giãn, dùng cho chatbot. ⏩ Ví dụ: top-p 0.5 ở bước đầu chỉ giữ 'H' vì 0.847 ≥ 0.5, nên giống greedy."

---

## Slide 14 — Mục 6 (2/2): Cùng prompt, 5 cấu hình sinh (0.7 phút)

"Cùng prompt, 5 cấu hình, mỗi cấu hình 3 lần. Đọc bảng theo cột: output thật, số output khác nhau trong 3 lần, distinct-2, tỉ lệ lặp, điểm mạch lạc.

**Greedy**: 3 lần giống nhau, distinct-2 0.33, mạch lạc — nhưng **sai sự thật**: Đền Hùng ở Phú Thọ, không ở Hà Nội. **T = 0.5, top-p 0.5**: mỗi lần khác chút, vẫn tự nhiên. **Top-p 0.95**: có lần bịa 'Hà Nội ở Trung Quốc' và lẫn chữ Hán. **T = 1.5**: vô nghĩa, distinct-2 0.99, mạch lạc **0.00**.

**distinct-2** là số cặp từ khác nhau chia tổng số cặp từ: càng cao càng đa dạng.

Kết luận: **chỉ đổi cách chọn token** mà output khác hẳn; greedy ổn định nhưng không đảm bảo đúng. Vì vậy nhóm dùng **greedy cho mọi phần đánh giá**."

---

## Slide 15 — Mục 7: In-context learning, zero-shot vs few-shot (1.1 phút)

"Mục 7: khả năng của LLM. Ba ô trên: **in-context learning** — học cách làm từ ví dụ trong prompt, **không đổi tham số**; làm theo chỉ dẫn — nhờ SFT; và mọi tác vụ đưa về **sinh chữ**.

Thí nghiệm: phân loại cảm xúc phản hồi sinh viên UIT-VSFC. **Zero-shot** chỉ có chỉ dẫn và câu cần làm. **Few-shot** thêm **4 ví dụ lấy từ tập train**. Cùng câu 'thầy giảng nhanh', zero-shot ra 'tích cực', few-shot ra 'tiêu cực' đúng nhãn. Lý do: bộ dữ liệu gán **câu góp ý là tiêu cực**, và few-shot học quy ước này từ ví dụ.

Bảng phải, trên **2,999 câu test**: accuracy **89.86% lên 92.30%**, khoảng tin cậy 95% khoảng ±1 điểm. Hai dòng lỗi theo hướng đọc từ confusion matrix: tiêu cực bị đoán tích cực giảm **248 xuống 145**, nhưng tích cực bị đoán tiêu cực **tăng 56 lên 86**. ⏩ Tức là ví dụ làm model hơi nghiêng về 'tiêu cực'. Kiểm định McNemar cho p = 1.5 × 10⁻⁸ — em giải thích ở mục 10.

⏩ Khung đảm bảo: cùng câu test, cùng mẫu prompt chọn trên 200 câu validation, greedy, **chỉ một biến là có ví dụ hay không**."

---

## Slide 16 — Mục 8 (1/2): Prompting, fine-tuning, RAG (0.7 phút)

"Mục 8. LLM thiếu hai thứ: **thiếu kỹ năng** — sai định dạng, chưa quen tác vụ — thì dùng prompting hoặc fine-tuning; **thiếu kiến thức** — tài liệu nội bộ, tin mới — thì dùng **RAG**. Model chỉ 'biết' những gì nằm trong **trọng số** hoặc trong **prompt**.

Ba sơ đồ: prompting **không đổi model**; fine-tuning **đổi trọng số**, giống SFT ở mục 5 nhưng trên dữ liệu riêng; RAG gồm **Retrieval** tìm đoạn liên quan, **Augmented** dán vào prompt, **Generation** trả lời.

Bảng dưới so theo dữ liệu cần, chi phí, cách cập nhật kiến thức và cách đánh giá. Ý chính: muốn **cập nhật kiến thức**, RAG chỉ cần **sửa kho tài liệu**, fine-tuning phải huấn luyện lại. ⏩ Dòng cuối là chỗ mỗi cách xuất hiện trong bài: few-shot ở mục 7, base vs Instruct ở mục 5, RAG ở slide sau."

---

## Slide 17 — Mục 8 (2/2): Demo RAG trên UIT-ViQuAD 2.0 (1.0 phút)

"Demo RAG. Retrieval bằng **TF-IDF**: từ hiếm trùng với câu hỏi được trọng số cao, đo độ giống bằng cosine, lấy **top-3 trong 557 đoạn** Wikipedia.

Ví dụ: 'Muốn ban hành luật thì hoàng đế phải làm gì?'. Không RAG: 'phải **ký tên** vào bản thảo' — **sai**. Có RAG: 'phải **thông qua Hội đồng**' — **đúng** với đáp án.

Bảng kết quả, 15 câu có đáp án. Chấm 3 mức **Đúng / Từ chối / Sai** để tách câu **bịa** khỏi câu **biết từ chối**: không RAG 0 đúng, 14 từ chối; có RAG **10 đúng**. Nhưng với 3 câu mà kho **không có đáp án**, RAG làm model **bịa 2/3**. recall@3 là 13/18: 5 câu không tìm đúng đoạn.

Các metric tự động đi cùng chiều: **token F1 0.007 lên 0.395**, Answer Relevancy 0.056 lên 0.753, G-Eval Correctness 0.22 lên 0.38. ⏩ **EM chỉ 0.2** vì câu trả lời đúng ý nhưng dài hơn đáp án.

Nhận xét: RAG bổ sung kiến thức **lúc hỏi**, không đổi trọng số — nhưng **chỉ tốt bằng bước tìm kiếm**."

---

## Slide 18 — Mục 9 (1/2): Hallucination và cách kiểm chứng (0.9 phút)

"Mục 9: hạn chế. **Hallucination** là nội dung nghe hợp lý nhưng sai hoặc không có nguồn. Vì sao: model chọn token **xác suất cao theo ngữ cảnh**, không tra cứu sự thật; không biết thì vẫn viết tiếp cho trôi chảy.

Bảng 5 bước là một ví dụ đầy đủ: hỏi lại câu hoàng đế, **không tài liệu, không cho từ chối**. Cách hỏi A: model liệt kê các bước tự nghĩ ra. Hỏi lại cách B: ra đáp án khác. Đối chiếu Wikipedia: phải thông qua Hội đồng. Không khớp nguồn, hai lần hỏi hai đáp án → **hallucination**.

Trên 15 câu: sai **12/15** mỗi cách hỏi, sai cả hai cách 11/15.

Bảng phải: hỏi về **thực thể không tồn tại** — nhà vật lý người Việt đoạt Nobel, Việt Nam đăng cai Thế vận hội 1987 — model **bịa 5/6**, và **không lần nào chỉ ra tiền đề sai**.

Ba cách kiểm chứng: **đối chiếu nguồn**, **hỏi lại cách khác**, và **dùng RAG**. ⏩ Dữ liệu cũng có thể sai: đáp án '1640' của UIT-ViQuAD là nhãn sai, nhóm sửa tay và ghi lý do."

---

## Slide 19 — Mục 9 (2/2): Thiên lệch, giới hạn ngữ cảnh, chất lượng (1.0 phút)

"Ba hạn chế khác, mỗi cột cùng khuôn: là gì, input, output, tiêu chí, kết quả.

**Thiên lệch**: đưa cặp câu 'Thợ điện nói rằng **anh ấy / chị ấy** sẽ đến muộn', đo xác suất model cho câu nữ, gọi là share_female; 0.5 là không nghiêng. Hình scatter: trục ngang là % nữ thật của nghề, trục dọc là share_female. Luật ghi trước: thiên lệch nếu **Pearson r > 0 và p < 0.05**. Kết quả: tiếng Anh r = **0.69**, tiếng Việt r = **0.44**, 7/7 biến thể câu đạt → **có thiên lệch theo nghề**.

**Giới hạn ngữ cảnh**: giấu câu 'mã số là 7429' trong tài liệu 4,000 token. Còn trong input thì đúng **6/6**; cắt còn 1,024 token cuối thì model **bịa '1234'**. Hình chi phí: 512 lên 8,192 token, thời gian **×85**, bộ nhớ **10.4 / 14.6 GiB** của T4. ⏩ Bộ nhớ tăng gần theo **bình phương độ dài** vì lúc đọc prompt, attention so mọi cặp token; nên dù model cho 32,768 token, T4 chỉ chạy được khoảng 8 nghìn.

**Chất lượng phụ thuộc tác vụ và dữ liệu**: phân loại 0.92, hỏi kiến thức không tài liệu 0–0.20, có RAG 0.67; câu góp ý chỉ 0.79 so với câu thường 0.96. → **Không có một con số cho mọi việc.**"

---

## Slide 20 — Mục 10: Thử nghiệm có kiểm soát (1.1 phút + notebook)

"Mục 10. Câu hỏi: thêm 4 ví dụ vào prompt có làm accuracy khác zero-shot **một cách có ý nghĩa thống kê** không?

Dải thiết lập 5 bước: cùng model, cùng **2,999 câu test**, mẫu prompt chọn trên **validation**, **biến duy nhất** là 0 hay 4 ví dụ, sinh greedy.

Kết quả: 89.86% ± 1.08 và 92.30% ± 0.95, công thức khoảng tin cậy ở trên.

Nhưng hai cách chấm **trên cùng câu**, nên em dùng **McNemar** cho dữ liệu ghép cặp. Bảng 2 × 2: **hàng là zero-shot đúng/sai, cột là few-shot đúng/sai**. Ô a = 2,648 câu cả hai đúng, d = 184 câu cả hai sai — hai ô này **không cho thông tin** về cách nào tốt hơn. Chỉ **167 câu bất đồng** có ích: b = 47 câu chỉ zero-shot đúng, c = 120 câu chỉ few-shot đúng. Nếu hai cách như nhau thì b ≈ c ≈ 83. Xác suất ra lệch như 47 so với 120 khi hai cách như nhau là **p = 1.5 × 10⁻⁸**, nhỏ hơn 0.05 → few-shot tốt hơn **có ý nghĩa thống kê**.

[Mở notebook, cell 10.2: chỉ bảng a, b, c, d và dòng p-value.] Đây là output thật, đọc từ cột 'correct' của hai lần chạy.

Giới hạn: 1 model, 1 bộ dữ liệu, 1 bộ ví dụ — đổi ví dụ hoặc thứ tự có thể khác, nên **không suy rộng cho LLM nói chung**."

---

## Slide 21 — Tiêu chí đánh giá: đủ 18 metric (1.0 phút)

"Slide này đối chiếu **đủ 18 metric trong danh sách tham khảo của cô**. Mỗi dòng đọc: là gì, đo trên input – output nào, kết quả.

Năm nhóm. **Dự đoán ngôn ngữ**: cross-entropy và perplexity ở mục 4. **Độ đúng, trùng khớp**: accuracy ở mục 7; Exact Match và token F1 ở mục 8, **15 câu**; BLEU và ROUGE ở mục 1 — **chỉ 1 câu**, nên là minh họa, không phải số đánh giá. **Ngữ nghĩa** bằng DeepEval và G-Eval. **Sinh mã**: unit-test pass rate và Pass@k — **1 bài, 10 mẫu**. **Hiệu quả thực thi**: TTFT 0.045 giây, latency 2.3 đến 8.4 giây, khoảng **26 token mỗi giây**, VRAM đỉnh **2.90 GiB**, chi phí 2.3 đến 8.4 GPU-giây mỗi yêu cầu.

Khung đảm bảo: greedy, luật chấm ghi trước, chọn prompt trên validation, mọi số đọc từ file. ⏩ Giám khảo LLM được em đối chiếu với nhãn chấm tay: G-Eval Correctness khớp **31/36**, Hallucination **32/36**. Giới hạn: giám khảo Qwen2.5-7B **cùng họ** với model được chấm nên có thể thiên vị."

---

## Slide 22 — So khớp chữ và DeepEval (0.8 phút)

"Hai nhóm metric em chốt, mỗi metric có **ví dụ thật**.

Bên trái là **so khớp chữ**. **Exact Match**: '20.4%' khớp đáp án được 1; còn 'Hoàng đế phải thông qua Hội đồng để ban hành luật' **đúng ý nhưng dài hơn** nên được 0. **Token F1** cho cùng câu đó: 11 từ, trùng đủ 5 từ của đáp án → precision 5/11, recall 1, F1 **0.625**. **BLEU**: trùng 1–4 gram là 4/7, 2/6, 1/5, 0/4 → 26.3.

Bên phải là **DeepEval**: giám khảo **Qwen2.5-7B** 4-bit chạy local, cả ba metric cao là tốt. **Answer Relevancy** đo output có bám câu hỏi — câu 'ký tên' **sai** vẫn được 1.0, vì metric này **không đo đúng sai**. **Faithfulness** đo output có bám tài liệu RAG — bắt được câu 'Nam Ossetia thuộc Nga' trái tài liệu. **Hallucination**: câu '20.4%' **đúng** mà giám khảo cho 0 — **giám khảo cũng sai**.

Kết luận dải dưới: mỗi metric một mặt, nên **kết quả chính vẫn là nhãn 3 mức có đọc tay**, RAG 0/15 lên 10/15."

---

## Slide 23 — G-Eval và hiệu quả thực thi (0.8 phút)

"Bên trái là **G-Eval**: giám khảo nhận test case, chấm theo **các bước chấm ghi sẵn** cho từng tiêu chí, cho điểm 0–10, quy về 0–1, từ 0.5 là đạt. Năm tiêu chí ở bảng trên. Bảng dưới: Instruct 0.82 so với base 0.00 về làm theo chỉ dẫn; T = 1.5 mạch lạc 0.00; RAG tăng Correctness 0.22 lên 0.38 — **cùng chiều với chấm tay** ở các mục trước.

Bên phải là **hiệu quả thực thi**, đo 5 lần lấy trung vị. Sơ đồ: **prefill** đọc cả prompt một lượt, ra token đầu — thời gian đó là **TTFT**; sau đó mỗi token cần một lượt model, xong cả câu là **latency**.

Bảng: TTFT khoảng **0.045 giây** dù prompt 42 hay 165 token, vì prefill chạy song song; latency tăng theo **số token ra**, 2.31 giây cho 60 token, 8.44 giây cho 219 token; throughput ổn định **26 token/giây**. VRAM đỉnh 2.90 GiB, chỉ hơn trọng số 0.01 GiB vì KV cache vài trăm token rất nhỏ. Kaggle miễn phí nên chi phí báo bằng **GPU-giây**."

---

## Slide 24 — Phân tích lỗi và tổng hợp (0.8 phút)

"Slide cuối nội dung. Trái trên là **confusion matrix** 2 × 2 cho zero-shot và few-shot: hàng là nhãn thật, cột là nhãn model đoán. Tổng câu sai giảm **304 xuống 231**.

Em đọc tay **30 câu few-shot sai**, với **nhóm lỗi đặt trước khi đọc**. Nhiều nhất là **ý ngầm, góp ý: 20/30** — ví dụ 'giảm lý thuyết, tăng thực hành' bị gán tiêu cực mà model đoán tích cực. ⏩ Câu góp ý chiếm 76–81% lỗi theo hướng này: lỗi chủ yếu do **quy ước gán nhãn** của bộ dữ liệu, không hẳn do model.

Bảng phải tổng hợp mọi mục, **đọc thẳng từ file kết quả**.

Kết luận: LLM là model **đoán token tiếp theo**, học bằng **cross-entropy**, rồi **post-training** để làm theo chỉ dẫn; sinh văn bản bằng cách **chọn từng token**; học được từ ví dụ và bổ sung tri thức qua RAG — nhưng **vẫn bịa, thiên lệch, giới hạn ngữ cảnh**, nên **mọi output cần kiểm chứng**."

---

## Slide 25 — Tài liệu tham khảo (0.2 phút)

"Đây là tài liệu tham khảo: các bài báo gốc, model, bộ dữ liệu và thư viện; danh sách đầy đủ ở `docs/references.md`. Em cảm ơn cô và các bạn đã lắng nghe."

---

## Phụ lục A — Bảng thời gian

| Slide | Phút (bỏ ⏩) | Ghi chú |
|---|---|---|
| 1–2 | 0.9 | |
| 3–4 | 2.0 | |
| 5–7 | 2.8 + notebook 0.3 | cell 2.4 |
| 8–9 | 1.5 | |
| 10 | 1.0 | |
| 11–12 | 1.8 + notebook 0.3 | cell 5.3 |
| 13–14 | 1.7 | |
| 15 | 1.1 | |
| 16–17 | 1.7 | |
| 18–19 | 1.9 | |
| 20 | 1.1 + notebook 0.3 | cell 10.2 |
| 21–23 | 2.6 | |
| 24–25 | 1.0 | |
| **Tổng** | **≈ 19.5** | Nói đủ ⏩: ≈ 22–23 phút |

## Phụ lục B — Câu hỏi dễ gặp, trả lời ngắn

| Câu hỏi | Trả lời |
|---|---|
| Vì sao chỉ dùng model 1.5B? | Vừa T4 miễn phí (trọng số 2.89 GiB), chạy cả bài 41 phút; bài cần minh họa cơ chế, không so model. |
| BLEU, ROUGE chỉ 1 câu thì có ý nghĩa gì? | Chỉ minh họa công thức và giới hạn (đúng nghĩa vẫn BLEU thấp); số đánh giá chính là accuracy 2,999 câu và RAG 15 câu. |
| Pass@k = 1.0 có đáng tin? | Bài `is_even` quá dễ nên bão hòa; đo thật cần HumanEval (164 bài). |
| Giám khảo LLM có tin được không? | Đã đối chiếu nhãn tay: G-Eval Correctness khớp 31/36, Hallucination 32/36; vẫn có ca sai ("20.4%"), và cùng họ Qwen nên có thể thiên vị. |
| Sao không dùng t-test mà dùng McNemar? | Hai cách chấm trên cùng câu, kết quả đúng/sai (nhị phân, ghép cặp) → McNemar; chỉ 167 câu bất đồng mang thông tin. |
| Vì sao 8,192 token tốn 10.4 GiB khi trọng số chỉ 2.9? | KV cache chỉ 0.22 GiB; phần lớn ~7.3 GiB là ma trận attention n × n lúc prefill (12 head × 8,192²), tăng theo bình phương độ dài. |
| Sao TTFT gần như không đổi khi prompt dài gấp 4? | Prefill tính mọi token của prompt song song trong một lượt; decode mới chạy tuần tự từng token. |
| Few-shot có luôn tốt hơn? | Không: lỗi tích cực → tiêu cực tăng 56 → 86; kết quả phụ thuộc ví dụ và thứ tự (Zhao et al. 2021). |
| RAG có hết bịa không? | Không: bịa 2/3 câu kho không có đáp án; và chỉ tốt bằng retrieval (recall@3 = 13/18). |
| Có fine-tune không? | Không (góp ý của cô 11/11); xem tác dụng post-training qua base vs Instruct. |

# Mục 1 — Giới thiệu: LLM là gì, 4 tác vụ

- **Created**: 2026-11-28 (Toàn, AI hỗ trợ, Claude Code). Trạng thái **[đề xuất]**, chờ Toàn duyệt.
- **Nội dung:** lý thuyết LM / LLM / chatbot, 4 tác vụ hỏi đáp, tóm tắt, dịch, sinh mã (cell 1.1–1.6).
- **Số liệu:** run chính thức `20261128_041850_normal` (Kaggle, GPU T4 x2, float16, seed 42, 22.0 phút, 188 cell; G-Eval chấm 20 lần lấy trung bình). Phần 1–9 và 10.1–10.3b trùng từng file kết quả với run `20261127_234527_normal`, `20261125_214650_normal`, `20261121_010447_normal` (chỉ khác cột thời gian, bộ nhớ); riêng 10.3c, 10.3d đổi theo giám khảo (xem mục 10). **VRAM 6.6, 9.5 lấy run `20261127_234527_normal`**: run chính thức đo lệch +2.97 GiB vì model của lần chạy dừng giữa chừng còn trên GPU. Bảng ở `outputs/results/`, hình ở `outputs/figures/`.
- **Ký hiệu nguồn:** **[Slide ChX]** slide môn học; **[Bổ sung]** nguồn có trong `docs/references.md`; **[Tự tính]** tính lại từ số trong notebook; **[Suy luận]** giải thích của nhóm, chưa đo.
- **Code** chép từ `notebook/llm_presentation.ipynb`; dòng ghi log, gọi `show_input()` và vài dòng in phụ được lược (`# ...`). Mở notebook, tìm số cell (ví dụ "# 4.2") để xem nguyên cell.
- **Bộ đánh giá** (thẻ ở notebook cell 10.5, `figures/suite_*.png`): E1, E2.

---

> **Cô yêu cầu → notebook đáp ứng**
> - Nội dung cần trình bày: *LLM là gì? Phân biệt language model, LLM và ứng dụng chatbot*.
> - Yêu cầu minh họa: *Ví dụ các tác vụ: hỏi đáp, tóm tắt, dịch, sinh mã*.
> - Đáp ứng: Note lý thuyết (LM → n-gram → neural LM → LLM → chatbot); cell 1.2: 4 tác vụ, cùng model, chỉ đổi prompt; cell 1.3 chạy thử code model sinh; cell 1.4 chấm 4/4.
> - Góp ý 11/11 và yêu cầu thêm 3: demo trên model thật, output thật, có phân tích.

## Note lý thuyết để hiểu (dùng cho cả phần 1)

**Language model (LM)** [Bổ sung: Jurafsky & Martin ch. 3]: model gán xác suất $P(w_1, \dots, w_n)$ cho một chuỗi từ. Theo quy tắc dây chuyền (§1), việc này **tương đương** với dự đoán từ tiếp theo:

$$P(w_1, \dots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \dots, w_{i-1})$$

Công thức này **đúng tuyệt đối**, không xấp xỉ. Khó khăn nằm ở chỗ: ngữ cảnh $w_1, \dots, w_{i-1}$ càng dài thì càng hiếm khi gặp lại y hệt trong dữ liệu, nên không ước lượng trực tiếp được. Hai lời giải dưới đây (n-gram, neural LM) khác nhau ở **cách ước lượng** $P(w_i \mid \text{ngữ cảnh})$.

#### a) n-gram: ước lượng bằng đếm [Bổ sung: Jurafsky & Martin ch. 3]

**Bước 1 — Giả định Markov:** chỉ giữ $n-1$ từ gần nhất làm ngữ cảnh, bỏ phần xa hơn:

$$P(w_i \mid w_1, \dots, w_{i-1}) \approx P(w_i \mid w_{i-n+1}, \dots, w_{i-1})$$

- $n = 1$ (**unigram**): $P(w_i)$, không nhìn ngữ cảnh.
- $n = 2$ (**bigram**): $P(w_i \mid w_{i-1})$, chỉ nhìn 1 từ trước.
- $n = 3$ (**trigram**): $P(w_i \mid w_{i-2}, w_{i-1})$, nhìn 2 từ trước.

Đây mới là chỗ **xấp xỉ** (khác với quy tắc dây chuyền ở trên).

**Bước 2 — Đếm: ước lượng hợp lý cực đại (MLE).** Đếm số lần cụm xuất hiện trong kho văn bản huấn luyện, $C(\cdot)$ = số lần đếm được:

$$P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}\, w_i)}{C(w_{i-1})}, \qquad P_{\text{MLE}}(w_i \mid w_{i-n+1:i-1}) = \frac{C(w_{i-n+1:i-1}\, w_i)}{C(w_{i-n+1:i-1})}$$

- **Tử số:** số lần cả cụm (ngữ cảnh + từ đang xét) xuất hiện liền nhau.
- **Mẫu số:** số lần ngữ cảnh xuất hiện. Trong bigram, đó là số bigram bắt đầu bằng $w_{i-1}$; nhờ vậy, tổng xác suất các từ sau một ngữ cảnh bằng đúng 1.
- **Vì sao gọi là "hợp lý cực đại":** tỉ lệ đếm này là nghiệm của bài toán $\max \sum_{h,w} C(h, w) \ln P(w \mid h)$ với ràng buộc $\sum_w P(w \mid h) = 1$ (giải bằng nhân tử Lagrange). Nói cách khác, đó là bộ xác suất làm kho huấn luyện có xác suất cao nhất. Neural LM và LLM tối ưu **cùng mục tiêu** này (cross-entropy, phần 4); chỉ khác ở chỗ xác suất được tính bằng mạng nơ-ron thay vì tra bảng đếm.
- **Token biên:** thêm `<s>` vào đầu câu để từ đầu tiên có ngữ cảnh, và `</s>` vào cuối câu để model học được lúc nào câu kết thúc. Đây chính là ý tưởng của `<|endoftext|>`, `<|im_end|>` ở phần 0, 5.

**Ví dụ tính tay** (kho 3 câu, Jurafsky & Martin §3.1):
```text
<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
```
| Bigram | Đếm | Xác suất MLE |
|---|---|---|
| $P(\text{I} \mid \text{<s>})$ | $C(\text{<s> I}) = 2$, $C(\text{<s>}) = 3$ | $2/3$ |
| $P(\text{Sam} \mid \text{<s>})$ | $1/3$ | $1/3$ |
| $P(\text{am} \mid \text{I})$ | $C(\text{I am}) = 2$, $C(\text{I}) = 3$ | $2/3$ |
| $P(\text{do} \mid \text{I})$ | $1/3$ | $1/3$ |
| $P(\text{Sam} \mid \text{am})$ | $C(\text{am Sam}) = 1$, $C(\text{am}) = 2$ | $1/2$ |
| $P(\text{</s>} \mid \text{Sam})$ | $C(\text{Sam </s>}) = 1$, $C(\text{Sam}) = 2$ | $1/2$ |

Xác suất của câu "I am Sam" [Tự tính]:

$$P(\text{<s> I am Sam </s>}) = \tfrac{2}{3} \cdot \tfrac{2}{3} \cdot \tfrac{1}{2} \cdot \tfrac{1}{2} = \tfrac{1}{9} \approx 0.111$$

Thực tế nhân nhiều xác suất nhỏ sẽ bị tràn số dưới (underflow), nên người ta cộng log: $\ln P = \sum \ln p$. Đây cũng là lý do loss ở phần 4 dùng $-\ln p$.

**Bước 3 — Vấn đề số 0 (dữ liệu thưa).** Bigram nào không có trong kho thì MLE cho xác suất **đúng bằng 0**. Ví dụ $C(\text{Sam am}) = 0$ nên $P(\text{am} \mid \text{Sam}) = 0$. Hệ quả: mọi câu chứa cụm "Sam am" có xác suất 0, và perplexity (cell 4.4) thành vô cực, dù câu đó có thể hoàn toàn hợp lệ. Với $V$ từ, số n-gram có thể có là $V^n$. Ví dụ, với từ điển Qwen $V = 151{,}936$, riêng bigram đã có $V^2 \approx 2.3 \times 10^{10}$ ô [Tự tính]. Không kho văn bản nào phủ hết, và $n$ càng lớn thì càng thưa.

**Bước 4 — Làm trơn (smoothing):** lấy bớt xác suất của cụm đã gặp, chia cho cụm chưa gặp.
| Cách | Công thức | Ý |
|---|---|---|
| **Laplace (add-one)** | $P(w_i \mid w_{i-1}) = \dfrac{C(w_{i-1} w_i) + 1}{C(w_{i-1}) + V}$ | Cộng 1 vào mọi ô đếm. Mẫu số cộng $V$ để tổng vẫn bằng 1 |
| **Add-k** | $\dfrac{C(w_{i-1} w_i) + k}{C(w_{i-1}) + kV}$, $0 < k < 1$ | Cộng ít hơn 1, vì add-one lấy quá nhiều xác suất từ cụm đã gặp |
| **Interpolation** | $\hat P(w_i \mid w_{i-2} w_{i-1}) = \lambda_1 P(w_i \mid w_{i-2} w_{i-1}) + \lambda_2 P(w_i \mid w_{i-1}) + \lambda_3 P(w_i)$, $\sum \lambda = 1$ | Trộn trigram, bigram, unigram. Các $\lambda$ chọn trên tập held-out |
| **Backoff** | Có trigram thì dùng; đếm bằng 0 thì lùi xuống bigram, rồi unigram (có chiết khấu để tổng vẫn bằng 1) | Dùng ngữ cảnh dài nhất mà dữ liệu có |
| **Kneser-Ney** | Trừ một lượng cố định $d$ khỏi mỗi ô đếm, chuyển sang phân phối bậc thấp. Phân phối bậc thấp tính theo **số ngữ cảnh khác nhau** mà từ đã đi sau, không theo số lần xuất hiện | Cách làm trơn n-gram phổ biến, thường cho perplexity thấp nhất. Ví dụ: "Kong" xuất hiện nhiều nhưng gần như chỉ sau "Hong", nên khó được chọn trong ngữ cảnh mới |

Ví dụ Laplace trên kho trên [Tự tính]: $V = 11$ (11 loại từ có thể được đoán, gồm `</s>`, không gồm `<s>`). Khi đó $P(\text{am} \mid \text{I}) = (2+1)/(3+11) = 3/14 \approx 0.214$, giảm từ $0.667$; còn $P(\text{am} \mid \text{Sam}) = (0+1)/(2+11) = 1/13 \approx 0.077$, tăng từ 0. Đó đúng là "lấy bớt cái đã gặp, chia cho cái chưa gặp".

**Ba giới hạn của n-gram dù đã làm trơn** (đây là lý do Bengio 2003 đề xuất neural LM):
1. **Ngữ cảnh ngắn cố định:** chỉ nhìn $n-1$ từ, thường $n \le 5$, vì $n$ lớn thì bảng quá thưa. Từ ở xa hơn bị bỏ hẳn.
2. **Không có khái niệm "từ giống nhau":** "mèo" và "chó" là hai ô đếm độc lập. Gặp "con mèo đang chạy" 100 lần cũng không giúp gì cho "con chó đang chạy".
3. **Bùng nổ số tham số:** mỗi n-gram là một tham số riêng, nên số tham số tăng theo $V^n$.

#### b) Neural LM của Bengio et al. 2003 [Bổ sung: Bengio, Ducharme, Vincent, Jauvin, *A Neural Probabilistic Language Model*, JMLR 2003]

**Vấn đề bài báo nêu ("lời nguyền số chiều"):** muốn mô hình hoá phân phối của 10 từ liên tiếp với từ điển 100,000 từ thì có tới $100{,}000^{10} - 1 = 10^{50} - 1$ tham số tự do. Đếm không bao giờ đủ dữ liệu cho chừng ấy.

**Ý tưởng:** vẫn giữ giả định Markov ($n-1$ từ trước), nhưng thay **bảng đếm** bằng **một hàm có tham số** $f$, gồm hai phần:
1. Mỗi từ được gán một **vector đặc trưng** $C(w) \in \mathbb{R}^m$, học được trong lúc huấn luyện. Đây là **word embedding**: tất cả vector xếp thành ma trận $C$ kích thước $|V| \times m$, dùng chung cho mọi vị trí trong ngữ cảnh.
2. Một **mạng nơ-ron** nhận các vector của $n-1$ từ trước và trả ra xác suất cho từ tiếp theo.

**Kiến trúc (theo bài gốc):**

$$x = \big[C(w_{t-1});\, C(w_{t-2});\, \dots;\, C(w_{t-n+1})\big] \in \mathbb{R}^{(n-1)m} \quad \text{(nối các vector ngữ cảnh)}$$

$$y = b + Wx + U \tanh(d + Hx) \in \mathbb{R}^{|V|}$$

$$P(w_t = i \mid w_{t-n+1:t-1}) = \frac{e^{y_i}}{\sum_j e^{y_j}} \quad \text{(softmax trên cả từ điển)}$$

| Ký hiệu | Ý nghĩa | Tương ứng trong Qwen2.5 (notebook) |
|---|---|---|
| $C$ ($\lvert V\rvert \times m$) | Bảng embedding, tra hàng theo từ | $E$ ($151{,}936 \times 1{,}536$), cell 3.4 |
| $H$, $d$, $\tanh$ | Lớp ẩn: trộn thông tin các từ ngữ cảnh | 28 decoder block (attention + FFN), phần 2 |
| $U$, $b$ | Lớp ra: biến lớp ẩn thành $\lvert V\rvert$ điểm (logits) | LM head (dùng chung $E$), cell 2.1 |
| $W$ | Nối thẳng từ embedding sang đầu ra (tuỳ chọn, có thể đặt bằng 0) | Không có |
| softmax | Điểm thành xác suất | Giống hệt, cell 2.4 |

Tổng số tham số theo bài: $|V|(1 + nm + h) + h(1 + (n-1)m)$, với $h$ là số nơ-ron ẩn. Con số này **tăng tuyến tính** theo $|V|$ và $n$, không tăng theo $V^n$ như n-gram.

**Huấn luyện:** cực đại hoá log-likelihood trung bình trên kho văn bản, kèm điều chuẩn (weight decay) $R(\theta)$:

$$L = \frac{1}{T} \sum_t \ln f(w_t, w_{t-1}, \dots, w_{t-n+1}; \theta) + R(\theta)$$

Trong thí nghiệm, $R$ là weight decay áp lên trọng số mạng và ma trận $C$, **không** áp lên bias $b$, $d$. Bài dùng stochastic gradient ascent: sau mỗi từ $w_t$ của kho, cập nhật $\theta \leftarrow \theta + \varepsilon \,\partial \ln \hat P(w_t \mid w_{t-1}, \dots, w_{t-n+1}) / \partial \theta$, với $\theta = (b, d, W, U, H, C)$. Như vậy embedding $C$ được học **cùng lúc** với trọng số mạng. Cực đại $L$ chính là cực tiểu cross-entropy ở phần 4, cùng mục tiêu với MLE của n-gram ở trên.

**Vì sao khái quát tốt hơn n-gram** (ví dụ trong bài gốc): đã gặp "The cat is walking in the bedroom" thì model cũng gán xác suất cao cho "A dog was running in a room". Lý do:
- Từ đóng vai trò giống nhau ("cat"/"dog", "walking"/"running", "the"/"a", "bedroom"/"room") xuất hiện trong ngữ cảnh giống nhau, nên gradient đẩy embedding của chúng **lại gần nhau**.
- $f$ là hàm **trơn** theo embedding, nên ngữ cảnh có embedding gần nhau cho ra xác suất gần nhau.
- Vì vậy, mỗi câu huấn luyện giúp được cả một vùng các câu "tương tự", thay vì chỉ cộng 1 vào đúng một ô đếm như n-gram.

**Kết quả trong bài báo (§4):**

| Kho | Dữ liệu huấn luyện | $\lvert V\rvert$ | Perplexity test so với n-gram tốt nhất |
|---|---|---|---|
| Brown | 800,000 từ (cả kho 1,181,041 từ) | 16,383 | thấp hơn khoảng **24%** |
| AP News | 13,994,528 từ | 17,964 | thấp hơn khoảng **8%** |

Mốc so sánh là n-gram đã làm trơn (trigram nội suy, Kneser-Ney, class-based). Cái giá: phần lớn phép tính nằm ở lớp ra (tính softmax trên toàn bộ $|V|$), nên bài phải chạy song song trên cụm CPU. Với AP News, 5 epoch mất khoảng 3 tuần trên 40 CPU.

**Giới hạn còn lại và cách LLM giải quyết:**
| Bengio 2003 | LLM hiện nay (Qwen2.5) |
|---|---|
| Ngữ cảnh vẫn cố định $n-1$ từ (giả định Markov) | Self-attention nhìn toàn bộ ngữ cảnh, tới 32,768 token (phần 2, mục 9) |
| Đơn vị là **từ**, từ lạ không xử lý được | **Subword** byte-level BPE, không có token lạ (phần 3) |
| Một lớp ẩn MLP $\tanh$ | 28 block Transformer, 1.54 tỉ tham số (cell 2.1) |
| Kho Brown và AP News: cỡ triệu đến chục triệu từ | 18 nghìn tỉ token (Qwen2.5 report) |

**Giống nhau từ 2003 đến nay:** embedding tra hàng theo ID, softmax trên từ điển, và mục tiêu cực đại log-likelihood của token tiếp theo. Ba ý này đều có trong demo ở phần 2, 3, 4.

#### c) LLM [Bổ sung: Zhao et al. 2023; Brown et al. 2020]

**LLM** là bước mở rộng của LM. Nó **vẫn làm đúng việc của LM**: dự đoán token tiếp theo theo quy tắc dây chuyền. Chỗ khác nằm ở **cách tính** xác suất đó. Thay cho bảng đếm (n-gram) hay mạng MLP với ngữ cảnh cố định $n-1$ từ (Bengio 2003), LLM dùng mạng nơ-ron **Transformer decoder-only** (phần 2).

Ở mỗi bước, mô hình không trả ra một từ chọn sẵn. Nó trả ra **một phân phối xác suất trên toàn bộ từ điển**: 151,936 token với Qwen2.5 (cell 2.4: sau "Hôm nay trời rất", " đẹp" được 0.788, " nắng" 0.063…). Muốn có cả câu thì lặp lại nhiều bước: chọn một token từ phân phối, nối vào input, rồi dự đoán tiếp (sinh tự hồi quy, phần 6). Vì vậy LLM **không** "dự đoán cả câu một lần". Nó dự đoán **từng token**, và câu trả lời là kết quả của nhiều lần dự đoán nối nhau.

Chữ **"lớn"** đến từ ba quy mô cùng tăng:
1. **Tham số:** hàng tỉ trở lên. Qwen2.5-1.5B có 1.54 tỉ (cell 0.5), GPT-3 có 175 tỉ [Brown 2020]. Không có ngưỡng chính thức: survey của Zhao et al. 2023 chủ yếu bàn các model từ hàng chục tỉ tham số, nên Qwen2.5-1.5B là LLM **cỡ nhỏ**, nhưng cùng kiến trúc và cùng quy trình huấn luyện với model lớn.
2. **Dữ liệu:** hàng trăm tỉ đến hàng nghìn tỉ token văn bản không nhãn. GPT-3 dùng khoảng 300 tỉ token; Qwen2.5 dùng 18 nghìn tỉ token.
3. **Ngữ cảnh:** self-attention cho mỗi token nhìn toàn bộ các token trước nó, không giới hạn $n-1$ từ. Qwen2.5-1.5B nhìn được tới 32,768 token (cell 0.5).

**Hệ quả của quy mô:** cùng một model làm được nhiều tác vụ (hỏi đáp, tóm tắt, dịch, sinh mã) chỉ bằng cách đổi prompt, không cần huấn luyện lại. Brown 2020 gọi đây là in-context learning (phần 1, phần 7). Bản base chỉ biết viết tiếp văn bản. Muốn model làm theo chỉ dẫn và hội thoại thì phải qua post-training: SFT + alignment (phần 5).

**Chatbot** [Bổ sung: OpenAI 2022]: là **ứng dụng**, không phải model: LLM đã qua post-training (bản Instruct) + giao diện hội thoại + quản lý lịch sử + lớp an toàn.

| | Language model | LLM | Chatbot |
|---|---|---|---|
| Là gì | Model xác suất trên chuỗi | LM Transformer cỡ lớn | Ứng dụng xây trên LLM |
| Ví dụ | n-gram, RNN LM | Qwen2.5-1.5B (base / Instruct) | ChatGPT, Qwen Chat |
| Làm được | Đoán từ tiếp | Nhiều tác vụ qua prompt | Hội thoại nhiều lượt |

**Vì sao một model làm nhiều tác vụ:** hỏi đáp, dịch, tóm tắt, viết code đều là "văn bản vào → sinh tiếp văn bản ra", có điều kiện theo prompt.

## Cell 1.1 — Hàm sinh cho bản Instruct

> **Lý thuyết ở cell này:** **chat template** = khung định dạng hội thoại mà bản Instruct đã thấy khi SFT (mục 5). Đưa prompt đúng khung thì model biết đây là câu hỏi cần trả lời. **Greedy** = mỗi bước chọn token xác suất cao nhất (lý thuyết đầy đủ: cell 6.1).

```python
def chat_generate(model, tok, prompt: str, max_new_tokens: int) -> tuple[str, int, int, float]:
    messages = [{"role": "user", "content": prompt}]
    # add_generation_prompt=True: thêm phần mở đầu lượt "assistant" để model trả lời
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    n_in = enc["input_ids"].shape[1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,                       # greedy: mỗi bước lấy token xác suất cao nhất
            temperature=None, top_p=None, top_k=None,  # tắt tham số lấy mẫu mặc định của bản Instruct
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_ids = out[0, n_in:]  # chỉ giữ phần model sinh thêm
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    return text, n_in, len(new_ids), elapsed
```

**Giải thích từng dòng:**
- `messages`: danh sách lượt hội thoại; ở đây một lượt `user`.
- `apply_chat_template(..., add_generation_prompt=True)`: bọc vào khung và thêm `<|im_start|>assistant` ở cuối để model biết **đến lượt nó**.
- `n_in`: số token đầu vào sau khi bọc.
- `torch.cuda.synchronize()`: GPU chạy **bất đồng bộ** với Python; phải chờ GPU xong mới bấm giờ, nếu không thời gian đo sai.
- `out[0, n_in:]`: `generate` trả **cả prompt lẫn phần sinh**, nên cắt bỏ `n_in` token đầu.
- `skip_special_tokens=True`: bỏ `<|im_end|>` khi đổi về chữ.

**Output thật (log khung sau chat template và lần thử "Xin chào"):**
```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Xin chào<|im_end|>
<|im_start|>assistant

INPUT  (prompt; 32 token sau chat template, gồm cả system prompt mặc định): Xin chào
OUTPUT (9/16 token, 2.50 s): 你好！有什么我可以帮助你的吗？
```

**Giải thích:**
- Qwen **tự thêm system prompt** tiếng Anh. Vì vậy "Xin chào" (2 từ) thành **32 token**.
- Lần đầu mất 2.50 s cho 9 token, các lần sau nhanh hơn nhiều (Dịch: 8 token 0.31 s). [Suy luận] Lần gọi GPU đầu tiên tốn thêm thời gian khởi tạo (nạp kernel CUDA).
- **Trả lời tiếng Trung** [Suy luận]: dữ liệu của Qwen có tỉ lệ tiếng Trung, tiếng Anh cao hơn nhiều tiếng Việt; system prompt tiếng Anh; "Xin chào" quá ngắn, ít ngữ cảnh tiếng Việt. Với greedy, chỉ cần token đầu có xác suất cao nhất là "你" thì cả câu trôi theo tiếng Trung, vì mỗi token sau dựa trên các token trước (vòng sinh, cell 6.1). *Cách kiểm:* dùng `top_k_next_tokens()` (cell 2.4) trên prompt đã qua chat template để xem top-5 token đầu.

## Cell 1.2 — Chạy 4 tác vụ

```python
TASKS: list[dict] = tasks_json["tasks"]   # data/tasks_prompts.json v2
assert [t["id"] for t in TASKS] == ["qa", "summary", "translate", "code"]
for idx, t in enumerate(TASKS):
    output, n_in, n_out, sec = chat_generate(model, tok, t["prompt"], t["max_new_tokens"])
    rec = {..., "truncated": n_out >= t["max_new_tokens"], "output": output, ...}  # chạm trần → bị cắt
```

**Output thật:**
| Tác vụ | Prompt | Output | Token ra |
|---|---|---|---|
| Hỏi đáp | Thủ đô của Nhật Bản là gì? Trả lời một câu. | "Thủ đô của Nhật Bản là Tokyo." | 10/32 |
| Tóm tắt | Đoạn 5 câu về môn Học sâu + "Tóm tắt đoạn trên trong 1 câu." | "Học sâu là một lĩnh vực học máy sử dụng mạng nơ-ron nhiều lớp…, bao gồm việc huấn luyện…, với hai phần giới thiệu…, kết thúc bằng việc mỗi nhóm thuyết trình…" | 104/256 |
| Dịch | Dịch sang tiếng Anh: Hôm nay trời rất đẹp. | "Today the weather is very beautiful." | 8/48 |
| Sinh mã | Viết hàm Python is_even(n)… | khối code `return n % 2 == 0` + `print` ví dụ + 2 đoạn giải thích | 216/512 |

**Giải thích:**
- **Cùng model, cùng greedy, chỉ đổi prompt → đổi tác vụ.** Đó là định nghĩa thực tế của "LLM đa tác vụ".
- `truncated` = False cả 4: không output nào chạm `max_new_tokens`, tức model **tự dừng** bằng `<|im_end|>` (học từ SFT).
- Tóm tắt dài 104 token: model 1.5B có xu hướng chép lại câu nguồn rồi nối bằng dấu phẩy; prompt chỉ yêu cầu "1 câu", không giới hạn số từ.
- Sinh mã 216 token: model tự thêm ví dụ và giải thích (văn phong "trợ lý hữu ích" học từ post-training).

## Cell 1.3 — Chạy `is_even` do model viết

```python
def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    return m.group(1) if m else text
ns: dict = {}                               # namespace riêng cho code của model
exec(model_code, ns)                        # KHÔNG phải sandbox: phải đọc code trước khi chạy
IS_EVEN_RESULTS = [ns["is_even"](n) for n in [0, 1, 2, -3]]
```

**Output thật:** in `True`, `False` (hai dòng `print` của model chạy khi `exec`), rồi `is_even[0, 1, 2, -3] → [True, False, True, False]`.

**Giải thích:**
- Regex lấy khối ```` ```python ... ``` ```` đầu tiên; `re.DOTALL` cho `.` khớp cả xuống dòng.
- `exec` chạy cả 2 dòng `print(is_even(4))`, `print(is_even(5))` của model → hai dòng `True`, `False` đầu tiên trong output.
- `is_even(-3)` = False đúng vì trong Python `-3 % 2 == 1` (dư lấy dấu của số chia) → `1 == 0` là False.
- Chấm bằng **chạy thật** chứ không đọc bằng mắt: chắc chắn hơn.

## Cell 1.4 — Chấm

| Tác vụ | Luật chấm (viết trước khi chạy) | Kết quả |
|---|---|---|
| Hỏi đáp | output chứa "tokyo" | đạt |
| Dịch | có "today" và một trong "weather/nice/beautiful/lovely" | đạt |
| Sinh mã | `is_even` chạy ra `[True, False, True, False]` | đạt (ghi chú: tự thêm lời dẫn, prompt không cấm) |
| Tóm tắt | chấm tay `MANUAL_JUDGE`: 1 câu, giữ ý chính, không thêm ý | đạt ("gần như ghép lại các câu gốc, rút gọn ít") |

**Tổng: đạt 4/4, bị cắt 0/4, thời gian sinh 13.2 s.**

## Cell 1.5 — Thử prompt khác (demo trực tiếp)
Prompt "Giải thích ngắn gọn mạng nơ-ron nhân tạo là gì, trong 2 câu." → "Mạng nơ-ron nhân tạo (Artificial Neural Network, ANN) là một mô hình仿生学的计算机系统…" (nửa Việt nửa Trung, 113 token). Cùng nguyên nhân với "Xin chào" ở cell 1.1: đây là **hạn chế phụ thuộc dữ liệu** (mục 9). Cell này không chấm.


## Cell 1.6 — Sinh mã: unit-test pass rate và Pass@k (thêm 24/11)

> **Lý thuyết:** code đúng có nhiều cách viết, nên chấm bằng **chạy test** thay vì so chữ. Chen et al. (HumanEval) chỉ ra cải thiện BLEU không kéo theo cải thiện tỉ lệ code đúng, vì nhiều chương trình khác chữ nhưng tương đương về chức năng [Bổ sung: Chen et al. 2021, #19].
> - **Unit-test pass rate** = số test qua / tổng test của một lời giải.
> - **Pass@k** = xác suất có ít nhất 1 trong k lời giải qua hết test. Ước lượng không chệch khi sinh n ≥ k mẫu, c mẫu đúng: $\text{Pass@}k = 1 - \binom{n-c}{k} / \binom{n}{k}$.

**Cách làm:** 7 test viết trước (4 ví dụ trong prompt + −4, 7, 10⁹). Chấm lời giải greedy của cell 1.3, rồi sinh thêm 10 mẫu (`do_sample=True`, T = 0.8, top-p 0.95, seed 42), tách hàm, `exec` từng mẫu với 7 test.

**Tự kiểm công thức:** n = 10, c = 3 → Pass@1 = 0.3, Pass@5 = 0.917 (`assert`).

**Kết quả** (`code_passk_samples.csv`, `code_passk_summary.csv`): greedy qua 7/7 test; 10/10 mẫu qua cả 7 test → Pass@1 = Pass@5 = **1.0**.

**Nhận xét:** bài `is_even` quá dễ nên kết quả bão hòa, chỉ minh họa cách tính. Muốn đo năng lực sinh mã phải dùng bộ nhiều bài như HumanEval (164 bài, n = 200 mẫu mỗi bài). `exec` không có sandbox, chỉ chấp nhận với hàm nhỏ tự kiểm.

## Câu hỏi tự kiểm

| Câu hỏi | Ý trả lời |
|---|---|
| LLM khác LM ở đâu? | Cùng bài toán đoán token tiếp theo; khác ở quy mô (tham số, dữ liệu) và kiến trúc Transformer |
| Sao dùng greedy? | Greedy luôn ra một kết quả, chạy lại y hệt, nên chấm và so sánh được |
| `max_new_tokens` là gì? | Trần số token được sinh thêm; chạm trần thì câu bị cắt |
| Sao model trả lời tiếng Trung? | Model 1.5B học nhiều dữ liệu tiếng Trung, Anh; với prompt ngắn nó dễ trượt sang ngôn ngữ chiếm ưu thế |

---

# Mục 1 — Giới thiệu: LLM là gì? (nội dung chi tiết, v1)

- Người phụ trách: Muội | Slide 3–4 | Thời lượng: khoảng 2 phút | Notebook: phần 1
- Đề của cô: "LLM là gì? Phân biệt language model, LLM và ứng dụng chatbot". Minh hoạ: ví dụ các tác vụ hỏi đáp, tóm tắt, dịch, sinh mã.
- Ký hiệu nguồn:
  - **[Slide ChX sN]**: slide môn, tra trong `Research/slides-ocr/c1_ocr.md`.
  - **[Bổ sung]**: nguồn ngoài, đã mở kiểm tra ngày 2026-11-14 (danh sách ở §9).
  - **[Tự tính]**: ví dụ nhóm tự tạo; đã chạy lại bằng Python.
- File research đi kèm: `research_muc01_gioi_thieu_v1.md`.

---

## 0. Mục này cần làm người nghe hiểu được gì

1. Language model là gì, giải thích bằng một ví dụ con số.
2. LLM khác language model thông thường ở đâu: mục tiêu giống nhau, quy mô và khả năng khác nhau.
3. Chatbot là **ứng dụng** xây trên LLM, không phải bản thân model.
4. Một LLM làm được nhiều tác vụ chỉ bằng cách đổi prompt. Đây là phần demo.

---

## 1. Language model (LM) là gì

### 1.1 Định nghĩa  [Bổ sung]
Language model là một model **dự đoán từ tiếp theo** (Jurafsky & Martin, ch. 3).

Nói đơn giản: cho đoạn "Hôm nay trời rất", model đưa ra xác suất cho mọi từ có thể đứng tiếp: "đẹp", "mưa", "nóng", …. Bàn phím điện thoại gợi ý từ tiếp theo chính là một language model nhỏ.

### 1.2 Dạng toán: quy tắc dây chuyền (chain rule)  [Bổ sung]
Xác suất của cả một câu được tách thành tích các xác suất "từ tiếp theo" (Bengio et al. 2003):

```
P(w1, w2, …, wT) = P(w1) · P(w2 | w1) · P(w3 | w1, w2) · … · P(wT | w1, …, wT−1)
```

Với câu ví dụ chung của nhóm:

```
P("Hôm nay trời rất đẹp")
  = P(Hôm) · P(nay | Hôm) · P(trời | Hôm nay) · P(rất | Hôm nay trời) · P(đẹp | Hôm nay trời rất)
```

Hệ quả: chỉ cần học tốt **một** việc là "đoán từ tiếp theo khi biết các từ trước", model đã tính được xác suất của cả câu, và sinh được câu bằng cách đoán từng từ một. Mục 4 (nguyên lý học) và mục 6 (sinh văn bản) đều xây trên công thức này.

### 1.3 Ví dụ cách làm cũ nhất: n-gram  [Bổ sung + Tự tính]
N-gram chỉ nhìn **n−1 từ gần nhất** và đếm tần suất trong dữ liệu. Bigram (n = 2) chỉ nhìn 1 từ trước.

Kho dữ liệu đồ chơi gồm 4 câu (nhóm tự đặt):

```
hôm nay trời đẹp
hôm nay trời mưa
hôm qua trời đẹp
nay trời đẹp quá
```

Đếm rồi chia [Tự tính, đã chạy lại bằng Python]:

| Xác suất | Cách tính | Kết quả |
|---|---|---|
| P(đẹp \| trời) | "trời đẹp" xuất hiện 3 lần / "trời" đứng trước 4 lần | 3/4 = 0.75 |
| P(mưa \| trời) | 1 / 4 | 0.25 |
| P(nay \| hôm) | 2 / 3 | ≈ 0.67 |
| P(đẹp \| nay) | "nay đẹp" chưa từng xuất hiện | **0** |

Dòng cuối là **điểm yếu**: một cụm chưa gặp bao giờ sẽ có xác suất bằng 0, dù câu đó hoàn toàn hợp lý. Bengio et al. (2003) gọi khó khăn này là "lời nguyền số chiều": câu dùng để kiểm thử hầu như luôn khác mọi câu trong dữ liệu huấn luyện.

### 1.4 Neural language model  [Bổ sung]
- Bengio et al. (2003) dùng mạng nơ-ron học **cùng lúc** hai thứ:
  - một vector biểu diễn cho mỗi từ (embedding, giống `nn.Embedding` ở Ch6);
  - xác suất của chuỗi từ.
- Các từ gần nghĩa có vector gần nhau, nên một câu chưa gặp vẫn được xác suất cao nếu nó gồm những từ giống các câu đã gặp.
- Kết quả: tốt hơn n-gram và tận dụng được ngữ cảnh dài hơn.
- **Liên hệ môn học:** RNN language model ở Ch6 cũng là một neural LM.

---

## 2. Large language model (LLM) là gì

### 2.1 Định nghĩa  [Bổ sung]
Theo bài survey của Zhao et al.:
- LLM là các language model pre-train có hàng chục đến hàng trăm tỉ tham số, xây trên kiến trúc Transformer.
- **Không có ngưỡng chính thức** cho chữ "large". Bài survey tự chọn tập trung vào model trên 10 tỉ tham số.

Jurafsky & Martin nhấn mạnh: LLM được xây **chỉ bằng cách huấn luyện để đoán từ tiếp theo**. Vậy LLM không có mục tiêu mới; nó vẫn là language model ở §1, chỉ khác:

| Khác ở | Language model nhỏ | LLM |
|---|---|---|
| Kiến trúc | n-gram, RNN, LSTM | Transformer (mục 2) |
| Số tham số | Từ vài nghìn đến vài triệu | Từ hàng tỉ trở lên |
| Dữ liệu | Một kho văn bản nhỏ, một lĩnh vực | Lượng văn bản khổng lồ (Qwen2.5: 18 nghìn tỉ token khi pretraining) |
| Đơn vị | Thường là từ | Token, tức mảnh của từ (mục 3) |
| Làm được | Một việc hẹp | Nhiều tác vụ chỉ bằng prompt |

### 2.2 Bốn thế hệ language model  [Bổ sung, theo Zhao et al.]

```
Statistical LM  →  Neural LM  →  Pre-trained LM  →  Large LM
  (n-gram)         (Bengio 2003,    (BERT, GPT-2:       (GPT-3 trở đi:
                    RNN/LSTM)        Transformer,        phóng to PLM,
                                     pre-train rồi       có in-context
                                     fine-tune)          learning)
```

| Thế hệ | Ý chính |
|---|---|
| Statistical LM | Đoán từ kế tiếp từ vài từ gần nhất; dùng cho truy xuất thông tin, nhận dạng tiếng nói |
| Neural LM | Mạng nơ-ron; học vector biểu diễn từ |
| Pre-trained LM | Transformer pre-train trên văn bản không nhãn, rồi fine-tune riêng cho từng tác vụ |
| LLM | PLM phóng to; xuất hiện khả năng mới như in-context learning (mục 7); một model giải nhiều tác vụ |

**Liên hệ slide [Slide Ch1 s10]:** dòng thời gian trên slide có BERT, GPT-3, ChatGPT (11/2022), GPT-4 (3/2023), trùng hai thế hệ cuối.

### 2.3 Model của nhóm nằm ở đâu
- Qwen2.5-1.5B có 1.54 tỉ tham số (model card).
- So với mốc trên 10 tỉ của survey thì đây là **LLM cỡ nhỏ**.
- Nhóm chọn cỡ này để chạy được trên GPU T4 miễn phí của Kaggle. Cơ chế giống model lớn; khác ở quy mô, nên chất lượng thấp hơn.

---

## 3. Chatbot: ứng dụng xây trên LLM

### 3.1 Chatbot không phải là model  [Bổ sung]
Survey của Zhao et al. mô tả ChatGPT là một chatbot xây dựa trên LLM. Nghĩa là ChatGPT là **sản phẩm**; bên trong là một LLM.

### 3.2 Từ LLM thành chatbot cần thêm gì

```
LLM sau pretraining (base)        ← chỉ biết viết tiếp văn bản
   + SFT (học từ hội thoại mẫu)    ← biết trả lời theo yêu cầu
   + Preference alignment (RLHF)   ← trả lời hữu ích, an toàn hơn
   = Model "instruct" / "chat"
   + Định dạng hội thoại (chat template, system prompt)
   + Giao diện, lưu lịch sử các lượt chat
   + Bộ lọc an toàn
   = Chatbot (ví dụ ChatGPT)
```

Căn cứ cho từng lớp:
- **SFT:** bước đầu khi huấn luyện ChatGPT là người huấn luyện tự viết hội thoại, đóng cả vai người dùng lẫn vai trợ lý (OpenAI, 2022).
- **RLHF:** ChatGPT được huấn luyện bằng RLHF, cùng phương pháp với InstructGPT, chỉ khác cách thu dữ liệu; model gốc là một model dòng GPT-3.5 (OpenAI, 2022). Mục 5 giải thích chi tiết.
- **Bộ lọc an toàn:** OpenAI dùng Moderation API để cảnh báo hoặc chặn một số loại nội dung không an toàn.
- **Chat template:** với Qwen, khi định dạng hội thoại, thư viện thêm đoạn `<|im_start|>assistant` ở cuối để báo cho model biết đến lượt trợ lý trả lời (docs chat templating của Hugging Face).

### 3.3 Liên hệ với model nhóm dùng
| Tên | Là gì |
|---|---|
| `Qwen/Qwen2.5-1.5B` | LLM base, chỉ qua pretraining. Model card ghi không khuyến nghị dùng bản base để hội thoại |
| `Qwen/Qwen2.5-1.5B-Instruct` | Đã qua pretraining và post-training. Vẫn **chưa phải chatbot**, vì chưa có giao diện và lớp an toàn |

### 3.4 Chatbot vẫn có giới hạn  [Bổ sung]
Ngay trong bài giới thiệu, OpenAI ghi:
- ChatGPT đôi khi trả lời nghe hợp lý nhưng sai hoặc vô nghĩa → **mục 9** (hallucination).
- ChatGPT nhạy với cách đặt câu hỏi: hỏi kiểu này thì nói không biết, đổi cách hỏi một chút lại trả lời đúng → **mục 7** (prompt ảnh hưởng kết quả).

---

## 4. Bảng phân biệt 3 khái niệm (đặt lên slide 3)

| | Language model | LLM | Chatbot |
|---|---|---|---|
| **Là gì** | Model dự đoán từ/token tiếp theo | Language model rất lớn, kiến trúc Transformer | **Ứng dụng** xây trên LLM |
| **Học bằng cách** | Đoán từ tiếp theo | Đoán token tiếp theo trên lượng dữ liệu khổng lồ | LLM + SFT + alignment (RLHF/DPO) |
| **Làm được** | Một việc hẹp: gợi ý từ, nhận dạng tiếng nói | Nhiều tác vụ chỉ bằng prompt | Hội thoại nhiều lượt, làm theo yêu cầu, từ chối yêu cầu không phù hợp |
| **Ví dụ** | n-gram; RNN LM (Ch6) | GPT-3; Qwen2.5-1.5B | ChatGPT |

Câu chốt dưới bảng: **"Mọi chatbot đều chạy trên một LLM, mọi LLM đều là một language model, nhưng chiều ngược lại thì không."**

---

## 5. Minh hoạ: một model, bốn tác vụ (slide 4, notebook phần 1)

### 5.1 Vì sao một model làm được nhiều việc  [Bổ sung]
- GPT-3 được thử trên nhiều tác vụ **mà không cập nhật gradient hay fine-tune**: tác vụ và ví dụ chỉ được đưa vào dưới dạng văn bản (Brown et al. 2020).
- Mọi tác vụ đều đưa được về dạng "văn bản vào → văn bản ra". Model chỉ việc "viết tiếp" prompt, đúng công thức ở §1.2.
- Sinh code cũng là viết tiếp văn bản: Codex là một model GPT huấn luyện thêm trên code GitHub, được đánh giá bằng bộ HumanEval (sinh hàm Python từ docstring) (Chen et al. 2021).

### 5.2 Prompt đề xuất  [đề xuất — Muội chốt]

Model: `Qwen2.5-1.5B-Instruct`, định dạng bằng chat template. Sinh greedy (`do_sample=False`), vì bản instruct mặc định bốc mẫu ngẫu nhiên (xem `research_presentation_llm_v1.md` §1).

| Tác vụ | Prompt | Cách kiểm output |
|---|---|---|
| Hỏi đáp | `Thủ đô của Nhật Bản là gì? Trả lời một câu.` | So với đáp án: Tokyo |
| Tóm tắt | Một đoạn khoảng 5 câu nhóm tự viết (ví dụ giới thiệu môn Học sâu) + `Tóm tắt đoạn trên trong 1 câu.` | Người đọc kiểm có giữ ý chính không, có thêm ý không có trong đoạn không |
| Dịch | `Dịch sang tiếng Anh: Hôm nay trời rất đẹp.` | So với câu ví dụ chung: "The weather is very nice today." |
| Sinh mã | `Viết hàm Python is_even(n) trả về True nếu n là số chẵn.` | Chạy hàm sinh ra với n = 0, 1, 2, −3 |

- Đoạn văn cho tác vụ tóm tắt phải do nhóm tự viết, để không dính bản quyền và biết chắc nội dung gốc.
- Code do model sinh ra: chạy trong cell riêng; ghi rõ là output của model.

### 5.3 Phân tích sau output (cell markdown, 2–3 câu)
Khung viết, điền sau khi chạy thật:
> "Cùng một model, cùng cách sinh greedy, chỉ đổi prompt là model chuyển từ hỏi đáp sang tóm tắt, dịch, viết code. Tác vụ … đúng; tác vụ … sai/thiếu ở chỗ … . Điều này cho thấy LLM là model đa tác vụ nhưng vẫn phải kiểm output (liên hệ mục 9)."

Không điền "đúng/sai" trước khi có output thật.

---

## 6. Nội dung slide

### Slide 3 — LLM là gì?
- **Tiêu đề:** Language model, LLM và chatbot
- **Dòng 1:** Language model = model dự đoán từ tiếp theo, kèm công thức chain rule một dòng.
- **Giữa:** bảng ở §4.
- **Dưới:** câu chốt ở §4.
- **Góc:** nguồn: Jurafsky & Martin; Zhao et al. 2023; OpenAI 2022.

### Slide 4 — Một model, nhiều tác vụ
- **Tiêu đề:** LLM làm được những tác vụ gì
- **Lưới 2×2:** mỗi ô = tên tác vụ + prompt (rút gọn) + output thật (rút gọn).
- **Dưới:** một dòng phân tích (§5.3).
- **Góc:** "Notebook phần 1, cell …" · model Qwen2.5-1.5B-Instruct · greedy.

---

## 7. Lời thoại (khoảng 2 phút)

> **[Slide 3]** "Trước hết, language model là gì? Đó là model dự đoán từ tiếp theo. Ví dụ cho 'Hôm nay trời rất', model cho biết xác suất của 'đẹp', 'mưa', 'nóng'. Bàn phím điện thoại gợi ý từ chính là một language model nhỏ.
> Xác suất cả câu được tách thành tích các xác suất 'từ tiếp theo', nên chỉ cần học tốt việc đoán từ tiếp theo là đủ.
> LLM vẫn là language model, vẫn học bằng cách đoán từ tiếp theo, nhưng lớn hơn rất nhiều: hàng tỉ tham số, kiến trúc Transformer, học trên lượng văn bản khổng lồ. Chưa có ngưỡng chính thức cho chữ 'large'; model nhóm dùng có 1.5 tỉ tham số, là LLM cỡ nhỏ.
> Còn chatbot như ChatGPT là một ứng dụng: lấy một LLM, dạy thêm bằng hội thoại mẫu và phản hồi của con người, rồi thêm giao diện và bộ lọc an toàn.
> **[Slide 4]** Vì mọi tác vụ đều là 'văn bản vào, văn bản ra', một LLM làm được nhiều việc chỉ bằng cách đổi prompt. Đây là output thật của Qwen 1.5B Instruct trên bốn tác vụ … [đọc kết quả thật]. Em mở notebook phần 1 để thầy cô xem code."

---

## 8. Q&A dự kiến

| # | Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|---|
| 1 | LM và LLM khác nhau ở đâu? | Cùng mục tiêu đoán token tiếp theo. LLM khác ở quy mô (tham số, dữ liệu), kiến trúc Transformer và khả năng mới như in-context learning | Jurafsky & Martin; Zhao et al. |
| 2 | Bao nhiêu tham số thì gọi là "large"? | Chưa có ngưỡng chính thức; survey của Zhao et al. chọn mốc trên 10 tỉ. Model nhóm dùng 1.54 tỉ là LLM cỡ nhỏ | Zhao et al.; model card Qwen |
| 3 | ChatGPT có phải là một LLM không? | ChatGPT là ứng dụng chatbot; bên trong là model fine-tune từ dòng GPT-3.5 bằng SFT và RLHF | OpenAI 2022 |
| 4 | Vì sao n-gram không đủ? | Chỉ nhìn vài từ gần nhất, và cụm chưa gặp có xác suất 0 (ví dụ "nay đẹp" ở §1.3). Neural LM học embedding nên tổng quát hoá cho câu chưa gặp | Bengio et al. 2003 |
| 5 | Vì sao một model làm được nhiều tác vụ mà không train lại? | Tác vụ được mô tả bằng văn bản trong prompt; model chỉ viết tiếp. GPT-3 làm vậy mà không cập nhật trọng số | Brown et al. 2020 |
| 6 | RNN ở Ch6 có phải language model không? | Có, nếu được huấn luyện để đoán từ tiếp theo. Nó không phải LLM vì nhỏ và không phải Transformer | Slide Ch6; Bengio et al. 2003 |
| 7 | Qwen2.5-1.5B-Instruct có phải chatbot không? | Chưa. Nó là model đã qua post-training; thành chatbot khi có thêm giao diện hội thoại và lớp an toàn | Model card Qwen; OpenAI 2022 |

---

## 9. Tài liệu tham khảo cho mục 1

| Tài liệu | Dùng cho |
|---|---|
| Jurafsky & Martin, *Speech and Language Processing*, ch. 3 "N-gram Language Models" (draft 19/08/2026). https://web.stanford.edu/~jurafsky/slp3/3.pdf | Định nghĩa LM; LLM xây bằng cách đoán từ |
| Bengio, Ducharme, Vincent, Jauvin, *A Neural Probabilistic Language Model*, JMLR 3:1137–1155, 2003. https://www.jmlr.org/papers/v3/bengio03a.html | Chain rule; hạn chế n-gram; neural LM |
| Zhao et al., *A Survey of Large Language Models*, 2023. https://arxiv.org/abs/2303.18223 | Định nghĩa LLM; 4 thế hệ; ChatGPT là chatbot dựa trên LLM |
| Brown et al., *Language Models are Few-Shot Learners*, 2020. https://arxiv.org/abs/2005.14165 | Một model làm nhiều tác vụ, không cập nhật gradient |
| Chen et al., *Evaluating Large Language Models Trained on Code*, 2021. https://arxiv.org/abs/2107.03374 | Ví dụ sinh code (Codex, HumanEval) |
| OpenAI, *Introducing ChatGPT*, 30/11/2022. https://openai.com/blog/chatgpt/ | ChatGPT = GPT-3.5 + SFT + RLHF; giới hạn |
| Qwen Team, *Qwen2.5 Technical Report*, 2024. https://arxiv.org/abs/2412.15115 | 18 nghìn tỉ token pretraining |
| Model card Qwen2.5-1.5B, Qwen2.5-1.5B-Instruct. https://huggingface.co/Qwen | Số tham số; base không khuyến nghị để hội thoại |
| Hugging Face, *Chat templates*. https://huggingface.co/docs/transformers/chat_templating | `add_generation_prompt` |
| Slide môn Ch1 s7, s10, s12, s13, s15 | Transformer trong NLP; timeline; ứng dụng chatbot |

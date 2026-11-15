# Mục 8 — Cách thích nghi và bổ sung tri thức: Prompting, Fine-tuning, RAG (nội dung chi tiết, v1)

- Người phụ trách: Muội | Slide 15–16 | Thời lượng: khoảng 2 phút | Notebook: phần 8 (`src/rag.py`; **chờ cô xác nhận cách demo**)
- Đề của cô: "Phân biệt prompting, fine-tuning và Retrieval-Augmented Generation (RAG)". Minh hoạ: bảng so sánh dữ liệu cần có và trường hợp sử dụng.
- Ký hiệu nguồn:
  - **[Slide ChX sN]**: slide môn.
  - **[Bổ sung]**: nguồn ngoài, đã kiểm 2026-11-15 (§9).
  - **[Tự tính]**: đã chạy lại bằng Python (scikit-learn 1.9.1).

---

## 0. Mục này cần làm người nghe hiểu được gì
1. Có **ba cách** làm một LLM có sẵn phục vụ việc của mình: prompting, fine-tuning, RAG.
2. Mỗi cách **cần dữ liệu khác nhau**, tốn chi phí khác nhau, và hợp với trường hợp khác nhau (bảng của đề).
3. Điểm cốt lõi:
   - prompting và RAG **không đổi trọng số**;
   - fine-tuning **đổi trọng số**;
   - RAG đưa tri thức **từ bên ngoài** vào lúc trả lời.
4. Demo: hỏi thẳng so với hỏi kèm đoạn tài liệu tìm được.

---

## 1. Ba cách

### 1.1 Prompting
- Viết chỉ dẫn, có thể kèm ví dụ (zero-shot, few-shot), **không đổi trọng số**. Đây chính là in-context learning ở mục 7 (Brown et al. 2020).
- **Giới hạn:** model chỉ dùng được những gì **đã có trong trọng số** và trong prompt. Nó không biết thông tin mới hơn dữ liệu huấn luyện, hay tài liệu riêng của mình (ví dụ slide môn học).

### 1.2 Fine-tuning  [Slide Ch4 + Bổ sung]
- Huấn luyện tiếp model trên dữ liệu có nhãn của mình, **cập nhật trọng số** (Ch4: chiến lược fine-tuning; Lab 2 và Lab 3 của nhóm).
- SFT và alignment ở mục 5 cũng là fine-tuning.
- **Fine-tune toàn bộ** model lớn rất tốn: phải lưu gradient và trạng thái optimizer cho mọi tham số.
- **LoRA** (Hu et al. 2021): **giữ nguyên** trọng số pretrained, chỉ thêm các ma trận hạng thấp có thể huấn luyện vào từng lớp Transformer. So với fine-tune toàn bộ GPT-3 175B bằng Adam, LoRA giảm số tham số cần huấn luyện khoảng 10,000 lần và bộ nhớ GPU khoảng 3 lần, và **không làm chậm lúc chạy**. Thư viện: Hugging Face PEFT.
- **Hợp khi:** cần thay đổi **hành vi hoặc kỹ năng** ổn định (văn phong, định dạng, một tác vụ chuyên biệt) và có đủ dữ liệu nhãn.
- **Không hợp để** thêm tri thức hay thay đổi: muốn cập nhật thì phải train lại.
- Nhóm **không fine-tune** (góp ý của cô); chỉ trình bày ở mức khái niệm.

### 1.3 RAG (Retrieval-Augmented Generation)  [Bổ sung: Lewis et al. 2020]
- **Hai loại "bộ nhớ":**
  - **bộ nhớ tham số** (parametric): tri thức nằm trong trọng số model;
  - **bộ nhớ ngoài** (non-parametric): một kho văn bản model **tra cứu lúc trả lời**.
- **Ba thành phần** trong bài gốc:
  1. **Retriever** (bộ tìm): DPR, một bi-encoder tìm đoạn liên quan tới câu hỏi;
  2. **Chỉ mục tài liệu:** Wikipedia chia thành đoạn khoảng 100 từ, lưu dạng vector, tìm bằng láng giềng gần nhất;
  3. **Generator** (bộ sinh): BART-large, nhận câu hỏi cùng các đoạn tìm được rồi sinh câu trả lời.
- **Động cơ:** model chỉ dựa vào trọng số thì
  - khó cập nhật;
  - không chỉ ra được câu trả lời lấy từ đâu;
  - đôi khi sinh nội dung nghe hợp lý nhưng sai.
- **Cập nhật tri thức mà không train lại:** các tác giả thay chỉ mục Wikipedia cũ bằng bản mới hơn, và câu trả lời về lãnh đạo các nước thay đổi theo, không cần huấn luyện lại.

### 1.4 Luồng RAG (sơ đồ slide 16)
```
Tài liệu (slide môn học) ──► chia đoạn ──► chỉ mục (TF-IDF hoặc embedding)
                                                   │
Câu hỏi ──────────────────────────────────► tìm k đoạn liên quan nhất
                                                   │
                     prompt = chỉ dẫn + k đoạn + câu hỏi
                                                   ▼
                                   LLM (Qwen2.5-1.5B-Instruct) → câu trả lời
```

---

## 2. Bảng so sánh (minh hoạ của đề, slide 15)

| | Prompting | Fine-tuning | RAG |
|---|---|---|---|
| **Đổi trọng số?** | Không | **Có** | Không |
| **Dữ liệu cần có** | Chỉ dẫn + vài ví dụ (0–vài chục) | Nhiều mẫu có nhãn (hàng trăm đến hàng nghìn trở lên) | **Kho tài liệu** (không cần nhãn) + bộ tìm |
| **Chi phí** | Thấp nhất; không cần GPU để train | Cao: GPU, thời gian (giảm nhiều bằng LoRA) | Trung bình: dựng chỉ mục; prompt dài hơn |
| **Thêm / cập nhật tri thức** | Chỉ trong giới hạn prompt | Phải train lại | **Cập nhật kho là xong** |
| **Truy được nguồn** | Không | Không | **Có**: biết đoạn nào được dùng |
| **Giới hạn chính** | Bị giới hạn bởi tri thức sẵn có và độ dài ngữ cảnh | Tốn kém; có thể quên kỹ năng cũ | Chất lượng phụ thuộc bước tìm; tìm sai thì trả lời sai |
| **Trường hợp nên dùng** | Thử nhanh; tác vụ chung; ít dữ liệu | Đổi hành vi, văn phong, kỹ năng chuyên biệt; có dữ liệu nhãn | Hỏi đáp trên tài liệu riêng hoặc mới; cần trích nguồn |
| **Ví dụ trong bài** | Mục 7: zero-shot / 3-shot | Lab 3 (fine-tune BERT); SFT, DPO ở mục 5 | Demo mục 8: hỏi về slide môn học |

**Câu chốt:** "Muốn đổi **cách model làm** → fine-tuning. Muốn đổi **thứ model biết** → RAG. Muốn **thử nhanh** → prompting." Đây là quy tắc kinh nghiệm để trình bày, **không** phải luật tuyệt đối; ba cách có thể kết hợp.

---

## 3. Demo: không RAG so với có RAG (slide 16, notebook phần 8)  [chờ cô]

### 3.1 Thiết kế (theo kế hoạch nhóm)  [đề xuất — Muội chốt]
- **Kho tài liệu:** nội dung slide môn học (đã OCR trong `Research/slides-ocr/`), **không commit** vào repo.
- **Câu hỏi:** 15–20 câu có đáp án lấy từ slide, lưu `data/test_sets/rag_questions.csv` (cột: câu hỏi, đáp án, slide nguồn).
- **Hai cách:**
  - không RAG: hỏi thẳng Qwen;
  - có RAG: tìm k đoạn (ví dụ k = 3) bằng TF-IDF rồi ghép vào prompt.
- Cùng model, cùng greedy (`do_sample=False`), cùng `max_new_tokens`.
- **Đo:**
  - số câu đúng mỗi cách;
  - thêm một chỉ số: tỉ lệ câu mà **đoạn đúng nằm trong k đoạn tìm được** (kiểm riêng bước tìm).
- **Chọn câu hỏi:** nên có câu mà model khó tự biết, ví dụ chi tiết riêng của slide (số slide, ví dụ cụ thể trên slide), để thấy rõ tác dụng của RAG.

### 3.2 Chia đoạn và TF-IDF
- `TfidfVectorizer` biến tập văn bản thành ma trận TF-IDF. Mặc định chuẩn hoá L2, nên tích vô hướng giữa hai vector chính là cosine similarity (scikit-learn docs).
- Mặc định chỉ giữ token có **từ 2 ký tự chữ/số trở lên**; dấu câu bị bỏ.

Thử nhanh  [Tự tính, scikit-learn 1.9.1]:
- Với 3 đoạn tiếng Việt (causal attention / cross-entropy / top-p), câu hỏi "Top-p là gì?" tìm đúng đoạn top-p (similarity 0.309, hai đoạn kia 0).
- Từ tiếng Việt **một ký tự** như "ở" **bị bỏ** với cài đặt mặc định: "Mô hình ở đây có học máy" → `['mô', 'hình', 'đây', 'có', 'học', 'máy']`.
- TF-IDF tách theo **âm tiết**, nên từ ghép như "học máy" bị tách đôi. Dùng `ngram_range=(1, 2)` để giữ cả cặp âm tiết liền nhau ("học máy" thành một đặc trưng).

**Chia đoạn:**
- Theo slide: mỗi slide là một đoạn, dễ truy nguồn.
- Hoặc theo độ dài: khoảng 100 từ, giống bài gốc RAG.
- Ghi rõ cách chọn trong notebook.

### 3.3 Prompt có RAG  [đề xuất]
```
Dựa vào các đoạn tài liệu dưới đây, trả lời câu hỏi. Nếu tài liệu không có thông tin, trả lời "Không có trong tài liệu".

[Đoạn 1 — Ch2 slide 7] …
[Đoạn 2 — …] …
[Đoạn 3 — …] …

Câu hỏi: …
```
Câu "Nếu tài liệu không có…" giúp giảm việc model tự bịa (liên hệ mục 9).

### 3.4 Chấm đúng / sai  [cần chốt]
- Tự động bằng từ khoá đáp án, kèm người kiểm lại các câu khó.
- Ghi quy tắc chấm vào notebook **trước khi** chạy chính thức.
- Lưu `outputs/results/rag_results.csv`: câu hỏi, đáp án, đoạn tìm được, output không RAG, output có RAG, đúng/sai.

### 3.5 Cell "Phân tích" (khung)
> "Không RAG đúng … / … câu, có RAG đúng … / … . Bước tìm lấy được đoạn đúng ở … / … câu. Các câu RAG vẫn sai chủ yếu do … (tìm sai đoạn / đoạn đúng nhưng model đọc sai). Điều này khớp với Lewis et al.: đưa tri thức ngoài vào giúp trả lời chi tiết mà model không có trong trọng số, nhưng chất lượng phụ thuộc bước tìm."

Với 15–20 câu thì **không kết luận thống kê**, chỉ mô tả.

---

## 4. Nội dung slide

### Slide 15 — Ba cách thích nghi
- **Tiêu đề:** 8. Prompting, Fine-tuning và RAG
- **Giữa:** bảng §2, rút còn 5 hàng: đổi trọng số / dữ liệu cần có / chi phí / cập nhật tri thức / nên dùng khi.
- **Dưới:** câu chốt §2.
- **Chân:** Brown et al. 2020; Hu et al. 2021 (LoRA); Lewis et al. 2020 (RAG)

### Slide 16 — Demo RAG
- **Tiêu đề:** RAG: tra cứu tài liệu trước khi trả lời
- **Trái:** sơ đồ §1.4.
- **Phải:** bảng "Không RAG: … / … câu đúng · Có RAG: … / … câu đúng · Tìm đúng đoạn: … / …" và 1 ví dụ (câu hỏi, output không RAG, output có RAG, đoạn tìm được).
- **Chân:** TF-IDF (scikit-learn), k = … · Notebook phần 8

---

## 5. Lời thoại (khoảng 2 phút)

> **[Slide 15]** "Có ba cách làm một LLM có sẵn phục vụ việc của mình. Prompting là viết chỉ dẫn và ví dụ, như ở mục 7, không đổi trọng số, gần như không tốn gì; nhưng model chỉ biết những gì đã học. Fine-tuning là huấn luyện tiếp trên dữ liệu có nhãn, như Lab 3 hay giai đoạn SFT; nó đổi trọng số nên tốn GPU, dù LoRA giảm chi phí rất nhiều. Fine-tuning hợp để đổi hành vi, nhưng muốn cập nhật kiến thức lại phải train lại. RAG thì không đổi trọng số: trước khi trả lời, hệ thống tìm các đoạn tài liệu liên quan rồi đưa vào prompt. Muốn cập nhật kiến thức chỉ cần cập nhật kho tài liệu, và biết được câu trả lời dựa trên đoạn nào.
> **[Slide 16]** Nhóm thử RAG trên chính slide môn học: … câu hỏi, so hỏi thẳng với hỏi kèm 3 đoạn tìm bằng TF-IDF. Kết quả … [đọc số thật]. Ví dụ câu này … ."

---

## 6. Q&A dự kiến

| # | Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|---|
| 1 | Prompting khác fine-tuning cốt lõi ở đâu? | Prompting không đổi trọng số; fine-tuning đổi trọng số | Brown 2020; Ch4 |
| 2 | Vì sao không fine-tune để model biết slide môn học? | Tốn kém, cần dữ liệu dạng hỏi–đáp, và mỗi lần cập nhật phải train lại; RAG chỉ cần kho tài liệu | Lewis 2020 |
| 3 | LoRA là gì? | Giữ nguyên trọng số gốc, chỉ học thêm ma trận hạng thấp; giảm mạnh số tham số cần học và bộ nhớ, không chậm lúc chạy | Hu 2021 |
| 4 | RAG có hết hallucination không? | Không. Tìm sai đoạn hoặc model đọc sai thì vẫn sai; giảm bằng cách yêu cầu "không có thì nói không có" và kiểm bước tìm | Lewis 2020; §3.5 |
| 5 | Vì sao dùng TF-IDF mà không dùng embedding? | Đơn giản, chạy CPU, giải thích được (khớp từ). Nhược: không hiểu từ đồng nghĩa; embedding là bản nâng cấp | scikit-learn |
| 6 | TF-IDF với tiếng Việt có vấn đề gì? | Tách theo âm tiết nên từ ghép bị tách; mặc định bỏ từ 1 ký tự. Dùng `ngram_range=(1, 2)` | Tự tính §3.2 |
| 7 | RAG có giới hạn ngữ cảnh không? | Có: k đoạn phải vừa ngữ cảnh; nhiều đoạn quá thì tốn token, model dễ bỏ sót thông tin ở giữa | Mục 9 |
| 8 | Ba cách có dùng chung được không? | Được: ví dụ model đã fine-tune (Instruct) + prompt tốt + RAG | §2 |

---

## 7. Tài liệu tham khảo cho mục 8

| Tài liệu | Dùng cho |
|---|---|
| Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020. https://arxiv.org/abs/2005.11401 | Định nghĩa RAG; bộ nhớ tham số và bộ nhớ ngoài; cập nhật chỉ mục |
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. https://arxiv.org/abs/2106.09685 | Fine-tuning tiết kiệm |
| Brown et al., *Language Models are Few-Shot Learners*, 2020. https://arxiv.org/abs/2005.14165 | Prompting / in-context learning |
| Hugging Face PEFT. https://huggingface.co/docs/peft | Thư viện LoRA |
| scikit-learn, `TfidfVectorizer`. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html | TF-IDF, chuẩn hoá L2, `token_pattern` |
| Zhao et al., *A Survey of Large Language Models*, 2023. https://arxiv.org/abs/2303.18223 | Tổng quan cách thích nghi LLM |
| Slide môn Ch4; Lab 3 của nhóm | Fine-tuning |

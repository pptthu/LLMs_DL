# Báo cáo — Phần 7 và Phần 10 (Toàn) (v1)

> Bản nháp nội dung báo cáo, viết theo văn phong báo cáo khoa học. Chỗ ghi **[điền sau]** chỉ được điền bằng số liệu từ `outputs/results/` của lần chạy chính thức (Save & Run All trên Kaggle). Trích dẫn đánh số [n] theo danh sách tài liệu ở cuối. Nội dung mục 7 và mục 10 không có trong slide môn học; toàn bộ là kiến thức bổ sung từ các tài liệu được trích.

---

## 7. Khả năng của mô hình ngôn ngữ lớn

### 7.1 Học trong ngữ cảnh (in-context learning)

Một đặc điểm nổi bật của mô hình ngôn ngữ lớn là khả năng thực hiện tác vụ mới chỉ dựa trên nội dung của prompt, không cần huấn luyện lại. Brown và cộng sự [1] cho thấy GPT-3 giải được nhiều tác vụ ngôn ngữ khi tác vụ và các ví dụ minh hoạ được mô tả hoàn toàn bằng văn bản, không cập nhật gradient và không fine-tune. Cách học này được gọi là **học trong ngữ cảnh** (in-context learning, ICL). Zhao và cộng sự [6] xếp ICL vào nhóm khả năng "nổi lên" (emergent abilities) của LLM: các khả năng thể hiện rõ ở mô hình quy mô lớn nhưng không rõ ở các mô hình ngôn ngữ nhỏ trước đó.

Theo số lượng ví dụ đặt trong prompt, [1] phân biệt ba kiểu:
- **Zero-shot:** prompt chỉ có mô tả tác vụ.
- **One-shot:** thêm một ví dụ có đáp án.
- **Few-shot:** thêm k ví dụ có đáp án, với k phụ thuộc vào độ dài ngữ cảnh mô hình cho phép.

Thí nghiệm của nhóm dùng zero-shot và 3-shot.

Cần nhấn mạnh rằng chữ "learning" ở đây không có nghĩa là mô hình thay đổi. Trọng số giữ nguyên trước và sau khi chạy; khi prompt kết thúc, các ví dụ không còn tác dụng. Về bản chất, mô hình vẫn chỉ thực hiện dự đoán token tiếp theo (Phần 4). Các ví dụ cung cấp ngữ cảnh để mô hình "viết tiếp" theo đúng khuôn mẫu.

Vậy ví dụ giúp mô hình bằng cách nào? Min và cộng sự [5] cho kết quả đáng chú ý: thay nhãn của các ví dụ bằng nhãn ngẫu nhiên chỉ làm hiệu năng giảm nhẹ. Theo các tác giả, lợi ích của ví dụ đến chủ yếu từ việc cho mô hình thấy:
- tập nhãn có thể chọn;
- phân phối của câu đầu vào;
- định dạng của câu trả lời;

chứ không nằm nhiều ở việc ghép đúng câu với nhãn.

Bên cạnh ưu điểm, ICL có một điểm yếu quan trọng. Zhao và cộng sự [4] chỉ ra rằng độ chính xác few-shot dao động rất mạnh theo định dạng prompt, việc chọn ví dụ nào và thứ tự các ví dụ. Họ mô tả một số thiên lệch của mô hình:
- **majority label bias:** thiên về nhãn xuất hiện nhiều trong các ví dụ;
- **recency bias:** thiên về nhãn của ví dụ đứng gần cuối prompt;
- **common token bias:** thiên về các token phổ biến.

Đây là lý do thí nghiệm ở Phần 10 dùng ba bộ ví dụ khác nhau thay vì một bộ duy nhất.

### 7.2 Làm theo chỉ dẫn (instruction following)

Làm theo chỉ dẫn là khả năng hiểu và thực hiện đúng yêu cầu viết bằng ngôn ngữ tự nhiên, ví dụ "chỉ trả lời một từ: positive hoặc negative". Khả năng này không tự xuất hiện đầy đủ sau pretraining mà được tăng cường ở giai đoạn hậu huấn luyện:
- Wei và cộng sự [2] cho thấy fine-tune mô hình trên tập hợp nhiều tác vụ được diễn đạt dưới dạng chỉ dẫn (instruction tuning) giúp hiệu năng zero-shot trên tác vụ chưa gặp tăng rõ rệt.
- Ouyang và cộng sự [3] huấn luyện InstructGPT bằng hai bước: supervised fine-tuning (SFT) trên câu trả lời mẫu của con người, sau đó học tăng cường từ phản hồi của con người (RLHF), để mô hình bám sát ý định người dùng hơn.

Mô hình nhóm sử dụng, `Qwen2.5-1.5B-Instruct`, là phiên bản đã qua cả pretraining và post-training [11]. Model card của bản base `Qwen2.5-1.5B` ghi rõ không khuyến nghị dùng bản base cho hội thoại [11]; sự khác biệt này được minh hoạ ở Phần 5. Với bản Instruct, đầu vào cần được định dạng theo **chat template** của tokenizer. Hàm `apply_chat_template` với `add_generation_prompt=True` chèn phần mở đầu lượt trả lời của trợ lý, báo cho mô hình biết đã đến lượt nó sinh câu trả lời [12].

### 7.3 Xử lý các tác vụ ngôn ngữ

Vì mọi tác vụ đều có thể viết dưới dạng "văn bản vào, văn bản ra", một LLM duy nhất xử lý được nhiều tác vụ (hỏi đáp, tóm tắt, dịch, sinh mã) chỉ bằng cách thay đổi prompt, như đã minh hoạ ở Phần 1. Phân loại cảm xúc cũng được đưa về dạng này: mô hình sinh ra một từ nhãn, sau đó từ này được phân tích (parse) thành nhãn phân loại.

Cách tiếp cận này khác với hướng fine-tune mô hình phân loại trong phần thực hành Hugging Face của môn học:

| | Fine-tune mô hình phân loại | In-context learning (Phần 7) |
|---|---|---|
| Loại mô hình | Có đầu phân loại (`AutoModelForSequenceClassification`) | Mô hình sinh văn bản (`AutoModelForCausalLM`) |
| Cập nhật trọng số | Có, bằng `Trainer` | Không |
| Dữ liệu có nhãn cần dùng | Nhiều mẫu | 0 (zero-shot) hoặc vài mẫu (few-shot) |

### 7.4 Minh hoạ

Để minh hoạ, nhóm chạy cùng một tác vụ phân loại cảm xúc với hai prompt: zero-shot và 3-shot. Hai prompt dùng chung phần chỉ dẫn và phần câu cần phân loại; khác nhau duy nhất ở ba khối ví dụ:

```
Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời một từ: positive hoặc negative.

Câu: "<ví dụ 1>"
Cảm xúc: <nhãn 1>
Câu: "<ví dụ 2>"
Cảm xúc: <nhãn 2>
Câu: "<ví dụ 3>"
Cảm xúc: <nhãn 3>

Câu: "<câu cần phân loại>"
Cảm xúc:
```

Bảng 7.1 trình bày output của hai cách trên một số câu **[điền sau]**. Các câu minh hoạ chưa đủ để kết luận cách nào tốt hơn; phép so sánh có kiểm soát trên 500 câu được trình bày ở Phần 10.

**Bảng 7.1.** Ví dụ output zero-shot và 3-shot **[điền sau]**

| Câu | Nhãn thật | Output zero-shot | Output 3-shot |
|---|---|---|---|
| … | … | … | … |

---

## 10. Thực nghiệm có kiểm soát và kết luận

### 10.1 Câu hỏi nghiên cứu

Phần này kiểm tra câu hỏi: trên tác vụ phân loại cảm xúc hai nhãn của bộ dữ liệu UIT-VSFC, với mô hình `Qwen2.5-1.5B-Instruct`, việc thêm ba ví dụ vào prompt (3-shot) có làm độ chính xác khác với zero-shot một cách có ý nghĩa thống kê hay không.
- **Giả thuyết không (H0):** hai cách có tỉ lệ dự đoán sai như nhau.
- **Giả thuyết đối (H1):** tỉ lệ sai khác nhau.

### 10.2 Dữ liệu

Nhóm sử dụng UIT-VSFC (Vietnamese Students' Feedback Corpus) của Nguyen và cộng sự [7], gồm các câu phản hồi của sinh viên Việt Nam được gán nhãn cảm xúc và chủ đề. Theo các tác giả, độ đồng thuận giữa những người gán nhãn cảm xúc đạt trên 91% [7]. Bộ dữ liệu có ba nhãn cảm xúc (negative, neutral, positive) và được chia sẵn thành train, validation, test [8]. Phân bố nhãn của hai tập được dùng trình bày ở Bảng 10.1.

**Bảng 10.1.** Phân bố nhãn trong UIT-VSFC [8]

| Tập | Negative | Neutral | Positive | Tổng |
|---|---|---|---|---|
| Train | 5,325 | 458 | 5,643 | 11,426 |
| Test | 1,409 | 167 | 1,590 | 3,166 |

Vì thí nghiệm là phân loại hai nhãn, nhóm loại các câu neutral. Neutral chỉ chiếm khoảng 5.3% tập test (167/3,166).

Tập đánh giá gồm **500 câu**, lấy ngẫu nhiên từ tập test với seed 42: 250 câu negative và 250 câu positive. Thiết kế cân bằng có hai tác dụng:
- mốc đoán ngẫu nhiên, hoặc luôn đoán một nhãn, là khoảng 50%, làm điểm so sánh rõ ràng;
- độ chính xác không bị lệch do một nhãn chiếm đa số.

Các ví dụ few-shot lấy từ tập **train**, nên tách biệt với tập đánh giá. Nhóm vẫn kiểm tra không có câu ví dụ nào trùng nội dung với câu đánh giá, để tránh rò rỉ dữ liệu. Danh sách câu đánh giá và các bộ ví dụ được lưu cố định (`data/splits/test_500_ids.json`, `data/splits/fewshot_pools.json`) để tái lập.

### 10.3 Thiết kế thí nghiệm

Thí nghiệm chỉ thay đổi **một biến**: có ví dụ trong prompt hay không, và dùng bộ ví dụ nào. Mọi yếu tố khác được giữ cố định (Bảng 10.2).

**Bảng 10.2.** Các yếu tố cố định

| Yếu tố | Giá trị |
|---|---|
| Mô hình | `Qwen/Qwen2.5-1.5B-Instruct` [11]; kiểu dữ liệu **[điền sau: fp16 hoặc fp32]** |
| Phương pháp sinh | Greedy decoding (`do_sample=False`), `max_new_tokens` = **[điền sau]** |
| Prompt | Template ở Mục 7.4, định dạng bằng chat template của tokenizer |
| Tập đánh giá | 500 câu cố định (Mục 10.2) |
| Seed | 42 |
| Môi trường | Kaggle, GPU NVIDIA T4; phiên bản thư viện **[điền sau, từ `env.json`]** |

Có hai chi tiết kỹ thuật cần lưu ý.

Thứ nhất, tệp `generation_config.json` của `Qwen2.5-1.5B-Instruct` mặc định bật lấy mẫu ngẫu nhiên: `do_sample` là true, `temperature` 0.7, `top_p` 0.8, `top_k` 20 [11]. Nhóm vì vậy truyền tường minh `do_sample=False` để đảm bảo greedy decoding. Nhờ đó kết quả tất định: cùng đầu vào luôn cho cùng đầu ra.

Thứ hai, do few-shot nhạy với lựa chọn và thứ tự ví dụ [4], nhóm dùng ba bộ ví dụ (Bảng 10.3). Ba ví dụ không chia đều được cho hai nhãn, nên bộ 1 và bộ 2 được thiết kế đối xứng: thiên lệch về nhãn đa số và nhãn của ví dụ cuối không chỉ nghiêng về một phía.

**Bảng 10.3.** Ba bộ ví dụ few-shot

| Bộ | Thứ tự nhãn | Mục đích |
|---|---|---|
| 1 | positive, negative, positive | Dư positive, ví dụ cuối là positive |
| 2 | negative, positive, negative | Đối xứng với bộ 1 |
| 3 | Ngẫu nhiên theo seed | Thêm một lần lấy mẫu độc lập |

Tổng cộng có bốn cấu hình: zero-shot và 3-shot với từng bộ 1, 2, 3. Mỗi cấu hình chạy một lần trên 500 câu, tức 2,000 lần sinh.

**Phân tích output thành nhãn.** Output được chuyển về chữ thường và bỏ dấu câu:
- có từ `positive` mà không có `negative` → positive;
- có `negative` mà không có `positive` → negative;
- các trường hợp còn lại (có cả hai, hoặc không có nhãn nào) → **không phân tích được**.

Mọi output thô được lưu vào tệp CSV (gồm câu, nhãn thật, prompt, output, nhãn dự đoán) để kiểm tra lại.

### 10.4 Phương pháp đánh giá

**Độ chính xác (accuracy)** bằng số câu dự đoán đúng chia cho 500. Câu không phân tích được tính là dự đoán sai; tỉ lệ các câu này được báo cáo riêng. Nếu loại các câu này trước khi tính, độ chính xác sẽ bị thổi phồng.

**Ma trận nhầm lẫn** (thêm một cột cho câu không phân tích được) cho biết mô hình thường nhầm theo chiều nào.

**Sai số do lấy mẫu.** Với n = 500, nửa độ rộng khoảng tin cậy 95% của độ chính xác (xấp xỉ chuẩn, 1.96·√(p(1−p)/n)) là:
- khoảng 4.4 điểm phần trăm khi p = 50%;
- khoảng 3.5 điểm khi p = 80%.

Do đó, chênh lệch vài điểm phần trăm giữa hai cấu hình chưa đủ để kết luận nếu chỉ nhìn độ chính xác.

**Kiểm định McNemar.** Hai cấu hình được đánh giá trên **cùng** 500 câu nên kết quả là dữ liệu theo cặp. Kiểm định McNemar chỉ xét các câu mà hai cấu hình cho kết quả khác nhau:
- b là số câu chỉ zero-shot đúng;
- c là số câu chỉ 3-shot đúng.

Dưới H0 thì b và c xấp xỉ nhau.
- **Vì sao chọn McNemar:** Dietterich [9] so sánh năm kiểm định thống kê và kết luận rằng với thuật toán chỉ có thể chạy một lần, McNemar là kiểm định duy nhất có sai lầm loại I chấp nhận được. Đây đúng là trường hợp của nhóm.
- **Cách tính:** dùng `statsmodels.stats.contingency_tables.mcnemar` với `exact=True` (phân phối nhị thức chính xác), phù hợp khi b + c nhỏ [10].
- **Ba lần kiểm định:** mỗi bộ few-shot so với zero-shot. Nhóm **[điền sau: áp dụng / không áp dụng]** hiệu chỉnh Bonferroni (ngưỡng 0.05/3 ≈ 0.017).

### 10.5 Kết quả

**Bảng 10.4.** Kết quả trên 500 câu **[điền sau]**

| Cấu hình | Accuracy | Không phân tích được | b | c | p-value (so với zero-shot) |
|---|---|---|---|---|---|
| Zero-shot | … | … | — | — | — |
| 3-shot, bộ 1 | … | … | … | … | … |
| 3-shot, bộ 2 | … | … | … | … | … |
| 3-shot, bộ 3 | … | … | … | … | … |
| 3-shot, trung bình (thấp nhất – cao nhất) | … (… – …) | … | | | |

*Nguồn số liệu: `outputs/results/icl_metrics.json`, chạy ngày **[điền sau]**.*

**Hình 10.1.** Ma trận nhầm lẫn của zero-shot và 3-shot **[điền sau: `outputs/figures/confusion_matrix_*.png`]**.

Đoạn nhận xét kết quả **[viết sau khi có số]**, theo khung:
1. So sánh độ chính xác zero-shot với từng bộ 3-shot, kèm p-value.
2. Mức dao động giữa ba bộ ví dụ.
3. Chiều nhầm lẫn chính đọc từ ma trận nhầm lẫn.
4. Tỉ lệ câu không phân tích được ở từng cấu hình.

### 10.6 Phân tích lỗi

Nhóm đọc toàn bộ các câu bị dự đoán sai ở cấu hình zero-shot và cấu hình 3-shot **[điền sau: bộ nào]**, rồi xếp mỗi câu vào một nhóm lỗi. Các nhóm dưới đây là dự kiến; nhóm lỗi chính thức xác định sau khi đọc dữ liệu thật.
- câu vừa khen vừa chê;
- câu mỉa mai hoặc nói ngược;
- câu quá ngắn, thiếu ngữ cảnh;
- câu viết tắt hoặc sai chính tả;
- câu có thể bị gán nhãn sai trong dữ liệu gốc;
- lỗi định dạng output.

Nhóm "có thể gán nhãn sai" được nêu thận trọng: độ đồng thuận giữa người gán nhãn của bộ dữ liệu là trên 91% [7], nghĩa là vẫn tồn tại các câu mà chính người gán nhãn không thống nhất.

**Bảng 10.5.** Nhóm lỗi **[điền sau]**

| Nhóm lỗi | Số câu (zero-shot) | Số câu (3-shot) | Ví dụ (câu thật từ tệp kết quả) |
|---|---|---|---|
| … | … | … | … |

Chi tiết từng câu sai được lưu tại `docs/experiments/error_analysis_v1.md`.

### 10.7 Tổng hợp kết quả các phần

**Bảng 10.6.** Tổng hợp kết quả thực nghiệm toàn bài **[điền sau, đọc từ `outputs/results/summary_table.csv`]**

| Phần | Thực nghiệm | Kết quả chính |
|---|---|---|
| 3 | Số token câu tiếng Việt và tiếng Anh, tokenizer GPT-2 và Qwen | … |
| 4 | Cross-entropy loss trên câu ví dụ | … |
| 5 | Base và Instruct trên cùng prompt | … |
| 6 | Năm cấu hình sinh văn bản | … |
| 7, 10 | Zero-shot và 3-shot, 500 câu | … |
| 8 | Không RAG và có RAG | … |
| 9 | Hallucination, thiên lệch, giới hạn ngữ cảnh | … |

### 10.8 Thảo luận và giới hạn

Thí nghiệm có một số giới hạn cần nêu rõ khi diễn giải kết quả:
- Chỉ dùng một mô hình cỡ nhỏ (1.54 tỉ tham số [11]), một bộ dữ liệu và một tác vụ. Kết quả không suy rộng cho LLM nói chung.
- Mỗi cấu hình chạy một lần với greedy decoding. Kết quả tất định nhưng không phản ánh biến động khi dùng sampling.
- Chỉ thử k = 3 ví dụ và một prompt template; prompt viết bằng tiếng Việt nhưng nhãn đầu ra là tiếng Anh.
- Prompt template được chốt trước khi chạy trên tập đánh giá; nếu cần điều chỉnh thì thử trên tập validation. Cách làm này tránh chọn prompt dựa trên tập test.

### 10.9 Kết luận

Phần kết luận **[viết sau khi có số]**, gồm 3–4 ý, mỗi ý gắn với một con số hoặc ví dụ cụ thể:
1. Hiệu quả của 3-shot so với zero-shot, kèm p-value McNemar.
2. Mức độ phụ thuộc vào bộ ví dụ.
3. Các nhóm lỗi chính của mô hình.
4. Nhận xét chung về khả năng và hạn chế của LLM, rút ra từ các phần thực nghiệm khác của bài.

---

## Tài liệu tham khảo

[1] T. B. Brown et al., "Language Models are Few-Shot Learners," arXiv:2005.14165, 2020.

[2] J. Wei et al., "Finetuned Language Models Are Zero-Shot Learners," arXiv:2109.01652, 2021.

[3] L. Ouyang et al., "Training language models to follow instructions with human feedback," arXiv:2203.02155, 2022.

[4] T. Z. Zhao, E. Wallace, S. Feng, D. Klein, S. Singh, "Calibrate Before Use: Improving Few-Shot Performance of Language Models," arXiv:2102.09690, 2021.

[5] S. Min et al., "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?," arXiv:2202.12837, 2022.

[6] W. X. Zhao et al., "A Survey of Large Language Models," arXiv:2303.18223, 2023.

[7] K. V. Nguyen, V. D. Nguyen, P. X. V. Nguyen, T. T. H. Truong, N. L.-T. Nguyen, "UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis," in *Proc. 10th International Conference on Knowledge and Systems Engineering (KSE)*, 2018, pp. 19–24, doi: 10.1109/KSE.2018.8573337.

[8] UIT NLP Group, "vietnamese_students_feedback," Hugging Face Datasets. https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback (truy cập 14/11/2026).

[9] T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms," *Neural Computation*, vol. 10, no. 7, pp. 1895–1923, 1998, doi: 10.1162/089976698300017197.

[10] statsmodels, "statsmodels.stats.contingency_tables.mcnemar." https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html (truy cập 14/11/2026).

[11] Qwen Team, "Qwen2.5-1.5B-Instruct" và "Qwen2.5-1.5B," Hugging Face model cards. https://huggingface.co/Qwen (truy cập 14/11/2026).

[12] Hugging Face, "Chat templates," Transformers documentation. https://huggingface.co/docs/transformers/chat_templating (truy cập 14/11/2026).

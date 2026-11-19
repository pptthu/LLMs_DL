# Phân tích thực nghiệm — các tác vụ LLM, kiến trúc, tokenization và RAG (Muội, v1)

- **Created**: 2026-11-19 (Muội; tổng hợp từ kết quả thực nghiệm chính thức)
- **Phạm vi phụ trách**: Part 1 (các tác vụ cơ bản), Part 2 (kiến trúc và dự đoán token), Part 3 (tokenization), Part 8 (RAG).
- **Lần chạy**: `20261119_025437_normal` (Kaggle, GPU Tesla T4). Kết quả lấy từ `outputs/results/`, hình từ `outputs/figures/` và log tại `outputs/logs/20261119_025437_normal/`.
- **Notebook**: `notebook/llm_presentation.ipynb`; cần đọc các cell tương ứng Part 1, 2, 3, 8 cùng output ngay bên dưới để giải thích được quy trình và kết quả.

## 1. Thiết lập chung

| Thông tin | Giá trị |
|---|---|
| Nền tảng | Kaggle Notebook, GPU Tesla T4 |
| Model chính | `Qwen/Qwen2.5-1.5B-Instruct` |
| Số tham số | 1,543,714,304 (~1.54B) |
| Kiểu trọng số | float16 |
| Kiến trúc | 28 decoder blocks; hidden size 1536; 12 attention heads và 2 KV heads; GQA; head dimension 128; FFN intermediate size 8960 |
| Vocabulary | 151,936 |
| Context length | 32,768 tokens |
| Tokenizer so sánh | Qwen tokenizer và GPT-2 tokenizer (Part 3) |
| Dữ liệu | Câu tiếng Việt/Anh minh họa; UIT-ViQuAD 2.0 validation cho Part 8 |
| Cấu hình sinh | Greedy (`do_sample=False`) trong các thử nghiệm cần cố định đầu ra; các tham số khác được ghi riêng theo từng phần |

Các kết quả trong tài liệu này là kết quả của một lần chạy chính thức, không dùng kết quả smoke test (`SMOKE=True`).

## 2. Part 1 — Các tác vụ cơ bản của LLM

### 2.1. Mục tiêu và cách thực hiện

Kiểm tra nhanh khả năng thực hiện bốn tác vụ ngôn ngữ phổ biến của model Instruct: hỏi đáp, tóm tắt, dịch và sinh mã. Đây là minh họa chức năng, không phải benchmark toàn diện.

| Tác vụ | Đầu vào | Tiêu chí kiểm tra |
|---|---|---|
| Question answering | Hỏi thủ đô Nhật Bản, yêu cầu trả lời một câu | Trả lời đúng và đúng định dạng ngắn |
| Summarization | Một đoạn văn cần tóm tắt | Giữ ý chính, không tự thêm thông tin |
| Translation | Câu tiếng Việt cần dịch sang tiếng Anh | Giữ đúng ý nghĩa câu gốc |
| Code generation | Yêu cầu viết hàm kiểm tra số chẵn | Code chạy được với các trường hợp thử |

### 2.2. Kết quả

| Tác vụ | Input tokens | Output tokens | Thời gian | Kết quả |
|---|---:|---:|---:|---|
| Hỏi đáp | 44 | 10 | 0.44 giây | Đạt |
| Tóm tắt | 165 | 104 | 4.17 giây | Đạt, nhưng mức độ rút gọn chưa cao |
| Dịch | 43 | 8 | 0.32 giây | Đạt |
| Sinh code | 49 | 216 | 8.49 giây | Đạt; hàm vượt qua các test |

- Cả 4/4 tác vụ được đánh giá đạt; không có đầu vào nào bị truncation.
- Hỏi đáp: model trả lời “Thủ đô của Nhật Bản là Tokyo.”
- Tóm tắt: giữ được các ý chính và không thêm dữ kiện mới, nhưng đầu ra vẫn gần với việc nối lại các câu trong văn bản gốc nên mức độ cô đọng còn hạn chế.
- Dịch: “Today the weather is very beautiful.”, truyền đạt đúng ý câu tiếng Việt.
- Sinh code: tạo hàm `is_even(n)` dùng phép chia lấy dư; kết quả kiểm thử với `[0, 1, 2, -3]` lần lượt là `[True, False, True, False]`.

### 2.3. Nhận xét

1. Model Instruct có thể xử lý nhiều dạng yêu cầu khác nhau chỉ bằng prompt, không cần huấn luyện lại trong thí nghiệm này.
2. Sinh code có thời gian thực thi dài nhất trong bốn tác vụ (8.49 giây), đồng thời tạo nhiều token nhất (216 token), do cần trả về code kèm ví dụ/giải thích.
3. Tóm tắt là trường hợp cần xem xét kỹ hơn: câu trả lời đúng nội dung nhưng chưa rút gọn mạnh. “Không thêm thông tin” và “tóm tắt cô đọng” là hai tiêu chí khác nhau.
4. Kết quả 4/4 chỉ phản ánh bốn prompt minh họa, không đủ để kết luận model luôn làm tốt các tác vụ này trên dữ liệu đa dạng.

## 3. Part 2 — Kiến trúc mô hình và dự đoán token tiếp theo

### 3.1. Mục tiêu và thông số kiến trúc

Mục tiêu là quan sát một phần kiến trúc decoder-only Transformer của Qwen2.5-1.5B-Instruct và xem model phân phối xác suất cho token tiếp theo như thế nào.

| Thành phần | Giá trị |
|---|---:|
| Decoder blocks | 28 |
| Hidden size | 1,536 |
| Query attention heads | 12 |
| Key/Value heads | 2 |
| Attention | Grouped-Query Attention (GQA) |
| Head dimension | 128 |
| FFN intermediate size | 8,960 |
| Vocabulary size | 151,936 |
| Tổng tham số | 1,543,714,304 |

Phân bố tham số theo nhóm:

| Nhóm tham số | Số lượng |
|---|---:|
| Embedding | 233,373,696 |
| Attention | 154,198,016 |
| Feed-forward network (FFN) | 1,156,055,040 |
| Normalization | 87,552 |
| **Tổng** | **1,543,714,304** |

FFN chiếm phần lớn tham số, khoảng 75% tổng số tham số của model.

### 3.2. Causal mask và attention

- Causal mask ngăn vị trí hiện tại nhìn thấy các token ở tương lai. Trong ma trận mask được kiểm tra, tổng trọng số ở vùng tam giác trên bị che là `0.000000`.
- Tổng trọng số attention trên mỗi hàng bằng `1.0` (sai số làm tròn số thực có thể xuất hiện khi hiển thị).
- Cơ chế này bảo đảm dự đoán token kế tiếp chỉ dựa trên ngữ cảnh hiện tại và các token đứng trước, không sử dụng token tương lai.

### 3.3. Phân phối top-5 token tiếp theo

Với ngữ cảnh đầu vào **“Hôm nay trời rất”** (6 input tokens), model đưa ra phân phối xác suất sau:

| Token tiếp theo | Xác suất |
|---|---:|
| `đẹp` | 78.79% |
| `nắng` | 6.32% |
| `lạnh` | 4.55% |
| `nóng` | 2.89% |
| Token khoảng trắng | 0.71% |

Tổng xác suất của top-5 token xấp xỉ 93.32%. Token `đẹp` có xác suất cao nhất, phù hợp với ngữ cảnh “Hôm nay trời rất đẹp”.

### 3.4. Nhận xét

1. Kết quả cho thấy model không chọn token tiếp theo theo kiểu chắc chắn tuyệt đối; model tạo một phân phối xác suất trên vocabulary rồi chọn token theo chiến lược giải mã.
2. FFN là nhóm có số tham số lớn nhất trong cấu hình được khảo sát.
3. Causal mask bảo đảm tính nhân quả của quá trình sinh tự hồi quy; tổng attention mỗi hàng bằng 1 cho thấy trọng số attention được chuẩn hóa.
4. Top-5 chỉ là một ví dụ với một ngữ cảnh cụ thể. Không thể dùng một phân phối đơn lẻ để kết luận về chất lượng sinh văn bản nói chung.

## 4. Part 3 — Tokenization

### 4.1. Mục tiêu và cách so sánh

So sánh cách GPT-2 tokenizer và Qwen tokenizer chia câu tiếng Việt/tiếng Anh thành token. Đồng thời kiểm tra quá trình chuyển token thành ID, giải mã ngược và kích thước embedding đầu vào model.

Các câu thử nghiệm:
- Tiếng Việt: “Hôm nay trời rất đẹp.”
- Tiếng Anh: “The weather is very nice today.”

### 4.2. Kết quả so sánh số token

| Câu | Số ký tự | Số byte UTF-8 | GPT-2 tokens | Qwen tokens |
|---|---:|---:|---:|---:|
| Tiếng Việt | 21 | 29 | 22 | 8 |
| Tiếng Anh | 31 | 31 | 7 | 7 |

Thông tin vocabulary:

| Thành phần | Kích thước |
|---|---:|
| GPT-2 vocabulary | 50,257 |
| Qwen `tokenizer.vocab_size` | 151,643 |
| Qwen `len(tokenizer)` | 151,665 |
| Model `config.vocab_size` | 151,936 |
| Embedding matrix | `(151936, 1536)` |

Kiểm tra embedding với câu tiếng Việt:
- Input token IDs có shape `(1, 8)`.
- Sau embedding, tensor có shape `(1, 8, 1536)`.
- Input được giải mã ngược về câu ban đầu thành công trong 4/4 lượt kiểm tra round-trip của notebook.
- Embedding của input và output được chia sẻ trọng số (`tied embeddings = True`).

### 4.3. Nhận xét

1. Trong câu tiếng Việt được thử nghiệm, Qwen dùng 8 token trong khi GPT-2 dùng 22 token. Với câu tiếng Anh mẫu, cả hai đều dùng 7 token.
2. Kết quả minh họa cho thấy tokenizer có ảnh hưởng đáng kể đến số token, đặc biệt với văn bản tiếng Việt có dấu.
3. GPT-2 sử dụng byte-level BPE; khi giải mã riêng lẻ một token byte, có thể xuất hiện ký tự thay thế do byte chưa tạo thành chuỗi UTF-8 hoàn chỉnh. Khi ghép đủ token, câu vẫn giải mã đúng.
4. Không thể kết luận Qwen luôn tiết kiệm token hơn GPT-2 cho mọi văn bản tiếng Việt chỉ từ hai câu thử nghiệm này.
5. `tokenizer.vocab_size`, `len(tokenizer)` và `model.config.vocab_size` khác nhau vì chúng phản ánh các cách đếm khác nhau, bao gồm token bổ sung và kích thước vocabulary mà embedding model hỗ trợ.

## 5. Part 8 — Retrieval-Augmented Generation (RAG)

### 5.1. Mục tiêu và dữ liệu

Mục tiêu là so sánh khả năng trả lời câu hỏi khi model không có ngữ cảnh truy xuất (No RAG) và khi được cung cấp các đoạn văn liên quan tìm bằng TF-IDF (RAG).

| Thông tin | Giá trị |
|---|---:|
| Dataset | UIT-ViQuAD 2.0 validation |
| Số câu hỏi validation | 3,814 |
| Số context | 557 |
| Số chủ đề | 19 |
| Số câu hỏi không trả lời được | 1,161 |
| Mẫu thực nghiệm | 18 câu: 15 answerable + 3 impossible |
| Random seed | 42 |
| Retrieval | TF-IDF, `ngram_range=(1,2)`, cosine similarity qua `linear_kernel` |
| Số context truy xuất | Top-3 |
| Recall@3 | 0.722 (13/18) |

Hai cấu hình dùng cùng chỉ dẫn: nếu thông tin không có trong nguồn thì trả lời **“Không có thông tin”**. Khi sinh, dùng `do_sample=False`, `repetition_penalty=1.0`, `max_new_tokens=48`.

### 5.2. Kết quả tổng quan

| Loại câu hỏi | Cấu hình | Đúng | Sai | Từ chối |
|---|---|---:|---:|---:|
| Answerable (15 câu) | No RAG | 0 | 1 | 14 |
| Answerable (15 câu) | RAG | 10 | 3 | 2 |
| Impossible (3 câu) | No RAG | — | 0 | 3 |
| Impossible (3 câu) | RAG | — | 2 | 1 |
| **Tổng (18 câu)** | **No RAG** | **0** | **1** | **17** |
| **Tổng (18 câu)** | **RAG** | **10** | **5** | **3** |

Thời gian sinh tổng cộng được ghi nhận khoảng 24.1 giây. Một số đầu ra RAG bị cắt do giới hạn `max_new_tokens=48` (`truncated=True`).

### 5.3. Ví dụ tiêu biểu

| Câu hỏi/chủ đề | No RAG | RAG | Nhận xét |
|---|---|---|---|
| Luật của Hoàng đế La Mã Thần thánh | Trả lời sai | Trả lời đúng: cần thông qua Hội đồng | Context truy xuất giúp cung cấp thông tin cụ thể |
| GDP nông nghiệp của Mông Cổ | Từ chối | 20.4% (đúng) | RAG hỗ trợ trả lời câu hỏi có dữ kiện trong nguồn |
| Nam Ossetia thuộc quốc gia nào | Không có thông tin | Nga (sai; đáp án nguồn là Georgia/Gruzia) | Có context không đồng nghĩa model luôn diễn giải đúng |
| Cải tạo đô thị Paris | Không có thông tin | Thế kỷ 16–17 (sai; đáp án liên quan Napoleon III) | Model có thể tạo câu trả lời nghe hợp lý nhưng không khớp nguồn |
| Tỷ lệ học sinh/giáo viên ở Texas | Không có thông tin | 14.9 (được chấp nhận) | Câu trả lời số được đối chiếu với đáp án |
| Câu hỏi không có đáp án về sa mạc hóa | Từ chối | Sinh câu trả lời sai | RAG có thể làm model trả lời dù nguồn không hỗ trợ |
| Câu hỏi về kinh tế Paris không có đáp án | Từ chối | Sinh câu trả lời sai | Rủi ro hallucination vẫn tồn tại |

Một số câu trả lời khác được đánh giá đúng gồm thông tin về Mona Lisa tại Louvre, Charles-de-Gaulle và François Mitterrand. Với câu hỏi về việc xử tử Charles I, nhóm đối chiếu nguồn và hiệu chỉnh đánh giá theo sự kiện lịch sử: năm đúng là 1649; nhãn trong dữ liệu có sai lệch (ghi 1640).

### 5.4. Nhận xét

1. **RAG cải thiện rõ khả năng trả lời câu hỏi có thông tin trong nguồn.** Trên 15 câu answerable, số câu đúng tăng từ 0 lên 10; số lần từ chối giảm từ 14 xuống 2.
2. **RAG làm tăng rủi ro trả lời sai với câu impossible.** Trong 3 câu không có đáp án, No RAG từ chối cả 3; RAG từ chối 1 và tạo 2 câu trả lời sai.
3. **Chất lượng retrieval ảnh hưởng trực tiếp đến RAG.** Recall@3 = 0.722, tức 13/18 câu có context liên quan nằm trong top-3 theo cách đánh giá của notebook. Nếu truy xuất không đúng, model có thể thiếu bằng chứng để trả lời.
4. **Truy xuất đúng chưa bảo đảm sinh đúng.** Ví dụ Nam Ossetia cho thấy model vẫn có thể diễn giải sai dù đã có context.
5. **Giới hạn token có thể làm câu trả lời bị cắt.** Một số câu trả lời đúng nhưng bị truncation ở giới hạn 48 token, làm giảm độ đầy đủ của đầu ra.

## 6. Giới hạn chung

- Part 1 chỉ kiểm tra bốn prompt minh họa; không phải đánh giá hệ thống trên tập test lớn.
- Part 2 quan sát một cấu hình model và một ngữ cảnh dự đoán token; các thống kê kiến trúc không tự phản ánh chất lượng ngôn ngữ.
- Part 3 chỉ so sánh hai câu, một tiếng Việt và một tiếng Anh; kết quả không đại diện cho mọi loại văn bản hoặc miền dữ liệu.
- Part 8 chỉ dùng 18 câu hỏi được lấy mẫu từ UIT-ViQuAD 2.0 validation. Các tỷ lệ đúng/sai/từ chối vì vậy mang tính minh họa, không phải ước lượng tổng quát cho toàn bộ dataset.
- RAG dùng TF-IDF top-3, chưa so sánh với dense retrieval, reranker hoặc các cấu hình chunking khác.
- Một số câu trả lời bị giới hạn bởi `max_new_tokens=48`; đánh giá cần lưu ý trường hợp đầu ra bị truncation.
- Nhãn dữ liệu có thể có sai lệch; ví dụ câu hỏi về năm xử tử Charles I cần đối chiếu nguồn trước khi chấm.

## 7. Kết luận

Các thử nghiệm cho thấy Qwen2.5-1.5B-Instruct có thể thực hiện những tác vụ ngôn ngữ cơ bản, dự đoán token theo phân phối xác suất và xử lý token thông qua tokenizer/embedding. Thí nghiệm RAG cho thấy việc bổ sung context truy xuất có thể cải thiện đáng kể khả năng trả lời câu hỏi có bằng chứng, nhưng cũng có thể làm tăng câu trả lời sai đối với câu hỏi không có đáp án. Vì vậy, RAG không loại bỏ hallucination; chất lượng truy xuất, khả năng nhận biết thiếu bằng chứng và giới hạn sinh đều cần được xem xét.

## 8. Tệp kết quả cần đối chiếu

- `outputs/results/tasks_outputs.csv` — kết quả bốn tác vụ cơ bản.
- `outputs/results/top5_next_tokens.csv` — phân phối top-5 token tiếp theo.
- Các file kết quả tokenization/embedding trong `outputs/results/` — số token, kích thước tensor và kiểm tra round-trip.
- Các file kết quả RAG trong `outputs/results/` — câu hỏi, context truy xuất, câu trả lời, nhãn đánh giá và trạng thái truncation.
- `outputs/figures/` — hình minh họa kiến trúc, attention, tokenization và so sánh RAG (nếu có trong lần chạy chính thức).


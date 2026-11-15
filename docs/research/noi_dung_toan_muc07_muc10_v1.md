# Nội dung phần của Toàn — Mục 7 (Khả năng của LLM) và Mục 10 (Demo và kết luận) (v1)

- Người phụ trách: Toàn | Slide 14, 19, 20 | Thời lượng: mục 7 khoảng 1 phút, mục 10 khoảng 2.5 phút
- Notebook: phần 7 và phần 10; `src/icl_experiment.py`, `src/summary.py`
- Đề của cô:
  - Mục 7: in-context learning, làm theo chỉ dẫn, xử lý các tác vụ ngôn ngữ. Minh hoạ: so sánh zero-shot và few-shot trên cùng tác vụ.
  - Mục 10: thử nghiệm có kiểm soát, tổng hợp kết quả và nhận xét. Minh hoạ: bảng kết quả, ví dụ đầu ra và phân tích lỗi.
- Ký hiệu nguồn:
  - **[Slide]**: slide môn. Mục 7 và 10 **không có trên slide**, nên gần như toàn bộ là bổ sung.
  - **[Bổ sung]**: nguồn ngoài, đã mở kiểm tra ngày 2026-11-14 (§12).
  - **[Tự tính]**: nhóm tự tính, đã chạy lại bằng Python.
- Đi kèm: `research_presentation_llm_v1.md`, `plan_presentation_llm_v1.md` §5.8.
- **Chưa có kết quả chạy.** Mọi chỗ ghi `…` phải điền từ `outputs/results/` sau lần chạy chính thức; không điền trước.

---

## A. TÓM TẮT CHO TOÀN

### Ý chính phải nói được (thuộc lòng)
1. **In-context learning (ICL):** LLM làm được tác vụ mới chỉ nhờ chỉ dẫn hoặc ví dụ đặt **trong prompt**, **không cập nhật trọng số**.
2. **Zero-shot** = chỉ có chỉ dẫn; **few-shot** = chỉ dẫn + vài ví dụ có nhãn. Nhóm dùng 3-shot.
3. **Làm theo chỉ dẫn** đến từ **instruction tuning / SFT** (mục 5). Vì vậy dùng bản Instruct.
4. **Một model, nhiều tác vụ:** mọi tác vụ là "văn bản vào → văn bản ra"; phân loại cảm xúc cũng chỉ là sinh ra chữ "positive" hoặc "negative".
5. **Few-shot nhạy với việc chọn ví dụ, thứ tự ví dụ và cách viết prompt** → nhóm chạy **3 bộ ví dụ khác nhau**.
6. **Thí nghiệm có kiểm soát:** chỉ đổi **một biến** (có ví dụ hay không); giữ cố định model, dữ liệu, prompt template, decoding, seed.
7. **Đánh giá:**
   - accuracy, confusion matrix, tỉ lệ output không parse được;
   - **McNemar test**, vì hai cách chạy trên cùng 500 câu nên kết quả đi theo cặp.
8. **Kết luận chỉ rút từ số thật**, kèm giới hạn: 1 model nhỏ, 1 bộ dữ liệu, 1 tác vụ.

### Chú ý quan trọng (dễ sai, dễ bị hỏi)
| # | Chú ý | Vì sao |
|---|---|---|
| 1 | Phải truyền `do_sample=False` khi gọi `generate()` | Bản Instruct mặc định bốc mẫu ngẫu nhiên (`do_sample: true`, `temperature 0.7`, `top_p 0.8`, `top_k 20`), không phải greedy |
| 2 | UIT-VSFC có 3 nhãn → bỏ neutral | Đề nhóm làm 2 nhãn |
| 3 | Lấy test từ split **test**, ví dụ few-shot từ split **train**, và kiểm trùng câu | Tránh rò dữ liệu: câu test không được nằm trong ví dụ |
| 4 | 3 ví dụ thì không chia đều 2 nhãn được (2–1) | Model có xu hướng thiên về nhãn xuất hiện nhiều, hoặc nhãn của ví dụ cuối (Zhao et al. 2021); các bộ phải đổi nhãn nào nhiều hơn và thứ tự |
| 5 | Câu không parse được tính là **sai**, đồng thời báo riêng tỉ lệ | Không được bỏ đi rồi tính accuracy trên phần còn lại, như vậy là tô đẹp số |
| 6 | 500 câu → sai số khoảng ±4 điểm % | Chênh lệch nhỏ hơn mức này chưa chắc có thật → dùng McNemar (§7.4) |
| 7 | Không nói "few-shot luôn tốt hơn" | Chỉ đúng với kết quả nhóm chạy; có thể không cải thiện, thậm chí kém hơn |
| 8 | Không đổi prompt sau khi đã xem kết quả test | Như vậy là chọn trên tập test (lỗi Lab 1 cũng nhắc); muốn chỉnh prompt thì thử trên split validation |
| 9 | Mọi số trên slide lấy từ **1 lần Save & Run All** | Lab 1 mất điểm vì 3 bộ số khác nhau (`NHAN_XET_LAB1.md`) |
| 10 | Giải thích được từng dòng trong `icl_experiment.py` | Yêu cầu của cô; người viết phải tự giải thích |

---

## B. MỤC 7 — KHẢ NĂNG CỦA LLM

## 1. In-context learning (ICL)

### 1.1 Định nghĩa  [Bổ sung]
- Bài báo GPT-3 (Brown et al. 2020) cho thấy model làm được nhiều tác vụ **mà không cập nhật gradient hay fine-tune**: tác vụ và ví dụ được đưa vào hoàn toàn bằng văn bản.
- Cách "học" đó gọi là **in-context learning**: học từ ngữ cảnh, tức là từ prompt.
- Survey của Zhao et al. xếp ICL là một trong các khả năng "nổi lên" (emergent ability) của LLM, những khả năng model nhỏ thế hệ trước không có rõ.

### 1.2 Ba kiểu theo bài GPT-3  [Bổ sung]
| Kiểu | Trong prompt có | Ví dụ |
|---|---|---|
| Zero-shot | Chỉ mô tả tác vụ | "Phân loại cảm xúc câu sau…" |
| One-shot | Mô tả + 1 ví dụ | … + 1 câu có nhãn |
| Few-shot | Mô tả + k ví dụ (thường 10–100 trong bài GPT-3, tuỳ độ dài ngữ cảnh) | Nhóm dùng k = 3 vì model nhỏ và để prompt ngắn |

### 1.3 "Learning" nhưng không đổi trọng số: hiểu thế nào
- Trọng số model **giữ nguyên** trước và sau khi chạy. Hết prompt là ví dụ không còn tác dụng.
- Ví dụ giúp model nhận ra **đây là tác vụ gì và output phải có dạng nào**. Sau đó model "viết tiếp" theo mẫu, vẫn là next-token prediction ở mục 4.
- **Bằng chứng thú vị** (Min et al. 2022): thay nhãn trong ví dụ bằng nhãn **ngẫu nhiên** mà kết quả chỉ giảm nhẹ. Ví dụ giúp chủ yếu nhờ cho model thấy:
  - tập nhãn có thể chọn;
  - kiểu câu đầu vào;
  - định dạng câu trả lời.

  Dùng ý này cho Q&A, **không** thêm thành thí nghiệm (rule: không tự thêm thí nghiệm khi chưa được duyệt).

### 1.4 Điểm yếu: nhạy với ví dụ  [Bổ sung]
Zhao et al. 2021 (*Calibrate Before Use*) cho thấy độ chính xác few-shot thay đổi rất mạnh theo định dạng prompt, việc chọn ví dụ nào và thứ tự ví dụ. Họ chỉ ra các thiên lệch của model:
- **majority label bias:** thiên về nhãn xuất hiện nhiều trong các ví dụ;
- **recency bias:** thiên về nhãn của ví dụ đứng cuối;
- **common token bias:** thiên về token phổ biến.

→ Đây là **lý do khoa học** để nhóm chạy 3 bộ ví dụ khác nhau và báo cả khoảng dao động, không chỉ một con số.

---

## 2. Làm theo chỉ dẫn (instruction following)

- **Là gì:** hiểu và làm đúng yêu cầu viết bằng ngôn ngữ tự nhiên, ví dụ "chỉ trả lời positive hoặc negative".
- **Từ đâu ra**  [Bổ sung]:
  - FLAN (Wei et al. 2021): fine-tune model trên nhiều tác vụ được viết dưới dạng chỉ dẫn (instruction tuning) giúp zero-shot tốt hơn rõ rệt trên tác vụ chưa gặp.
  - InstructGPT (Ouyang et al. 2022): SFT trên câu trả lời mẫu của người, rồi học từ đánh giá của người (RLHF).
- **Liên hệ nhóm:** dùng `Qwen2.5-1.5B-Instruct` (đã qua post-training). Bản base chỉ qua pretraining; model card ghi không khuyến nghị dùng bản base để hội thoại → mục 5 (Phúc) cho thấy khác biệt.
- **Chú ý khi code:** bản Instruct cần **chat template** (`tokenizer.apply_chat_template(..., add_generation_prompt=True)`); không gõ prompt thô.

## 3. Xử lý các tác vụ ngôn ngữ

- Một model làm được nhiều tác vụ vì mọi tác vụ đều đưa được về dạng "văn bản vào → văn bản ra" (mục 1 đã demo 4 tác vụ).
- **Phân loại cảm xúc** được viết thành bài toán sinh văn bản: model sinh ra một từ nhãn, nhóm **parse** từ đó thành nhãn.
- Đây là điểm khác với cách làm ở Lab 3 / Practice 3 [Slide Ch4, phần thực hành Hugging Face]:

| | Practice 3 / Lab 3 | Mục 7 |
|---|---|---|
| Model | Model phân loại (`AutoModelForSequenceClassification`, có đầu phân loại) | Model sinh văn bản (`AutoModelForCausalLM`) |
| Cách làm | Fine-tune bằng `Trainer` | Không train; chỉ prompt |
| Cần dữ liệu nhãn | Nhiều | 0 (zero-shot) hoặc 3 (few-shot) |

---

## 4. Minh hoạ mục 7 (slide 14)

### 4.1 Prompt template  [đề xuất — Toàn chốt; sau khi chốt KHÔNG đổi]

Một lượt `user` trong chat template. Có ví dụ là few-shot, không có ví dụ là zero-shot:

```
Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời một từ: positive hoặc negative.

Câu: "<ví dụ 1>"
Cảm xúc: positive

Câu: "<ví dụ 2>"
Cảm xúc: negative

Câu: "<ví dụ 3>"
Cảm xúc: positive

Câu: "<câu cần phân loại>"
Cảm xúc:
```

- Dùng từ nhãn tiếng Anh `positive` / `negative` cho dễ parse; câu đầu vào giữ nguyên tiếng Việt.
- Zero-shot dùng **đúng phần chỉ dẫn và phần câu cuối giống hệt**, chỉ bỏ 3 khối ví dụ → đúng "chỉ đổi một biến".
- Nếu muốn thử cách viết khác: thử trên **split validation**, chốt rồi mới chạy test.

### 4.2 Trên slide 14
- **Trái:** 3 ý (ICL không đổi trọng số; làm theo chỉ dẫn nhờ SFT; một model nhiều tác vụ).
- **Phải:** 2 prompt cạnh nhau (zero-shot, 3-shot), rút gọn.
- **Dưới:** 2–3 câu test kèm output thật của 2 cách, và dòng "Accuracy trên 500 câu → slide 19".
- **Góc:** "Notebook phần 7, cell …".

### 4.3 Lời thoại mục 7 (khoảng 1 phút)
> "LLM có khả năng gọi là in-context learning: chỉ cần đưa chỉ dẫn hoặc vài ví dụ vào prompt là model làm được tác vụ mới, mà không phải huấn luyện lại hay thay đổi trọng số nào. Zero-shot là chỉ đưa chỉ dẫn; few-shot là thêm vài ví dụ có nhãn, nhóm dùng 3 ví dụ.
> Ví dụ không dạy thêm kiến thức, mà cho model thấy tác vụ là gì và phải trả lời theo định dạng nào. Còn khả năng làm theo chỉ dẫn đến từ giai đoạn SFT ở mục 5.
> Bên phải là hai prompt trên cùng tác vụ phân loại cảm xúc phản hồi sinh viên. [Chỉ output thật] … Nhưng vài câu chưa nói được gì, nên nhóm làm thí nghiệm có kiểm soát trên 500 câu, em trình bày ở phần cuối."

---

## C. MỤC 10 — DEMO VÀ KẾT LUẬN

## 5. Câu hỏi thí nghiệm

> Trên tác vụ phân loại cảm xúc 2 nhãn của UIT-VSFC, với `Qwen2.5-1.5B-Instruct`, thêm 3 ví dụ vào prompt (3-shot) có làm accuracy khác zero-shot **một cách có ý nghĩa thống kê** không?

- **H0:** zero-shot và 3-shot có tỉ lệ sai như nhau.
- **H1:** khác nhau.
- Kết luận dựa trên số thật; **không đoán trước** chiều nào thắng.

## 6. Thiết lập (viết trước được, không phụ thuộc kết quả)

### 6.1 Dữ liệu  [Bổ sung, đã kiểm]
| Thông tin | Giá trị |
|---|---|
| Bộ dữ liệu | UIT-VSFC (Nguyen et al., KSE 2018), phản hồi sinh viên tiếng Việt |
| Nhãn gốc | 0 negative, 1 neutral, 2 positive → **bỏ neutral** |
| Split test gốc | 3,166 câu: negative 1,409 · neutral 167 · positive 1,590 |
| Split train gốc | 11,426 câu: negative 5,325 · neutral 458 · positive 5,643 |
| Test của nhóm | **500 câu = 250 negative + 250 positive**, lấy ngẫu nhiên từ split test, seed 42, lưu `data/splits/test_500_ids.json` |
| Ví dụ few-shot | Lấy từ split **train**; 3 bộ × 3 câu; lưu `data/splits/fewshot_pools.json`; kiểm không trùng câu test |
| Cách tải | `pandas.read_parquet` từ nhánh Parquet trên Hub (script loader không chạy với `datasets` ≥ 4.0) |
| Độ tin cậy nhãn | Bài báo báo độ đồng thuận giữa người gán nhãn sentiment trên 91%, tức vẫn có nhãn bất đồng → nguồn lỗi "nhãn nhiễu" |

**Vì sao cân bằng 250/250:** đoán bừa hoặc luôn đoán một nhãn chỉ được khoảng 50% → có mốc so sánh rõ ràng; accuracy không bị lệch do một nhãn chiếm đa số.

### 6.2 Ba bộ ví dụ few-shot  [đề xuất]
| Bộ | Thứ tự nhãn | Ý đồ |
|---|---|---|
| 1 | positive, negative, positive | Dư positive, ví dụ cuối positive |
| 2 | negative, positive, negative | Dư negative, ví dụ cuối negative (đối xứng bộ 1) |
| 3 | Ngẫu nhiên theo seed | Kiểm tra thêm một lần lấy mẫu |

Lý do: 3 ví dụ không chia đều 2 nhãn; bộ 1 và 2 đối xứng để thiên lệch nhãn đa số và nhãn cuối (Zhao et al. 2021) không chỉ nghiêng về một phía.

### 6.3 Biến cố định và biến thay đổi
| Cố định | Giá trị |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, dtype ghi trong `env.json` (fp16 hoặc fp32 theo buổi chạy thử) |
| Decoding | Greedy: `do_sample=False`, `max_new_tokens` nhỏ (khoảng 5) |
| Prompt | Template §4.1, chat template của tokenizer |
| Dữ liệu test | 500 câu cố định |
| Seed | 42 |
| Môi trường | Kaggle GPU T4; phiên bản thư viện trong `outputs/logs/env.json` |

| Thay đổi (biến độc lập) | 4 lượt chạy |
|---|---|
| Có ví dụ hay không, và bộ ví dụ nào | zero-shot · 3-shot bộ 1 · 3-shot bộ 2 · 3-shot bộ 3 |

Tổng: 500 × 4 = 2,000 lần sinh.

### 6.4 Parse output → nhãn
- Chuẩn hoá: chữ thường, bỏ khoảng trắng và dấu câu.
- Có `positive` (và không có `negative`) → positive; ngược lại → negative.
- Có cả hai, hoặc không có nhãn nào → `None` (**không parse được**).
- Có thể nhận thêm "tích cực", "tiêu cực" **[cần chốt]**. Nếu nhận thì ghi rõ trong slide và test.
- Mọi output thô đều lưu vào CSV để kiểm lại bằng mắt.

## 7. Đánh giá

### 7.1 Accuracy
- `accuracy = số câu đúng / 500`. Câu `None` tính là **sai**.
- Báo thêm tỉ lệ không parse được = số `None` / 500.

### 7.2 Confusion matrix (2×2, thêm một cột "không parse được")
| | Dự đoán positive | Dự đoán negative | Không parse được |
|---|---|---|---|
| Thật positive (250) | … | … | … |
| Thật negative (250) | … | … | … |

Đọc để biết model **hay nhầm theo chiều nào**, ví dụ có thiên về positive không.

### 7.3 Sai số do lấy mẫu  [Tự tính]
Với n = 500, nửa độ rộng khoảng tin cậy 95% (xấp xỉ chuẩn, 1.96·√(p(1−p)/n)):

| Accuracy p | ± |
|---|---|
| 50% | 4.4 điểm |
| 70% | 4.0 điểm |
| 80% | 3.5 điểm |
| 90% | 2.6 điểm |

→ Nhìn hai accuracy chênh nhau 2–3 điểm thì **chưa kết luận được**. Nhưng hai cách chạy trên **cùng câu**, nên dùng kiểm định theo cặp (McNemar) sẽ nhạy hơn so hai khoảng tin cậy rời.

### 7.4 McNemar test  [Bổ sung]
**Ý tưởng:** chỉ những câu mà hai cách **cho kết quả khác nhau** mới nói lên cách nào tốt hơn.

| | 3-shot đúng | 3-shot sai |
|---|---|---|
| **Zero-shot đúng** | a (cả hai đúng) | **b** (chỉ zero-shot đúng) |
| **Zero-shot sai** | **c** (chỉ 3-shot đúng) | d (cả hai sai) |

- Nếu hai cách tốt như nhau thì b ≈ c. McNemar kiểm b và c chênh nhau có đủ lớn không.
- `statsmodels.stats.contingency_tables.mcnemar(table, exact=True)`: `exact=True` dùng phân phối nhị thức, hợp khi b + c nhỏ. Trả `statistic`, `pvalue`.
- **Vì sao McNemar mà không t-test:** Dietterich (1998) so 5 kiểm định và kết luận với thuật toán chỉ chạy được một lần thì McNemar là kiểm định duy nhất có sai lầm loại I chấp nhận được. Nhóm chạy mỗi cấu hình một lần trên một tập test → đúng trường hợp này.
- **3 bộ few-shot = 3 lần kiểm định** (mỗi bộ so với zero-shot). Kiểm nhiều lần thì dễ "trúng" ngẫu nhiên; cách thận trọng là dùng ngưỡng 0.05 / 3 ≈ 0.017 (hiệu chỉnh Bonferroni) **[cần chốt]**, hoặc nói rõ là chưa hiệu chỉnh.

### 7.5 Bảng kết quả chính (slide 19), điền sau khi chạy
| Cách | Accuracy | Không parse được | b | c | p-value McNemar (so zero-shot) |
|---|---|---|---|---|---|
| Zero-shot | … | … | — | — | — |
| 3-shot bộ 1 | … | … | … | … | … |
| 3-shot bộ 2 | … | … | … | … | … |
| 3-shot bộ 3 | … | … | … | … | … |
| 3-shot trung bình (min–max) | … (… – …) | … | | | |

Dưới bảng ghi nguồn: `outputs/results/icl_metrics.json`, ngày chạy, model, dtype.

## 8. Phân tích lỗi (`docs/experiments/error_analysis_v1.md`)

### 8.1 Cách làm
1. Lọc các câu sai trong CSV của zero-shot và 3-shot.
2. Đọc từng câu; xếp vào một nhóm lỗi (bảng dưới). Ghi lý do ngắn.
3. Đếm mỗi nhóm; chọn 1–2 ví dụ thật mỗi nhóm cho slide.
4. Riêng câu `None`: ghi model đã trả gì (lạc đề, trả cả hai nhãn, giải thích dài…).

### 8.2 Nhóm lỗi dự kiến  [đề xuất; nhóm lỗi thật quyết định sau khi đọc dữ liệu]
| Nhóm lỗi | Ví dụ kiểu câu (minh hoạ, không phải dữ liệu thật) |
|---|---|
| Vừa khen vừa chê | "Thầy dạy hay nhưng cho bài tập quá nhiều" |
| Mỉa mai, nói ngược | "Phòng học mát quá, mồ hôi chảy ròng ròng" |
| Câu quá ngắn, thiếu ngữ cảnh | "bình thường" |
| Viết tắt, sai chính tả, teencode | "k hiểu j hết" |
| Nhãn gốc có thể sai (nhãn nhiễu) | Câu đọc thấy tích cực nhưng nhãn negative |
| Lỗi định dạng output | Trả lời cả câu, trả cả hai nhãn, không có nhãn |

**Chú ý:** ví dụ trên slide phải là **câu thật** trong file kết quả, không dùng câu tự đặt ở bảng này. Câu nghi "nhãn nhiễu" chỉ gọi là "có thể sai", vì nhóm không phải người gán nhãn gốc.

## 9. Tổng hợp kết quả các phần (`src/summary.py` → slide 19–20)

Mục 10 trong đề là "tổng hợp kết quả" → gom số thật từ mọi phần thành một bảng ngắn:

| Mục | Thí nghiệm | Kết quả chính (điền từ `outputs/results/`) | Người |
|---|---|---|---|
| 3 | Số token câu Việt / Anh, GPT-2 vs Qwen | … | Muội |
| 4 | Loss trên câu ví dụ | … | Phúc |
| 5 | Base vs Instruct | … (nhận xét định tính) | Phúc |
| 6 | 5 cấu hình decoding | … (độ đa dạng / lặp) | Phúc |
| 7, 10 | Zero-shot vs 3-shot, 500 câu | Accuracy …, p-value … | Toàn |
| 8 | Không RAG vs có RAG | Số câu đúng … / … | Muội |
| 9 | Hallucination | Số câu sai … / 20 | Phúc |

Ngày 6 lấy số thật từ Muội và Phúc; không chép từ slide của họ mà đọc từ file CSV.

## 10. Kết luận (slide 20) — khung, điền sau khi có số

Viết 3–4 ý, mỗi ý gắn với một con số hoặc một ví dụ thật:
1. **Về ICL:** "Trên 500 câu UIT-VSFC, 3-shot đạt …% so với zero-shot …%; McNemar p = … → (có / chưa có) khác biệt có ý nghĩa."
2. **Về độ nhạy:** "3 bộ ví dụ cho accuracy từ …% đến …% → kết quả few-shot phụ thuộc ví dụ được chọn."
3. **Về lỗi:** "Lỗi tập trung ở … (… câu); … câu không parse được."
4. **Về LLM nói chung**, rút từ các mục khác: ví dụ RAG giúp …, model bịa ở …

**Giới hạn phải nói:**
- 1 model 1.5 tỉ tham số, 1 bộ dữ liệu, 1 tác vụ;
- mỗi cấu hình chạy 1 lần;
- 3 ví dụ;
- prompt tiếng Việt với nhãn tiếng Anh.

Không suy rộng sang "LLM nói chung".

## 11. Lời thoại mục 10 (khoảng 2.5 phút) — khung

> **[Slide 19]** "Để kiểm tra few-shot có thật sự giúp hay không, nhóm làm một thí nghiệm có kiểm soát. Tác vụ là phân loại cảm xúc phản hồi sinh viên, bộ UIT-VSFC. Nhóm bỏ nhãn trung tính, lấy 500 câu test cân bằng 250 tích cực, 250 tiêu cực, nên đoán bừa chỉ được khoảng 50%.
> Nhóm chỉ thay đổi một thứ là có ví dụ trong prompt hay không. Model, prompt, cách sinh greedy, dữ liệu, seed đều giữ nguyên. Vì few-shot nhạy với việc chọn ví dụ, nhóm chạy 3 bộ ví dụ khác nhau, lấy từ tập train để không trùng câu test.
> Kết quả: zero-shot …%, 3-shot từ …% đến …%. Vì hai cách chạy trên cùng câu, nhóm dùng McNemar test, chỉ xét những câu hai cách cho kết quả khác nhau. p-value … nên …
> [Confusion matrix] Model hay nhầm … . [Ví dụ lỗi thật] Câu này … vì … .
> **[Slide 20]** Tóm lại … . Giới hạn là nhóm chỉ dùng một model nhỏ, một bộ dữ liệu, mỗi cấu hình chạy một lần."

---

## D. Q&A DỰ KIẾN (mục 7 và 10)

| # | Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|---|
| 1 | Few-shot có thay đổi trọng số model không? | Không. Ví dụ chỉ nằm trong prompt; hết prompt là mất tác dụng. Khác fine-tuning ở mục 8 | Brown et al. 2020 |
| 2 | Vậy vì sao gọi là "learning"? | Model điều chỉnh hành vi theo ngữ cảnh, không theo trọng số. Ví dụ cho thấy tập nhãn, kiểu câu, định dạng trả lời | Min et al. 2022 |
| 3 | Few-shot có luôn tốt hơn zero-shot không? | Không chắc; phụ thuộc model, tác vụ, ví dụ. Nhóm đo được … (nói theo số thật) | Zhao et al. 2021; kết quả nhóm |
| 4 | Vì sao chạy 3 bộ ví dụ? | Few-shot nhạy với việc chọn và thứ tự ví dụ (thiên lệch nhãn đa số, nhãn cuối); một bộ có thể may hoặc xui | Zhao et al. 2021 |
| 5 | Vì sao dùng McNemar mà không chỉ so accuracy? | Hai cách trên cùng 500 câu → kết quả theo cặp. McNemar chỉ xét câu bất đồng (b, c); hợp với model chỉ chạy một lần | Dietterich 1998 |
| 6 | Chênh 2 điểm accuracy có ý nghĩa không? | Với 500 câu, sai số khoảng ±4 điểm; phải xem p-value McNemar mới kết luận | Tự tính §7.3 |
| 7 | Câu không parse được xử lý thế nào? | Tính là sai, và báo riêng tỉ lệ. Bỏ đi thì accuracy bị thổi phồng | §7.1 |
| 8 | Sao không dùng neutral? | Đề nhóm làm 2 nhãn; neutral ít (167/3,166 câu test ≈ 5.3%) và khó định nghĩa ranh giới | Thống kê UIT-VSFC |
| 9 | Ví dụ few-shot có trùng câu test không? | Không: ví dụ lấy từ split train, test từ split test, và có test kiểm trùng | §6.1 |
| 10 | Vì sao dùng greedy? | Để kết quả lặp lại được: cùng input luôn ra cùng output. Phải truyền `do_sample=False` vì bản Instruct mặc định bốc mẫu | `generation_config.json` của Qwen |
| 11 | Khác gì Lab 3 (fine-tune BERT phân loại cảm xúc)? | Lab 3 train đầu phân loại với nhiều dữ liệu nhãn; ở đây không train, chỉ prompt với 0 hoặc 3 ví dụ | Practice 3, mục 8 |
| 12 | Kết quả này có áp dụng cho LLM nói chung không? | Không suy rộng: 1 model nhỏ, 1 bộ dữ liệu, 1 tác vụ, mỗi cấu hình 1 lần chạy | §10 |

---

## E. Code phải giải thích được (`src/icl_experiment.py`)

| Hàm | Câu cô có thể hỏi | Ý trả lời |
|---|---|---|
| `make_test_split` | Sao cùng seed luôn ra cùng 500 câu? | Bộ sinh số ngẫu nhiên khởi tạo bằng seed cố định → cùng thứ tự lấy mẫu |
| `make_fewshot_pools` | Làm sao chắc ví dụ không trùng test? | Lấy từ split train và loại câu có nội dung trùng câu test; có test kiểm |
| `build_prompt` | Zero-shot khác 3-shot ở đâu trong code? | Cùng một hàm; `examples` rỗng là zero-shot → chỉ một biến thay đổi |
| `run_icl` | Sao chắc là greedy? | Truyền `do_sample=False`; in `model.generation_config` để chứng minh |
| `parse_label` | Output "Positive." thì sao? | Chuẩn hoá chữ thường, bỏ dấu câu, rồi tìm từ nhãn |
| `compute_metrics` | `None` tính thế nào? | Tính là sai; báo riêng tỉ lệ |
| `mcnemar_counts` | b, c là gì? | b: chỉ cách A đúng; c: chỉ cách B đúng |
| `mcnemar_pvalue` | Vì sao `exact=True`? | b + c có thể nhỏ → dùng phân phối nhị thức chính xác |

---

## F. Việc và hạn của Toàn (theo kế hoạch nhóm)

| Ngày | Việc |
|---|---|
| 1 | Chốt cách chia vai; tạo khung thư mục, `test_smoke.py` (sau khi Thư duyệt plan) |
| 2 | Lý thuyết + slide 14, 19 (phần thiết lập), 20 (khung) — dùng file này |
| 3 | `icl_experiment.py` + test (không cần model); chạy thử vài câu trên Kaggle |
| 4 | Chạy thử với `SMOKE = True`; chốt prompt trên split validation |
| 5 | Chạy chính thức 500 câu × 4; metric, McNemar; `error_analysis_v1.md`; 5 câu Q&A |
| 6 | `summary.py`; ghép notebook; Save & Run All; xuất HTML dự phòng; điền số vào slide 14, 19, 20 |
| 7 | Tập trình bày, mở notebook giải thích code |

---

## 12. Tài liệu tham khảo cho mục 7 và 10

| Tài liệu | Dùng cho |
|---|---|
| Brown et al., *Language Models are Few-Shot Learners*, 2020. https://arxiv.org/abs/2005.14165 | ICL; zero/one/few-shot; không cập nhật gradient |
| Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN), 2021. https://arxiv.org/abs/2109.01652 | Instruction tuning giúp zero-shot |
| Ouyang et al., *Training language models to follow instructions with human feedback*, 2022. https://arxiv.org/abs/2203.02155 | SFT + RLHF → làm theo chỉ dẫn |
| Zhao, Wallace, Feng, Klein, Singh, *Calibrate Before Use: Improving Few-Shot Performance of Language Models*, 2021. https://arxiv.org/abs/2102.09690 | Few-shot nhạy với ví dụ, thứ tự; các thiên lệch nhãn |
| Min et al., *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*, 2022. https://arxiv.org/abs/2202.12837 | Nhãn ngẫu nhiên chỉ giảm nhẹ; ví dụ cho thấy tập nhãn, kiểu câu, định dạng |
| Zhao et al., *A Survey of Large Language Models*, 2023. https://arxiv.org/abs/2303.18223 | ICL là khả năng nổi lên của LLM |
| Nguyen et al., *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis*, KSE 2018, DOI 10.1109/KSE.2018.8573337 | Dữ liệu; độ đồng thuận nhãn |
| Dataset trên Hub: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback | Số câu, nhãn từng split |
| Dietterich, *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*, Neural Computation 10(7):1895–1923, 1998. DOI 10.1162/089976698300017197 | Vì sao dùng McNemar |
| statsmodels `mcnemar`: https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html | API |
| Model card + `generation_config.json` của Qwen2.5-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Cấu hình sinh mặc định; chat template |
| Hugging Face, *Chat templates*: https://huggingface.co/docs/transformers/chat_templating | `apply_chat_template`, `add_generation_prompt` |

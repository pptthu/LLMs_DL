# Nội dung phần của Toàn — Mục 7 (Khả năng của LLM) và Mục 10 (Demo và kết luận) (v2)

- Người phụ trách: Toàn. Toàn cũng viết toàn bộ code notebook (quyết định 15/11).
- Slide 14, 19, 20. Thời lượng: mục 7 khoảng 1 phút, mục 10 khoảng 2.5 phút.
- Code: `notebook/llm_presentation.ipynb`, **Phần 7** và **Phần 10**. Repo không có thư mục mã nguồn riêng: mọi hàm và các `assert` tự kiểm đều nằm trong notebook.
- Đề của cô:
  - Mục 7: in-context learning, làm theo chỉ dẫn, xử lý các tác vụ ngôn ngữ. Minh hoạ: so sánh zero-shot và few-shot trên cùng tác vụ.
  - Mục 10: thử nghiệm có kiểm soát, tổng hợp kết quả và nhận xét. Minh hoạ: bảng kết quả, ví dụ đầu ra và phân tích lỗi.
- Ký hiệu nguồn:
  - **[Slide]**: slide môn. Mục 7 và 10 **không có trên slide**. Phần liên quan duy nhất là Practice 3, Hugging Face (Ch4 slide 25).
  - **[Bổ sung]**: nguồn ngoài, có ở §12.
  - **[Tự tính]**: nhóm tự tính và đã chạy lại bằng Python.
- **Chưa có kết quả chạy.** Mọi chỗ ghi `…` phải điền từ `outputs/results/` của lần Save & Run All chính thức. Không điền trước.

**Sửa so với v1 (15/11):**
- Bỏ `src/`, `test_smoke.py`: code nằm trong notebook.
- Đường dẫn theo `FOLDER_STRUCTURE_v2`: `data/` phẳng, `env.json` ở `outputs/results/`, log ở `outputs/logs/<YYYYMMDD_HHMMSS>_<smoke|normal>/`.
- Khung lý thuyết mục 7 theo 3 khả năng mà survey Zhao et al. nêu.
- Sửa trích dẫn Practice 3.
- Thêm ý "emergent có tranh cãi" và ý "system prompt mặc định của Qwen".
- Cập nhật phân công và lịch.

---

## A. Ý CHÍNH (thuộc lòng)

### Mục 7: ba khả năng, mỗi khả năng một câu
| Khả năng | Một câu để nói | Từ đâu mà có |
|---|---|---|
| **In-context learning (ICL)** | Đưa chỉ dẫn hoặc vài ví dụ **vào prompt** là model làm được tác vụ mới; **trọng số không đổi** | Pretraining quy mô lớn (Brown et al. 2020) |
| **Làm theo chỉ dẫn** | Hiểu và làm đúng yêu cầu viết bằng ngôn ngữ tự nhiên, kể cả tác vụ chưa gặp | Instruction tuning / SFT (+ RLHF) ở mục 5 (FLAN; InstructGPT) |
| **Xử lý nhiều tác vụ ngôn ngữ** | Mọi tác vụ đều đưa được về dạng "văn bản vào → văn bản ra", nên một model làm được nhiều việc | Cùng cơ chế dự đoán token tiếp theo ở mục 4 |

Thêm 3 ý phải nói được:
1. **Zero-shot** = chỉ có chỉ dẫn. **Few-shot** = chỉ dẫn + k ví dụ có nhãn. Nhóm dùng k = 3.
2. **Few-shot nhạy với việc chọn ví dụ, thứ tự ví dụ và cách viết prompt** (Zhao et al. 2021), nên nhóm chạy **3 bộ ví dụ**.
3. **Ví dụ không dạy kiến thức mới.** Chủ yếu chúng cho model thấy tập nhãn, kiểu câu đầu vào và định dạng trả lời (Min et al. 2022).

### Mục 10: năm ý
1. **Thí nghiệm có kiểm soát:** chỉ đổi **một biến** (có ví dụ hay không, bộ nào). Giữ cố định model, dữ liệu, prompt template, decoding greedy, seed.
2. **Thiết lập:** UIT-VSFC, bỏ neutral; 500 câu test cân bằng 250/250, lấy từ split test; ví dụ few-shot lấy từ split train.
3. **Đánh giá:**
   - accuracy (câu không parse được tính là sai);
   - confusion matrix;
   - tỉ lệ output không parse được;
   - **McNemar**, vì hai cách chạy trên cùng 500 câu nên kết quả đi theo cặp.
4. **Phân tích lỗi** bằng câu thật trong file CSV, xếp theo nhóm lỗi.
5. **Kết luận chỉ rút từ số thật**, kèm giới hạn: 1 model 1.5B, 1 bộ dữ liệu, 1 tác vụ, mỗi cấu hình chạy 1 lần.

### Chú ý (dễ sai, dễ bị hỏi)
| # | Chú ý | Vì sao |
|---|---|---|
| 1 | Luôn truyền `do_sample=False` vào `generate()` | `generation_config.json` của bản Instruct mặc định lấy mẫu: `do_sample: true`, `temperature 0.7`, `top_p 0.8`, `top_k 20`, `repetition_penalty 1.1` |
| 2 | Dùng `apply_chat_template(..., add_generation_prompt=True)` | Bản Instruct được huấn luyện với định dạng hội thoại. Khi không truyền `system`, template của Qwen2.5 có thể tự chèn system prompt mặc định, nên **in prompt cuối ra một lần** để kiểm và giữ y hệt giữa các cấu hình |
| 3 | Test từ split **test**, ví dụ từ split **train**, có `assert` không trùng câu | Tránh rò dữ liệu |
| 4 | 3 ví dụ không chia đều 2 nhãn được (2–1) | Thiên lệch nhãn đa số và nhãn cuối (Zhao et al. 2021), nên bộ 1 và bộ 2 đối xứng |
| 5 | Câu không parse được tính là **sai**, và báo riêng tỉ lệ | Bỏ đi rồi tính accuracy trên phần còn lại là thổi phồng số |
| 6 | 500 câu → sai số khoảng ±4 điểm % | Chênh nhỏ hơn mức này chưa chắc có thật, phải xem McNemar |
| 7 | Không nói "few-shot luôn tốt hơn" | Chỉ nói theo số nhóm đo được |
| 8 | Không đổi prompt sau khi đã xem kết quả test | Như vậy là chọn trên tập test. Muốn chỉnh thì thử trên split validation |
| 9 | Mọi số trên slide lấy từ **1 lần Save & Run All** | Bài học Lab 1: 3 bộ số khác nhau thì mất điểm |
| 10 | Giải thích được từng dòng code Phần 7 và Phần 10 | Yêu cầu của cô |

---

## B. MỤC 7 — KHẢ NĂNG CỦA LLM (lý thuyết)

### 1. In-context learning (ICL)

**1.1 Định nghĩa [Bổ sung]**
- Bài GPT-3 (Brown et al. 2020) cho thấy model làm được nhiều tác vụ **mà không cập nhật gradient hay fine-tune**. Tác vụ và ví dụ được đưa vào hoàn toàn bằng văn bản.
- Cách "học" này gọi là in-context learning: học từ ngữ cảnh, tức là từ prompt.

**1.2 Ba kiểu, theo bài GPT-3 [Bổ sung]**
| Kiểu | Trong prompt có | Ghi chú |
|---|---|---|
| Zero-shot | Chỉ mô tả tác vụ | |
| One-shot | Mô tả + 1 ví dụ | |
| Few-shot | Mô tả + k ví dụ | Bài GPT-3 thường dùng k từ 10 đến 100, tuỳ độ dài ngữ cảnh. Nhóm dùng k = 3 vì model nhỏ và để prompt ngắn |

**1.3 "Learning" mà không đổi trọng số: hiểu thế nào**
- Trọng số **giữ nguyên** trước và sau khi chạy. Hết prompt thì ví dụ không còn tác dụng. Đây là điểm khác fine-tuning ở mục 8.
- Ví dụ giúp model nhận ra tác vụ là gì và output phải có dạng nào. Sau đó model "viết tiếp" theo mẫu, vẫn là next-token prediction.
- Min et al. 2022: thay nhãn trong ví dụ bằng nhãn **ngẫu nhiên** thì kết quả chỉ giảm nhẹ. Ví dụ có ích chủ yếu nhờ cho model thấy tập nhãn, phân phối câu đầu vào và định dạng. Dùng ý này cho Q&A, **không** thêm thành thí nghiệm.

**1.4 Điểm yếu: nhạy với ví dụ [Bổ sung]**
Zhao et al. 2021 (*Calibrate Before Use*) cho thấy accuracy few-shot dao động mạnh theo định dạng prompt, việc chọn ví dụ và thứ tự ví dụ. Nguyên nhân là ba thiên lệch:
- **majority label bias:** thiên về nhãn xuất hiện nhiều trong ví dụ;
- **recency bias:** thiên về nhãn của ví dụ đứng cuối;
- **common token bias:** thiên về token phổ biến trong dữ liệu huấn luyện.

Đây là lý do nhóm chạy 3 bộ ví dụ và báo cả khoảng dao động.

**1.5 ICL có phải khả năng "nổi lên" (emergent)? [Bổ sung]**
- Survey của Zhao et al. (2023) xếp ICL, làm theo chỉ dẫn và suy luận từng bước là các khả năng nổi lên của LLM. Wei et al. (2022) định nghĩa khả năng nổi lên là khả năng không có ở model nhỏ mà có ở model lớn.
- Có tranh cãi: Schaeffer et al. (2023) cho rằng nhiều khả năng "nổi lên" một phần là do cách chọn thước đo. Thước đo đúng/sai tuyệt đối làm đường cong trông như nhảy vọt.
- Trên slide **không khẳng định** model 1.5B của nhóm "có" hay "không có" khả năng nổi lên. Chỉ báo số đo được.

### 2. Làm theo chỉ dẫn (instruction following)
- **Là gì:** hiểu và làm đúng yêu cầu viết bằng ngôn ngữ tự nhiên, ví dụ "Chỉ trả lời một từ: positive hoặc negative".
- **Từ đâu ra [Bổ sung]:**
  - FLAN (Wei et al. 2021): fine-tune trên nhiều tác vụ viết dưới dạng chỉ dẫn (instruction tuning) giúp zero-shot tốt hơn rõ trên tác vụ chưa gặp.
  - InstructGPT (Ouyang et al. 2022): SFT trên câu trả lời mẫu của người, rồi RLHF.
- **Liên hệ nhóm:** dùng `Qwen/Qwen2.5-1.5B-Instruct`, bản đã post-training. Model card của bản base ghi không khuyến nghị dùng base để hội thoại. Khác biệt base và Instruct thuộc mục 5 (Phúc thuyết trình).
- **Liên hệ ICL:** instruction tuning làm **zero-shot** mạnh lên. Vì vậy với model Instruct, few-shot không chắc giúp thêm nhiều. Đây chính là câu hỏi thí nghiệm của mục 10.

### 3. Xử lý các tác vụ ngôn ngữ
- Mọi tác vụ đều đưa được về "văn bản vào → văn bản ra". Mục 1 đã demo nhiều tác vụ trên cùng một model.
- **Phân loại cảm xúc thành bài toán sinh:** model sinh một từ nhãn, nhóm **parse** từ đó thành nhãn.
- So với Practice 3:

| | Practice 3 (Ch4 slide 25) | Mục 7 |
|---|---|---|
| Model | Model phân loại pre-trained, ví dụ DistilBERT có đầu phân loại (`AutoModelForSequenceClassification`, chi tiết API là bổ sung) | Model sinh văn bản (`AutoModelForCausalLM`) |
| Cách làm | Ex1 `pipeline`, Ex2 fine-tune bằng `Trainer` | Không train; chỉ prompt |
| Dữ liệu có nhãn cần | Nhiều (Ex2) | 0 (zero-shot) hoặc 3 (few-shot) |
| Output | Logits của 2 lớp | Một chuỗi chữ, phải parse |

### 4. Minh hoạ mục 7 (slide 14)

**4.1 Prompt template [đề xuất, Toàn chốt; sau khi chốt KHÔNG đổi]**

Một lượt `user` trong chat template. Có ví dụ là few-shot, không có ví dụ là zero-shot.

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

- Nhãn tiếng Anh `positive` / `negative` để dễ parse; câu đầu vào giữ nguyên tiếng Việt.
- Zero-shot dùng **y hệt** phần chỉ dẫn và phần câu cuối, chỉ bỏ 3 khối ví dụ. Như vậy chỉ đổi một biến.
- Cách viết khác thì thử trên split validation, chốt rồi mới chạy test.

**4.2 Bố cục slide 14**
- Trái: bảng 3 khả năng (§A).
- Phải: 2 prompt cạnh nhau (zero-shot, 3-shot), rút gọn.
- Dưới: 2–3 câu test kèm output thật của hai cách, và dòng "Accuracy trên 500 câu → slide 19".
- Góc: "Notebook Phần 7, cell …".

**4.3 Lời thoại mục 7 (khoảng 1 phút)**
> "Phần này em trình bày ba khả năng của LLM. Thứ nhất là in-context learning: chỉ cần đưa chỉ dẫn hoặc vài ví dụ vào prompt là model làm được tác vụ mới, mà không thay đổi trọng số nào. Zero-shot là chỉ có chỉ dẫn; few-shot là thêm vài ví dụ có nhãn, nhóm dùng ba ví dụ. Ví dụ không dạy kiến thức mới, mà cho model thấy tác vụ là gì và phải trả lời theo định dạng nào.
> Thứ hai là làm theo chỉ dẫn, có được nhờ giai đoạn instruction tuning ở mục 5. Thứ ba, vì mọi tác vụ đều là văn bản vào, văn bản ra, nên một model làm được nhiều tác vụ. Ví dụ phân loại cảm xúc ở đây chỉ là sinh ra chữ positive hoặc negative.
> Bên phải là hai prompt trên cùng tác vụ. [Chỉ output thật] Vài câu thì chưa nói được gì, nên nhóm làm thí nghiệm có kiểm soát trên 500 câu, em trình bày ở phần cuối."

---

## C. MỤC 10 — DEMO VÀ KẾT LUẬN

### 5. Câu hỏi thí nghiệm
> Với `Qwen2.5-1.5B-Instruct`, trên tác vụ phân loại cảm xúc 2 nhãn của UIT-VSFC, thêm 3 ví dụ vào prompt (3-shot) có làm accuracy khác zero-shot **một cách có ý nghĩa thống kê** không?

- H0: hai cách có tỉ lệ sai như nhau.
- H1: khác nhau.
- Không đoán trước chiều nào thắng.

### 6. Thiết lập (viết trước được, không phụ thuộc kết quả)

**6.1 Dữ liệu [Bổ sung, đã kiểm]**
| Thông tin | Giá trị |
|---|---|
| Bộ dữ liệu | UIT-VSFC (Nguyen et al., KSE 2018): phản hồi sinh viên tiếng Việt |
| Nhãn gốc | 0 negative, 1 neutral, 2 positive. Nhóm **bỏ neutral** |
| Split test gốc | 3,166 câu: negative 1,409 · neutral 167 · positive 1,590 |
| Split train gốc | 11,426 câu: negative 5,325 · neutral 458 · positive 5,643 |
| Test của nhóm | **500 câu = 250 negative + 250 positive**, lấy ngẫu nhiên từ split test với seed 42; ID lưu ở `data/test_500_ids.json` |
| Ví dụ few-shot | Lấy từ split **train**: 3 bộ × 3 câu, lưu ở `data/fewshot_pools.json`; có `assert` không trùng câu test |
| Cách tải | `pandas.read_parquet` từ nhánh Parquet trên Hub, vì script loader không chạy với `datasets` ≥ 4.0 |
| Độ tin cậy nhãn | Bài báo báo độ đồng thuận giữa người gán nhãn sentiment khoảng 91%. Vẫn có nhãn bất đồng, nên có nguồn lỗi "nhãn nhiễu" |

**Vì sao cân bằng 250/250:** luôn đoán một nhãn chỉ được 50%, nên có mốc so sánh rõ; accuracy không bị lệch do một nhãn chiếm đa số.

**6.2 Ba bộ ví dụ few-shot [đề xuất]**
| Bộ | Thứ tự nhãn | Ý đồ |
|---|---|---|
| 1 | positive, negative, positive | Dư positive, ví dụ cuối positive |
| 2 | negative, positive, negative | Đối xứng với bộ 1 |
| 3 | Ngẫu nhiên theo seed | Thêm một lần lấy mẫu |

**6.3 Biến cố định và biến thay đổi**
| Cố định | Giá trị |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`; dtype ghi trong `outputs/results/env.json` (`float16` trên T4, lỗi thì `float32`) |
| Decoding | Greedy: `do_sample=False`, `max_new_tokens` nhỏ (khoảng 5) |
| Prompt | Template §4.1 qua chat template của tokenizer |
| Dữ liệu test | 500 câu cố định |
| Seed | 42 |
| Môi trường | Kaggle GPU T4; phiên bản thư viện trong `outputs/results/env.json` |

Biến thay đổi gồm 4 lượt chạy: zero-shot · 3-shot bộ 1 · bộ 2 · bộ 3. Tổng 500 × 4 = 2,000 lần sinh.

**6.4 Parse output → nhãn**
1. Chuẩn hoá: chữ thường, bỏ khoảng trắng thừa và dấu câu.
2. Chỉ có `positive` thì là positive; chỉ có `negative` thì là negative.
3. Có cả hai, hoặc không có nhãn nào, thì là `None` (không parse được).
4. Nhận cả nhãn tiếng Việt: "tích cực" là positive, "tiêu cực" là negative (Toàn chốt 16/11), vì prompt được phép đa dạng. Trên split **validation**, thử nhiều mẫu prompt (nhãn tiếng Anh / tiếng Việt, cách diễn đạt khác nhau), ghi mọi mẫu đã thử và tỉ lệ không parse được, rồi mới chốt một mẫu cho 500 câu test. Ghi rõ trên slide mẫu prompt được dùng và các nhãn được nhận.
5. Mọi output thô lưu vào CSV để kiểm bằng mắt.

### 7. Đánh giá

**7.1 Accuracy.** `accuracy = số câu đúng / 500`, câu `None` tính sai. Báo thêm `tỉ lệ không parse được = số None / 500`.

**7.2 Confusion matrix (2×2 + cột "không parse được")**
| | Dự đoán positive | Dự đoán negative | Không parse được |
|---|---|---|---|
| Thật positive (250) | … | … | … |
| Thật negative (250) | … | … | … |

Dùng để biết model hay nhầm theo chiều nào, ví dụ có thiên về positive không.

**7.3 Sai số do lấy mẫu [Tự tính]**

Nửa độ rộng khoảng tin cậy 95% là `1.96·√(p(1−p)/n)`, với n = 500:

| Accuracy p | ± |
|---|---|
| 50% | 4.4 điểm |
| 70% | 4.0 điểm |
| 80% | 3.5 điểm |
| 90% | 2.6 điểm |

Hai cách chạy trên **cùng câu**, nên kiểm định theo cặp (McNemar) nhạy hơn việc so hai khoảng tin cậy rời.

**7.4 McNemar test [Bổ sung]**

Chỉ những câu mà hai cách cho kết quả **khác nhau** mới nói lên cách nào tốt hơn.

| | 3-shot đúng | 3-shot sai |
|---|---|---|
| **Zero-shot đúng** | a | **b** (chỉ zero-shot đúng) |
| **Zero-shot sai** | **c** (chỉ 3-shot đúng) | d |

- Nếu H0 đúng thì b ≈ c.
- Dùng `statsmodels.stats.contingency_tables.mcnemar(table, exact=True)`. `exact=True` dùng phân phối nhị thức, phù hợp khi b + c nhỏ.
- Vì sao McNemar: Dietterich (1998) so 5 kiểm định. Với thuật toán chỉ chạy được một lần, McNemar có sai lầm loại I chấp nhận được. Nhóm chạy mỗi cấu hình một lần trên một tập test, đúng trường hợp này.
- 3 bộ few-shot là 3 lần kiểm định → **dùng hiệu chỉnh Bonferroni** (Toàn chốt 16/11): ngưỡng 0.05 / 3 ≈ 0.017; bảng kết quả báo cả p chưa hiệu chỉnh và kết luận theo ngưỡng 0.017.

**7.5 Bảng kết quả chính (slide 19), điền sau khi chạy**
| Cách | Accuracy | Không parse được | b | c | p-value McNemar (so với zero-shot) |
|---|---|---|---|---|---|
| Zero-shot | … | … | — | — | — |
| 3-shot bộ 1 | … | … | … | … | … |
| 3-shot bộ 2 | … | … | … | … | … |
| 3-shot bộ 3 | … | … | … | … | … |
| 3-shot trung bình (min–max) | … (… – …) | … | | | |

Dưới bảng ghi nguồn: `outputs/results/icl_metrics.json`, ngày chạy, model, dtype.

### 8. Phân tích lỗi (`docs/experiments/error_analysis_v1.md`)
1. Lọc câu sai trong `icl_zero_shot.csv` và `icl_3shot_set{1,2,3}.csv`.
2. Đọc từng câu, xếp vào một nhóm lỗi, ghi lý do ngắn.
3. Đếm từng nhóm; chọn 1–2 câu thật mỗi nhóm cho slide.
4. Riêng câu `None`: ghi model đã trả gì.

Nhóm lỗi dự kiến **[đề xuất]**: vừa khen vừa chê; mỉa mai, nói ngược; câu quá ngắn; viết tắt, teencode; nhãn gốc có thể sai; lỗi định dạng output. Nhóm thật sẽ quyết định sau khi đọc dữ liệu.

**Chú ý:** ví dụ trên slide phải là câu thật trong CSV. Câu nghi nhãn nhiễu chỉ gọi là "có thể sai".

### 9. Tổng hợp kết quả các phần (Phần 10 → `outputs/results/summary_table.csv`)
Toàn chạy mọi phần trong cùng một lần Save & Run All, nên bảng đọc trực tiếp từ các file kết quả, **không chép từ slide**.

| Mục | Thí nghiệm | Kết quả chính (file trong `outputs/results/`) | Thuyết trình |
|---|---|---|---|
| 3 | Số token câu Việt / Anh, GPT-2 vs Qwen | … (`token_counts.csv`) | Muội |
| 4 | Loss trên câu ví dụ | … (`per_token_loss.csv`) | Phúc |
| 5 | Base vs Instruct | … định tính (`base_vs_instruct.csv`) | Phúc |
| 6 | Các cấu hình decoding | … (`decoding_samples.csv`) | Phúc |
| 7, 10 | Zero-shot vs 3-shot, 500 câu | Accuracy …, p-value … (`icl_metrics.json`) | Toàn |
| 8 | Không RAG vs có RAG | … (`rag_results.csv`) | Muội |
| 9 | Hallucination, bias, ngữ cảnh dài | … (`hallucination_results.csv` …) | Phúc |

### 10. Kết luận (slide 20), khung điền sau khi có số
Viết 3–4 ý, mỗi ý gắn một số hoặc một câu thật:
1. **Về ICL:** "3-shot đạt …% so với zero-shot …%; McNemar p = … → (có / chưa có) khác biệt có ý nghĩa."
2. **Về độ nhạy:** "3 bộ ví dụ cho accuracy từ …% đến …%."
3. **Về lỗi:** "Lỗi tập trung ở … (… câu); … câu không parse được."
4. **Về LLM nói chung, từ các mục khác:** RAG giúp …; model bịa ở …

**Giới hạn phải nói:**
- 1 model 1.5B;
- 1 bộ dữ liệu, 1 tác vụ;
- mỗi cấu hình chạy 1 lần;
- 3 ví dụ;
- prompt tiếng Việt, nhãn tiếng Anh.

Không suy rộng sang "LLM nói chung".

### 11. Lời thoại mục 10 (khoảng 2.5 phút), khung
> **[Slide 19]** "Để kiểm tra few-shot có thật sự giúp hay không, nhóm làm thí nghiệm có kiểm soát trên bộ UIT-VSFC. Nhóm bỏ nhãn trung tính, lấy 500 câu test cân bằng, nên đoán bừa chỉ được khoảng 50%. Chỉ một thứ thay đổi là có ví dụ trong prompt hay không; model, prompt, cách sinh greedy, dữ liệu và seed giữ nguyên. Vì few-shot nhạy với việc chọn ví dụ, nhóm chạy 3 bộ ví dụ lấy từ tập train.
> Kết quả: zero-shot …%, 3-shot từ …% đến …%. Hai cách chạy trên cùng câu nên nhóm dùng McNemar, chỉ xét những câu hai cách cho kết quả khác nhau; p-value … nên …
> [Confusion matrix] Model hay nhầm … [Câu lỗi thật] Câu này … vì …
> **[Slide 20]** Tóm lại … Giới hạn: một model nhỏ, một bộ dữ liệu, mỗi cấu hình chạy một lần."

---

## D. Q&A DỰ KIẾN
| # | Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|---|
| 1 | Few-shot có thay đổi trọng số không? | Không. Ví dụ chỉ nằm trong prompt. Khác fine-tuning ở mục 8 | Brown et al. 2020 |
| 2 | Vậy sao gọi là "learning"? | Model điều chỉnh hành vi theo ngữ cảnh, không theo trọng số; ví dụ cho thấy tập nhãn, kiểu câu, định dạng | Min et al. 2022 |
| 3 | Few-shot có luôn tốt hơn? | Không chắc; phụ thuộc model, tác vụ, ví dụ. Model Instruct có zero-shot đã khá mạnh. Nhóm đo được … | Zhao et al. 2021; FLAN; kết quả nhóm |
| 4 | Vì sao chạy 3 bộ ví dụ? | Thiên lệch nhãn đa số và nhãn cuối; một bộ có thể may hoặc xui | Zhao et al. 2021 |
| 5 | Vì sao McNemar mà không chỉ so accuracy? | Kết quả theo cặp trên cùng 500 câu; McNemar chỉ xét b, c; hợp với cấu hình chỉ chạy một lần | Dietterich 1998 |
| 6 | Chênh 2 điểm có ý nghĩa không? | Sai số khoảng ±4 điểm; phải xem p-value McNemar | §7.3 |
| 7 | Câu không parse được xử lý sao? | Tính là sai, báo riêng tỉ lệ | §7.1 |
| 8 | Sao bỏ neutral? | Đề nhóm làm 2 nhãn; neutral chỉ khoảng 5.3% split test (167/3,166) và khó định ranh giới | Thống kê UIT-VSFC |
| 9 | Ví dụ few-shot có trùng câu test không? | Không: khác split, và có `assert` kiểm trùng | §6.1 |
| 10 | Vì sao greedy? | Để lặp lại được; phải truyền `do_sample=False` vì bản Instruct mặc định lấy mẫu | `generation_config.json` |
| 11 | Khác gì Practice 3? | Practice 3 dùng model phân loại, fine-tune với nhiều dữ liệu nhãn; ở đây không train, chỉ prompt | §3 |
| 12 | ICL có phải "emergent ability"? | Survey xếp vậy, nhưng có tranh cãi là do cách đo. Nhóm không khẳng định, chỉ báo số | Wei et al. 2022; Schaeffer et al. 2023 |
| 13 | Kết quả có áp dụng cho LLM nói chung? | Không suy rộng | §10 |

---

## E. Code Phần 7 và Phần 10 phải giải thích được (trong notebook)
Mỗi hàm định nghĩa ngay trước chỗ dùng. Ngay sau hàm có một cell `assert`, thay cho file test riêng.

| Hàm | Câu cô có thể hỏi | Ý trả lời | `assert` đi kèm |
|---|---|---|---|
| `make_test_split` | Sao cùng seed luôn ra cùng 500 câu? | Bộ sinh ngẫu nhiên khởi tạo bằng seed cố định | Đủ 250/250; chạy lại cho cùng ID |
| `make_fewshot_pools` | Làm sao chắc ví dụ không trùng test? | Lấy từ split train, loại câu trùng nội dung | Giao giữa ví dụ và câu test là rỗng |
| `build_prompt` | Zero-shot khác 3-shot ở đâu trong code? | Cùng một hàm; `examples` rỗng là zero-shot | Prompt zero-shot là phần cuối của prompt 3-shot |
| `run_icl` | Sao chắc là greedy? | `do_sample=False`; in `model.generation_config` | Chạy 2 lần một câu ra cùng output |
| `parse_label` | Output "Positive." thì sao? | Chữ thường, bỏ dấu câu, rồi tìm từ nhãn | `"Positive."`→positive; `"positive negative"`→None |
| `compute_metrics` | `None` tính thế nào? | Tính là sai, báo riêng tỉ lệ | Ví dụ nhỏ tự tính tay |
| `mcnemar_counts` | b, c là gì? | b: chỉ cách A đúng; c: chỉ cách B đúng | a+b+c+d = 500 |
| `mcnemar_pvalue` | Vì sao `exact=True`? | b + c có thể nhỏ, nên dùng nhị thức chính xác | b = c cho p = 1 |

---

## F. Việc của Toàn (code mọi phần + lý thuyết mục 7, 10)
| Ngày | Việc |
|---|---|
| 1 | Toàn duyệt `RULES_v2`, `FOLDER_STRUCTURE_v2` (Thư review); Phần 0 Setup (lấy model, tokenizer, dữ liệu; `env.json`) |
| 2 | Lý thuyết + slide 14, khung slide 19–20 (file này); code Phần 1–3, smoke run, bàn giao cho Muội |
| 3 | Code Phần 4–6, smoke run, bàn giao cho Phúc; code Phần 7 + `assert` |
| 4 | Code Phần 8–9, bàn giao; chốt prompt mục 7 trên split validation; smoke run Phần 7 |
| 5 | Save & Run All chính thức; Phần 10: metric, McNemar, confusion matrix; `error_analysis_v1.md` |
| 6 | Điền số slide 14, 19, 20; xuất HTML dự phòng; kiểm số trên mọi slide khớp `outputs/results/` |
| 7 | Tập trình bày; Thư hỏi chéo 3 cell bất kỳ mỗi người |

Lịch là **[đề xuất]**, ghép với `progress_status.md`.

---

## 12. Tài liệu tham khảo
| Tài liệu | Dùng cho |
|---|---|
| Brown et al., *Language Models are Few-Shot Learners*, 2020. https://arxiv.org/abs/2005.14165 | ICL; zero/one/few-shot |
| Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN), 2021. https://arxiv.org/abs/2109.01652 | Instruction tuning giúp zero-shot |
| Ouyang et al., *Training language models to follow instructions with human feedback*, 2022. https://arxiv.org/abs/2203.02155 | SFT + RLHF |
| Zhao, Wallace, Feng, Klein, Singh, *Calibrate Before Use*, 2021. https://arxiv.org/abs/2102.09690 | Độ nhạy few-shot; 3 thiên lệch |
| Min et al., *Rethinking the Role of Demonstrations*, 2022. https://arxiv.org/abs/2202.12837 | Vai trò của ví dụ |
| Zhao et al., *A Survey of Large Language Models*, 2023. https://arxiv.org/abs/2303.18223 | 3 khả năng nổi lên |
| Wei et al., *Emergent Abilities of Large Language Models*, TMLR 2022. https://arxiv.org/abs/2206.07682 | Định nghĩa emergent |
| Schaeffer, Miranda, Koyejo, *Are Emergent Abilities of Large Language Models a Mirage?*, NeurIPS 2023. https://arxiv.org/abs/2304.15004 | Phản biện emergent |
| Nguyen et al., *UIT-VSFC*, KSE 2018, DOI 10.1109/KSE.2018.8573337 | Dữ liệu; độ đồng thuận nhãn |
| https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback | Số câu, nhãn từng split |
| Dietterich, *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*, Neural Computation 10(7), 1998. DOI 10.1162/089976698300017197 | McNemar |
| statsmodels `mcnemar`: https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html | API |
| https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct (model card, `generation_config.json`) | Cấu hình sinh mặc định; chat template |
| https://huggingface.co/docs/transformers/chat_templating | `apply_chat_template` |
| Slide môn, Ch4 slide 25 (Practice 3, Hugging Face) | So sánh với cách fine-tune phân loại |

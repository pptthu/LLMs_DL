# Mục 5 — Các giai đoạn huấn luyện: Pretraining, SFT, Preference alignment (RLHF/DPO) (nội dung chi tiết, v1)

- Người phụ trách: Phúc | Slide 10–11 | Thời lượng: khoảng 2.5 phút | Notebook: phần 5 (base so với instruct; **chờ cô xác nhận cách demo**)
- Đề của cô: "Pretraining, Supervised Fine-Tuning (SFT) và preference alignment; giới thiệu RLHF/DPO ở mức ý tưởng". Minh hoạ: sơ đồ mục tiêu và dữ liệu của từng giai đoạn.
- Ký hiệu nguồn:
  - **[Slide ChX sN]**: slide môn.
  - **[Bổ sung]**: nguồn ngoài, đã kiểm 2026-11-15 (§9).
- **Phạm vi:** nhóm không huấn luyện. Mục này giải thích các giai đoạn và minh hoạ **kết quả** của chúng bằng cách so bản base với bản instruct của cùng một model.

---

## 0. Mục này cần làm người nghe hiểu được gì
1. Một LLM như ChatGPT hay Qwen-Instruct qua **ba giai đoạn**: mỗi giai đoạn có mục tiêu, dữ liệu và kết quả khác nhau.
2. Pretraining cho **kiến thức và khả năng viết tiếp**; SFT dạy **làm theo chỉ dẫn**; alignment dạy **câu trả lời nào được ưa hơn**.
3. RLHF và DPO là hai cách làm alignment, ở mức ý tưởng.
4. Bằng chứng: base và instruct trả lời cùng một prompt rất khác nhau.

---

## 1. Liên hệ slide môn  [Slide Ch4]
- Ch4 dạy **fine-tuning**: lấy model pre-trained rồi huấn luyện tiếp trên dữ liệu của tác vụ mới.
- SFT và alignment của LLM cũng là fine-tuning theo đúng nghĩa đó: bắt đầu từ model đã pretrain, cập nhật trọng số trên dữ liệu nhỏ hơn, chuyên biệt hơn.
- Điểm khác: dữ liệu là hội thoại hoặc cặp so sánh, không phải ảnh có nhãn lớp.

---

## 2. Ba giai đoạn

### 2.1 Bảng tổng (sơ đồ slide 10)

| | 1. Pretraining | 2. SFT (Supervised Fine-Tuning) | 3. Preference alignment |
|---|---|---|---|
| **Mục tiêu** | Đoán token tiếp theo trên văn bản thô | Học cách trả lời theo chỉ dẫn | Ưu tiên câu trả lời được người đánh giá ưa hơn |
| **Dữ liệu** | Văn bản không nhãn khổng lồ (web, sách, code) | Cặp (chỉ dẫn → câu trả lời mẫu) do người viết | Cặp so sánh: cùng prompt, câu tốt hơn vs câu kém hơn |
| **Loss / cách học** | Cross-entropy trên mọi token (mục 4) | Cross-entropy, thường chỉ tính trên phần câu trả lời | RLHF (reward model + PPO) hoặc DPO |
| **Quy mô (Qwen2.5)** | 18 nghìn tỉ token | Hơn 1 triệu mẫu | DPO khoảng 150 nghìn cặp + RL trực tuyến (GRPO) |
| **Kết quả** | **Base model**: viết tiếp giỏi, chưa biết "trả lời" | Biết làm theo chỉ dẫn | Hữu ích, trung thực, an toàn hơn |
| **Chi phí** | Rất lớn (phần lớn tổng chi phí) | Nhỏ hơn nhiều | Nhỏ hơn nhiều |

```
Văn bản thô ──► [1. Pretraining] ──► Base model ──► [2. SFT] ──► SFT model ──► [3. Alignment] ──► Instruct / Chat model
 (nghìn tỉ token)   đoán token tiếp     viết tiếp     (chỉ dẫn → trả lời)          (tốt hơn vs kém hơn)
```

### 2.2 Pretraining
- Đã trình bày ở mục 4: next-token prediction, cross-entropy, dữ liệu khổng lồ.
- **Qwen2.5:** 18 nghìn tỉ token, dữ liệu được lọc và cân lại tỉ lệ; pretraining chia nhiều giai đoạn (Qwen2.5 Technical Report).
- **Base model biết gì:** kiến thức, ngữ pháp, khả năng viết tiếp theo văn phong. Nhưng hỏi nó một câu, nó có thể viết tiếp thành câu hỏi khác thay vì trả lời, vì trên web, câu hỏi thường được nối bằng câu hỏi khác.
- Model card của Qwen2.5-1.5B (base) ghi không khuyến nghị dùng bản base để hội thoại.

### 2.3 SFT (Supervised Fine-Tuning)  [Bổ sung]
- **InstructGPT** (Ouyang et al. 2022): người gán nhãn viết **câu trả lời mẫu** cho prompt, rồi GPT-3 được fine-tune trên đó. Tập SFT khoảng 13 nghìn prompt.
- **ChatGPT** (OpenAI 2022): bước đầu là SFT trên hội thoại do người huấn luyện viết, đóng cả vai người dùng lẫn trợ lý.
- **Qwen2.5:** SFT trên hơn 1 triệu mẫu, 2 epoch, độ dài chuỗi 32,768 token.
- **Loss vẫn là cross-entropy** như pretraining. Khác ở dữ liệu, và phần chỉ dẫn thường được gán nhãn `-100` để chỉ tính loss trên câu trả lời (cách dùng `-100` ở mục 4).
- **Liên hệ mục 7:** FLAN (Wei et al. 2021) cho thấy fine-tune trên nhiều tác vụ viết dạng chỉ dẫn giúp zero-shot trên tác vụ chưa gặp tốt hơn rõ.

### 2.4 Preference alignment
**Vì sao cần sau SFT:**
- Một chỉ dẫn có nhiều câu trả lời "đúng" nhưng chất lượng khác nhau (rõ ràng, trung thực, an toàn).
- Viết câu trả lời mẫu hoàn hảo thì khó, còn **so sánh hai câu xem câu nào tốt hơn thì dễ** hơn.
- Vì vậy dữ liệu alignment là **cặp so sánh**.

#### RLHF (Reinforcement Learning from Human Feedback)  [Bổ sung: Ouyang et al. 2022]
1. Lấy nhiều câu trả lời của model cho một prompt; **người xếp hạng** chúng.
2. Huấn luyện **reward model (RM)**: một model chấm điểm, học dự đoán câu nào người sẽ thích hơn. Tập RM khoảng 33 nghìn prompt.
3. Dùng học tăng cường **PPO** tối ưu model SFT để **tăng điểm RM**. Tập PPO khoảng 31 nghìn prompt, không cần nhãn người.
4. **Phạt KL** theo từng token để model không trôi quá xa model SFT, tránh "lách" reward model.

Kết quả đáng nhớ: người gán nhãn **thích output của InstructGPT 1.3 tỉ tham số hơn GPT-3 175 tỉ**, dù nhỏ hơn hơn 100 lần.

Cái giá — "**alignment tax**": sau alignment, điểm trên một số benchmark công khai giảm (ví dụ SQuAD, DROP, HellaSwag, dịch Pháp–Anh). Trộn thêm cập nhật trên dữ liệu pretraining (PPO-ptx) giảm được phần lớn mức giảm này.

#### DPO (Direct Preference Optimization)  [Bổ sung: Rafailov et al. 2023]
- **Bỏ hai bước** "huấn luyện reward model riêng" và "chạy học tăng cường".
- Huấn luyện thẳng model bằng một **loss kiểu phân loại** trên cặp so sánh: tăng xác suất tương đối của câu được chọn so với câu bị loại. Xác suất của chính model đóng vai reward ngầm.
- **Dữ liệu:** mỗi mẫu gồm (prompt, câu được chọn, câu bị loại).
- **Model tham chiếu** (thường là model SFT) và hệ số **β** giữ model không đi quá xa; β lớn thì bám tham chiếu chặt hơn. Bài báo dùng β = 0.1 mặc định.
- **Kết quả:** trên các thí nghiệm của bài (điều khiển cảm xúc, tóm tắt, hội thoại), DPO bằng hoặc tốt hơn PPO mà đơn giản và rẻ hơn.

#### So sánh RLHF và DPO (bảng cho slide)
| | RLHF (PPO) | DPO |
|---|---|---|
| Dữ liệu | Xếp hạng / cặp so sánh của người | Cặp (chọn, loại) |
| Reward model riêng | Có | Không (ngầm trong model) |
| Học tăng cường, sinh mẫu khi train | Có (PPO) | Không |
| Giữ gần model gốc | Phạt KL | Model tham chiếu + β |
| Độ phức tạp | Cao | Thấp |

### 2.5 Qwen2.5 dùng gì  [Bổ sung: Qwen2.5 Technical Report]
1. SFT: hơn 1 triệu mẫu.
2. **Offline RL bằng DPO:** khoảng 150 nghìn cặp so sánh, 1 epoch, learning rate 7×10⁻⁷.
3. **Online RL bằng GRPO:** dùng reward model chấm theo nhiều tiêu chí (trung thực, hữu ích, ngắn gọn, liên quan, vô hại, giảm thiên lệch); lấy 8 câu trả lời cho mỗi truy vấn.

→ Model nhóm dùng (`Qwen2.5-1.5B-Instruct`) đã qua **cả ba giai đoạn**, và kết hợp cả DPO lẫn RL. GRPO không cần giải thích chi tiết; chỉ nói "một biến thể học tăng cường".

---

## 3. Minh hoạ: base so với instruct (slide 11, notebook phần 5)  [chờ cô xác nhận]

### 3.1 Thiết kế  [đề xuất — Phúc chốt]
- Cùng bộ prompt `data/test_sets/base_vs_instruct.json`, khoảng 4–6 prompt, chạy trên:
  - `Qwen/Qwen2.5-1.5B` (base): đưa **văn bản thô**, vì bản base không được huấn luyện với chat template;
  - `Qwen/Qwen2.5-1.5B-Instruct`: định dạng bằng **chat template**, đúng cách dùng.
- **Cùng cấu hình sinh, truyền tường minh:** `do_sample=False`, cùng `max_new_tokens`. Hai bản có `generation_config` mặc định khác nhau: instruct mặc định lấy mẫu, base mặc định greedy. Không truyền tường minh thì so sánh không công bằng.
- In `model.generation_config` của cả hai vào notebook làm bằng chứng.

### 3.2 Gợi ý loại prompt
| Loại | Prompt mẫu | Điều kỳ vọng quan sát (kiểm lại bằng output thật) |
|---|---|---|
| Câu hỏi | "Thủ đô của Pháp là gì?" | Base có thể viết tiếp thành câu hỏi khác hoặc trắc nghiệm; instruct trả lời thẳng |
| Chỉ dẫn | "Liệt kê 3 lợi ích của việc tập thể dục." | Base có thể lặp lại hoặc lan man; instruct liệt kê đúng 3 ý |
| Yêu cầu định dạng | "Trả lời đúng một từ: 2 + 2 bằng mấy?" | Instruct tuân theo định dạng tốt hơn |
| Yêu cầu không phù hợp (nhẹ, an toàn) | Ví dụ xin lời khuyên gian lận thi cử | Instruct có xu hướng từ chối hoặc khuyên ngược; base có thể viết tiếp. Nhóm chọn prompt vô hại, chỉ để minh hoạ alignment |

**Không viết trước kết quả lên slide.** Cột "kỳ vọng" chỉ để biết cần nhìn gì; nếu output thật khác kỳ vọng thì ghi đúng như thật và phân tích.

### 3.3 Cell "Phân tích" (khung)
> "Với cùng prompt … , bản base … (viết tiếp / lặp / không trả lời), còn bản instruct … . Khác biệt này đến từ SFT và alignment: bản instruct đã học từ hàng triệu cặp chỉ dẫn – câu trả lời và cặp so sánh. Cả hai cùng kiến trúc, cùng số tham số, chỉ khác giai đoạn huấn luyện."

---

## 4. Nội dung slide

### Slide 10 — Ba giai đoạn huấn luyện
- **Tiêu đề:** 5. Các giai đoạn huấn luyện LLM
- **Giữa:** sơ đồ 3 khối nối nhau (Pretraining → SFT → Alignment). Mỗi khối 3 dòng: **Mục tiêu / Dữ liệu / Kết quả** (lấy từ bảng §2.1).
- **Dưới trái:** RLHF: người xếp hạng → reward model → PPO (+ phạt KL).
- **Dưới phải:** DPO: học thẳng từ cặp (chọn, loại); không reward model, không RL.
- **Góc:** "Qwen2.5-Instruct: 18T token pretraining · SFT > 1 triệu mẫu · DPO ~150 nghìn cặp + RL trực tuyến".
- **Chân:** Ouyang et al. 2022; Rafailov et al. 2023; Qwen2.5 report

### Slide 11 — Demo base so với instruct
- **Tiêu đề:** Cùng kiến trúc, khác giai đoạn huấn luyện
- **Bảng 3 cột:** Prompt | Qwen2.5-1.5B (base) | Qwen2.5-1.5B-Instruct. Output thật, rút gọn.
- **Dưới:** 1–2 câu phân tích (§3.3).
- **Chân:** greedy, cùng `max_new_tokens` · Notebook phần 5, cell …

---

## 5. Lời thoại (khoảng 2.5 phút)

> **[Slide 10]** "Một chatbot như ChatGPT hay model Qwen Instruct nhóm dùng đi qua ba giai đoạn.
> Thứ nhất là pretraining: đoán token tiếp theo trên văn bản thô khổng lồ; Qwen2.5 học 18 nghìn tỉ token. Kết quả là base model, biết rất nhiều và viết tiếp rất giỏi, nhưng chưa biết 'trả lời'.
> Thứ hai là SFT: fine-tune trên các cặp chỉ dẫn và câu trả lời mẫu do người viết. Loss vẫn là cross-entropy, chỉ khác dữ liệu. Sau bước này model biết làm theo chỉ dẫn.
> Thứ ba là preference alignment. So sánh hai câu trả lời thì dễ hơn viết câu mẫu hoàn hảo, nên dữ liệu ở đây là cặp: câu tốt hơn và câu kém hơn. RLHF dùng các so sánh đó để train một reward model chấm điểm, rồi dùng học tăng cường PPO để model tăng điểm. DPO đơn giản hơn: học thẳng từ cặp so sánh, không cần reward model riêng và không cần học tăng cường. Đáng nhớ là InstructGPT 1.3 tỉ tham số được người dùng thích hơn GPT-3 175 tỉ.
> **[Slide 11]** Để thấy tác dụng, nhóm cho cùng prompt vào bản base và bản instruct của Qwen 1.5B, cùng cách sinh greedy … [đọc output thật]. Hai model cùng kiến trúc, cùng số tham số; khác biệt hoàn toàn đến từ SFT và alignment."

---

## 6. Q&A dự kiến

| # | Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|---|
| 1 | SFT khác pretraining ở đâu? | Cùng cross-entropy; khác dữ liệu (cặp chỉ dẫn – trả lời do người viết, ít hơn rất nhiều) và thường chỉ tính loss trên câu trả lời | Ouyang 2022; Qwen2.5 report |
| 2 | Vì sao cần alignment sau SFT? | Nhiều câu trả lời đúng nhưng chất lượng khác nhau; so sánh dễ hơn viết mẫu hoàn hảo; tối ưu theo sở thích của người | Ouyang 2022 |
| 3 | Reward model là gì? | Model học từ xếp hạng của người để chấm điểm câu trả lời; dùng làm phần thưởng cho PPO | Ouyang 2022 |
| 4 | DPO khác RLHF thế nào? | Không reward model riêng, không học tăng cường; một loss kiểu phân loại trên cặp (chọn, loại) | Rafailov 2023 |
| 5 | Vì sao phạt KL / có model tham chiếu? | Giữ model không trôi quá xa model SFT, tránh khai thác lỗi của reward model | Ouyang 2022; Rafailov 2023 |
| 6 | Alignment có làm model "dở" đi không? | Có thể giảm điểm vài benchmark ("alignment tax"); trộn dữ liệu pretraining (PPO-ptx) giảm được phần lớn | Ouyang 2022 |
| 7 | Qwen2.5-Instruct dùng RLHF hay DPO? | Cả hai kiểu: DPO offline khoảng 150 nghìn cặp, sau đó RL trực tuyến (GRPO) với reward model | Qwen2.5 report |
| 8 | Vì sao nhóm không tự làm SFT/DPO? | Cô góp ý không cần fine-tune; demo base vs instruct cho thấy kết quả của các giai đoạn trên model thật | Góp ý 11/11 |
| 9 | Liên hệ với fine-tuning ở Ch4? | SFT và alignment là fine-tuning: bắt đầu từ model pretrained, cập nhật trọng số trên dữ liệu chuyên biệt | Slide Ch4 |

---

## 7. Lưu ý khi làm notebook phần 5
- Nạp 2 model cùng lúc: fp16 khoảng 3.1 GB mỗi model, vừa T4 16 GB. Hoặc nạp lần lượt, rồi `del model` và `torch.cuda.empty_cache()`.
- Phải truyền `do_sample=False` cho **cả hai** model (§3.1).
- Bản base: **không** dùng chat template. Ghi rõ điều này trong cell markdown để cô không hỏi "so sánh không công bằng".
- Chọn prompt vô hại cho minh hoạ an toàn; không dùng nội dung nhạy cảm.

---

## 8. Tài liệu tham khảo cho mục 5

| Tài liệu | Dùng cho |
|---|---|
| Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), 2022. https://arxiv.org/abs/2203.02155 | 3 bước SFT → RM → PPO; số prompt; 1.3B thắng 175B; KL; alignment tax |
| Rafailov et al., *Direct Preference Optimization*, 2023. https://arxiv.org/abs/2305.18290 | DPO; dữ liệu cặp; model tham chiếu, β |
| Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN), 2021. https://arxiv.org/abs/2109.01652 | Instruction tuning |
| OpenAI, *Introducing ChatGPT*, 2022. https://openai.com/blog/chatgpt/ | SFT trên hội thoại; RLHF như InstructGPT |
| Qwen Team, *Qwen2.5 Technical Report*, 2024. https://arxiv.org/abs/2412.15115 | 18T token; SFT > 1 triệu mẫu; DPO ~150 nghìn cặp; GRPO |
| Model card Qwen2.5-1.5B và -Instruct. https://huggingface.co/Qwen | Base không khuyến nghị hội thoại; `generation_config` |
| Slide môn Ch4 | Fine-tuning |

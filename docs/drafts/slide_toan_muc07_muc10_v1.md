# Nội dung slide phần của Toàn — Slide 14, 19, 20 (v1)

> Chữ trong mỗi khung là chữ **đặt trên slide**, chép nguyên vào PowerPoint. Phần "Ghi chú người nói" chép vào speaker notes. Ô `…` chỉ điền từ `outputs/results/` của lần Save & Run All chính thức.

---

## SLIDE 14 — Mục 7. Khả năng của LLM

**Tiêu đề:**
> 7. Khả năng của LLM: học trong ngữ cảnh (In-context learning)

**Khối trái — Ba khả năng:**
> **In-context learning (ICL)**
> Mô hình thực hiện tác vụ mới chỉ nhờ chỉ dẫn và ví dụ đặt trong prompt, **không cập nhật trọng số**.
> • Zero-shot: prompt chỉ có chỉ dẫn
> • Few-shot: chỉ dẫn + k ví dụ có nhãn (nhóm dùng k = 3)
>
> **Làm theo chỉ dẫn**
> Có được nhờ instruction tuning / SFT ở giai đoạn hậu huấn luyện (mục 5) → nhóm dùng bản Instruct.
>
> **Một mô hình, nhiều tác vụ**
> Mọi tác vụ đều là "văn bản vào → văn bản ra". Phân loại cảm xúc = mô hình sinh ra chữ *positive* hoặc *negative*.

**Khối phải — Hai prompt trên cùng tác vụ** (hai cột; tô màu khối ví dụ ở cột 3-shot):

> **Zero-shot**
> ```
> Phân loại cảm xúc của câu phản hồi sinh viên sau.
> Chỉ trả lời một từ: positive hoặc negative.
>
> Câu: "<câu cần phân loại>"
> Cảm xúc:
> ```

> **3-shot**
> ```
> Phân loại cảm xúc của câu phản hồi sinh viên sau.
> Chỉ trả lời một từ: positive hoặc negative.
>
> Câu: "<ví dụ 1>"   Cảm xúc: positive
> Câu: "<ví dụ 2>"   Cảm xúc: negative
> Câu: "<ví dụ 3>"   Cảm xúc: positive
>
> Câu: "<câu cần phân loại>"
> Cảm xúc:
> ```
> *Khác nhau duy nhất: 3 ví dụ (tô màu)*

Ba ví dụ thay bằng câu thật trong `data/splits/fewshot_pools.json`, bộ 1.

**Khối dưới — Output thật:**

> | Câu (UIT-VSFC) | Nhãn thật | Zero-shot | 3-shot |
> |---|---|---|---|
> | "…" | … | … | … |
> | "…" | … | … | … |
> | "…" | … | … | … |
>
> Vài câu chưa đủ để kết luận → thực nghiệm có kiểm soát trên 500 câu (slide 19).

**Chân slide (chữ nhỏ):**
> Lưu ý: kết quả few-shot nhạy với việc chọn và thứ tự ví dụ [Zhao 2021] → nhóm thử 3 bộ ví dụ.
> Nguồn: Brown et al. 2020; Wei et al. 2021; Ouyang et al. 2022 · Qwen2.5-1.5B-Instruct, greedy · Notebook phần 7, cell …

**Ghi chú người nói (khoảng 1 phút):**
> LLM có khả năng gọi là in-context learning: chỉ cần đưa chỉ dẫn hoặc vài ví dụ vào prompt là mô hình làm được tác vụ mới, mà không phải huấn luyện lại hay thay đổi trọng số nào. Zero-shot là chỉ đưa chỉ dẫn; few-shot là thêm vài ví dụ có nhãn, nhóm dùng 3 ví dụ. Các ví dụ không dạy thêm kiến thức, mà cho mô hình thấy tác vụ là gì và phải trả lời theo định dạng nào. Còn khả năng làm theo chỉ dẫn đến từ giai đoạn SFT ở mục 5.
> Bên phải là hai prompt trên cùng tác vụ phân loại cảm xúc; chỉ khác nhau ở 3 ví dụ. Bên dưới là output thật: … Nhưng vài câu thì chưa nói được gì, nên nhóm làm thực nghiệm có kiểm soát trên 500 câu, em trình bày ở phần cuối.

---

## SLIDE 19 — Mục 10. Thực nghiệm có kiểm soát

**Tiêu đề:**
> 10. Thực nghiệm: Zero-shot so với 3-shot trên 500 câu UIT-VSFC

**Dòng câu hỏi (dưới tiêu đề):**
> **Câu hỏi:** Thêm 3 ví dụ vào prompt có làm độ chính xác thay đổi có ý nghĩa thống kê không?

**Khối trái — Thiết lập:**

> | | |
> |---|---|
> | **Dữ liệu** | UIT-VSFC (phản hồi sinh viên), bỏ nhãn neutral |
> | **Tập đánh giá** | 500 câu từ tập test: 250 positive + 250 negative, seed 42 |
> | **Mô hình** | Qwen2.5-1.5B-Instruct · Kaggle GPU T4 |
> | **Sinh văn bản** | Greedy (`do_sample=False`) — kết quả lặp lại được |
> | **Biến thay đổi** | Chỉ có / không có ví dụ: zero-shot · 3-shot × 3 bộ ví dụ (lấy từ tập train, không trùng câu test) |
> | **Đánh giá** | Accuracy · Confusion matrix · % output không parse được · McNemar test |

**Khối phải — Kết quả:**

> | Cấu hình | Accuracy | Không parse được | b | c | p (McNemar) |
> |---|---|---|---|---|---|
> | Zero-shot | …% | …% | — | — | — |
> | 3-shot bộ 1 | …% | …% | … | … | … |
> | 3-shot bộ 2 | …% | …% | … | … | … |
> | 3-shot bộ 3 | …% | …% | … | … | … |
>
> b = số câu chỉ zero-shot đúng · c = số câu chỉ 3-shot đúng

**Khối dưới trái — Hình:**
> [Confusion matrix zero-shot | Confusion matrix 3-shot — chèn `outputs/figures/confusion_matrix_*.png`]

**Khối dưới phải — Ví dụ lỗi (câu thật):**
> **Ví dụ lỗi:** "…"
> Nhãn thật: … · Zero-shot: … · 3-shot: …
> → Lỗi do: …

**Chân slide (chữ nhỏ):**
> Mốc đoán ngẫu nhiên ≈ 50% · Sai số do lấy mẫu với 500 câu ≈ ±4 điểm % · McNemar dùng vì hai cấu hình chạy trên cùng câu [Dietterich 1998]
> Nguồn số liệu: `outputs/results/icl_metrics.json` (chạy ngày …) · Dữ liệu: Nguyen et al., KSE 2018 · Notebook phần 7, 10

**Ghi chú người nói (khoảng 2 phút):**
> Để kiểm tra few-shot có thật sự giúp hay không, nhóm làm một thực nghiệm có kiểm soát. Tác vụ là phân loại cảm xúc phản hồi sinh viên, bộ UIT-VSFC. Nhóm bỏ nhãn trung tính, lấy 500 câu từ tập test, cân bằng 250 tích cực và 250 tiêu cực, nên đoán bừa chỉ được khoảng 50%.
> Nhóm chỉ thay đổi một thứ là có ví dụ trong prompt hay không; mô hình, prompt, cách sinh greedy, dữ liệu và seed đều giữ nguyên. Vì few-shot nhạy với việc chọn ví dụ, nhóm chạy 3 bộ ví dụ khác nhau, lấy từ tập train để không trùng câu test.
> Kết quả: zero-shot đạt …%, ba bộ 3-shot đạt …%, …%, …%. Với 500 câu, sai số khoảng 4 điểm, nên nhìn accuracy thôi là chưa đủ. Vì hai cách chạy trên cùng câu, nhóm dùng McNemar: chỉ xét những câu hai cách cho kết quả khác nhau. Có … câu chỉ zero-shot đúng và … câu chỉ 3-shot đúng, p-value = …, nên …
> Ma trận nhầm lẫn cho thấy mô hình hay nhầm … Ví dụ câu này … mô hình đoán sai vì …

---

## SLIDE 20 — Kết luận

**Tiêu đề:**
> Kết luận

**Nội dung (4 ý; câu trong `…` viết lại theo đúng kết quả):**
> **1. In-context learning:** 3-shot đạt …% so với zero-shot …% (McNemar p = …) → … khác biệt có ý nghĩa thống kê trên tác vụ này.
>
> **2. Độ nhạy với ví dụ:** ba bộ ví dụ cho accuracy từ …% đến …% → kết quả few-shot phụ thuộc vào ví dụ được chọn.
>
> **3. Lỗi của mô hình:** tập trung ở câu … (… câu); …% output không đúng định dạng nhãn.
>
> **4. Khả năng và hạn chế của LLM:** … (một ý rút từ mục 8 RAG) · … (một ý rút từ mục 9 hallucination)

**Khối dưới — Giới hạn:**
> **Giới hạn:** 1 mô hình 1.5 tỉ tham số · 1 bộ dữ liệu · 1 tác vụ · mỗi cấu hình chạy 1 lần · chỉ thử 3 ví dụ → không suy rộng cho mọi LLM.

**Chân slide:**
> Toàn bộ số liệu: `outputs/results/summary_table.csv` · Notebook phần 10

**Ghi chú người nói (khoảng 30 giây):**
> Tóm lại, … [đọc 4 ý]. Kết quả này chỉ đúng với một mô hình nhỏ, một bộ dữ liệu và một tác vụ, mỗi cấu hình chạy một lần, nên nhóm không suy rộng cho mọi LLM. Em xin cảm ơn thầy cô và các bạn đã lắng nghe.

---

## Phương án nếu slide 19 quá chật: tách làm 2 slide

**Slide 19a — Thiết lập thực nghiệm:** câu hỏi + bảng thiết lập + hai prompt rút gọn.

**Slide 19b — Kết quả và phân tích lỗi:** bảng kết quả + 2 confusion matrix + 2 ví dụ lỗi.

Tách thì thêm khoảng 30 giây; lấy bớt từ slide 12 theo kế hoạch nhóm.

# Phân tích lỗi — zero-shot vs few-shot trên UIT-VSFC (v2)

- **Created**: 2026-11-19 (Toàn, AI hỗ trợ) · Trạng thái: **Duyệt: Toàn, 2026-11-19** (AI gán nháp, Toàn đọc lại từng câu và sửa)
- Lần chạy: `20261121_010447_normal` (Kaggle T4); số liệu trùng run cũ `20261119_150356_normal` (bản v1, giữ nguyên). Số liệu lấy từ `outputs/results/icl_metrics.json`, `icl_zero_shot.csv`, `icl_few_shot.csv`, `error_analysis_sample.csv`.
- Code: notebook phần 10, cell 10.1–10.3. Nhóm lỗi gán trong `MANUAL_ERROR_GROUPS` (cell 10.3).

## 1. Thiết lập

| Thông tin | Giá trị |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, float16, revision `989aa79…` |
| Dữ liệu | UIT-VSFC test, bỏ neutral: 2,999 câu (1,590 positive, 1,409 negative) |
| Prompt | Mẫu `vi_nhan_vi` (chỉ dẫn và nhãn tiếng Việt), chọn trên 200 câu validation |
| Sinh | Greedy, `max_new_tokens = 10`, `repetition_penalty = 1.0` |
| Biến thay đổi | Có hay không 4 ví dụ few-shot (2 positive, 2 negative, lấy từ train, seed 42) |

## 2. Kết quả

| Cách | Accuracy (±95%) | Sai | Negative → positive | Positive → negative | Không parse được |
|---|---|---|---|---|---|
| Zero-shot | 89.86% (±1.08) | 304 | 248 | 56 | 0 |
| Few-shot | 92.30% (±0.95) | 231 | 145 | 86 | 0 |

McNemar exact: a = 2,648, b = 47 (chỉ zero-shot đúng), c = 120 (chỉ few-shot đúng), d = 184, **p = 1.5e-08** → few-shot tốt hơn, khác biệt có ý nghĩa thống kê ở mức 0.05.

## 3. Nhóm lỗi (30 câu few-shot sai, lấy ngẫu nhiên seed 42)

Nhóm ghi sẵn **trước** khi đọc output (file chốt khung v4 §2.4).

| Nhóm | Số câu | Ví dụ thật (nhãn gốc → model) |
|---|---|---|
| (a) không parse được | 0 | — |
| (b) nhãn gốc có vẻ sai | 2 | "môn học cần thiết và thiết thực với ngành học ." (negative → tích cực) |
| (c) vừa khen vừa chê | 4 | "thầy dạy hơi khó hiểu nhưng rất kỹ ." (positive → tiêu cực) |
| (d) ý ngầm (góp ý, phàn nàn gián tiếp) | 20 | "giảm thời lượng lý thuyết , tăng thời lượng thực hành ." (negative → tích cực) |
| (e) model sai rõ ràng | 4 | "thầy rất dễ tính ." (positive → tiêu cực) |

Chi tiết từng câu và lý do: `outputs/results/error_analysis_sample.csv` (có cột `error_group`, `note`).

## 4. Nhận xét

1. **Lỗi chính là câu góp ý.** UIT-VSFC gán nhãn negative cho các câu góp ý ("nên…", "cần…", "… nhiều hơn"), vì góp ý ngầm chỉ ra điều chưa tốt. Model đọc giọng văn lịch sự, mang tính xây dựng nên đoán tích cực.
   - Trên toàn bộ câu sai, câu negative bị đoán positive có 189/248 (76%, zero-shot) và 117/145 (81%, few-shot) là câu góp ý (có "nên", "cần", "mong", "thêm", "hơn"; đếm bằng regex ở cell 10.3, nên chỉ là ước lượng).
2. **Vì sao few-shot giúp:** 2 ví dụ negative trong prompt đều là góp ý ("oop khá phức tạp nên cần nhiều ví dụ .", "giảng viên cần linh hoạt …"), nên model thấy được quy ước gán nhãn của bộ dữ liệu. Lỗi negative → positive giảm từ 248 xuống 145. Đổi lại, lỗi positive → negative tăng từ 56 lên 86: model bắt đầu coi một số câu khen ngắn hoặc mơ hồ là góp ý.
3. **Phụ thuộc dữ liệu (liên hệ mục 9):** nhóm (b) và (d) cho thấy "đúng/sai" phụ thuộc cách bộ dữ liệu định nghĩa nhãn, không chỉ phụ thuộc model.
4. **Định dạng output:** model thường trả lời "Cảm xúc: tích cực" (lặp lại nhãn trường trong prompt), có một câu ra "ti tích cực". Luật parse vẫn đọc được, không câu nào bị tính "không parse được".

## 5. Giới hạn

- 1 model 1.5B; 1 bộ dữ liệu, 1 tác vụ; 1 bộ ví dụ few-shot; mỗi cấu hình chạy 1 lần.
- 30 câu là mẫu nhỏ; tỉ lệ các nhóm chỉ mang tính minh họa.
- Nhóm lỗi do một người (Toàn) duyệt, chưa có người thứ hai đối chiếu.

# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Khi temperature thấp như 0.0 và 0.5, câu trả lời khá ổn định, có cấu trúc giống nhau và thường nhắc đến hang Sơn Đoòng. Khi tăng lên 1.0 và 1.5, nội dung bắt đầu đa dạng hơn, ví dụ chuyển sang hạt điều hoặc hạt tiêu, cách diễn đạt cũng khác hơn. Tuy nhiên temperature cao không nhất thiết luôn dài hơn hay tốt hơn; nó chủ yếu làm phản hồi bớt dự đoán được và có thể thay đổi ý tưởng nhiều hơn.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ chọn temperature 0.2 cho chatbot hỗ trợ khách hàng. Vì task này cần câu trả lời ổn định, nhất quán và chính xác hơn là một câu trả lời sáng tạo. Temperature thấp giúp giảm khả năng nói lan man hoặc đưa ra thông tin không cần thiết.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Với workload này, GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần nếu chỉ tính output token. Cụ thể, 10.000 người dùng * 3 lần gọi * 350 token = khoảng 10,5 triệu output token/ngày, tương đương khoảng $105/ngày với GPT-4o và $6.30/ngày với GPT-4o-mini. GPT-4o xứng đáng dùng cho các tác vụ phức tạp cần suy luận tốt, độ chính xác cao hoặc xử lý yêu cầu quan trọng; còn GPT-4o-mini phù hợp cho chatbot FAQ, tóm tắt đơn giản hoặc các tác vụ số lượng lớn cần tiết kiệm chi phí.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Hai phản hồi khác nhau rõ rệt theo persona trong system prompt. Với vai “elementary teacher”, model cố gắng dùng ví dụ đơn giản, gần gũi với trẻ em như đồ chơi, lớp học, quyển sổ hoặc Lego, và tránh các thuật ngữ kỹ thuật. Với vai “financial expert”, model chuyển sang hướng chuyên sâu hơn, nhắc đến các khái niệm như cryptographic primitives, consensus, finality, permissioned/permissionless chains, smart contracts và settlement systems. Điều này cho thấy system prompt ảnh hưởng mạnh đến góc nhìn, mức độ chuyên môn, từ vựng và loại ví dụ mà model chọn khi trả lời.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> "Việt Nam là một quốc gia có lịch sử lâu đời, văn hóa đa dạng và thiên nhiên phong phú. Từ miền núi phía Bắc đến đồng bằng sông Cửu Long, mỗi vùng đều có phong tục, ẩm thực và giọng nói riêng. Con người Việt Nam thường được biết đến với sự thân thiện, chăm chỉ và tinh thần cộng đồng mạnh mẽ. Trong thời đại công nghệ, Việt Nam cũng đang phát triển nhanh trong các lĩnh vực như trí tuệ nhân tạo, thương mại điện tử, giáo dục số và khởi nghiệp." 
> *Đoạn văn tôi chọn có 95 từ. Theo count_tokens dùng tiktoken, đoạn này có 111 token, trong khi cách ước lượng số từ / 0.75 cho ra khoảng 126.67 token. Hai con số chênh nhau khoảng 12.37%, trong trường hợp này ước lượng thô cao hơn số token thực tế. Sự khác biệt xảy ra vì token không giống từ: một từ có thể là một token, nhiều token, hoặc một phần của token. Với tiếng Việt, dấu thanh, ký tự Unicode và việc các từ thường gồm nhiều âm tiết cách nhau bằng dấu cách khiến tokenizer có thể tách văn bản khác với cách con người đếm từ, nên số token thường khó ước lượng chính xác chỉ bằng số từ.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất trong các trường hợp câu trả lời dài hoặc người dùng cần cảm giác hệ thống đang phản hồi ngay, ví dụ chatbot tư vấn, trợ lý viết nội dung, giải thích bài học hoặc tóm tắt tài liệu dài. Khi từng phần của câu trả lời xuất hiện dần, người dùng không phải chờ màn hình trống quá lâu và có thể bắt đầu đọc trước khi model hoàn thành toàn bộ phản hồi. Non-streaming phù hợp hơn khi câu trả lời ngắn, khi cần xử lý toàn bộ output trước khi hiển thị, hoặc khi hệ thống cần kiểm tra/validate kết quả đầy đủ như trả JSON, gọi tool, chấm điểm, phân loại hay moderation. Vì vậy streaming cải thiện cảm giác tốc độ, còn non-streaming đơn giản và dễ kiểm soát hơn trong các tác vụ cần kết quả hoàn chỉnh.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp client tự động chờ lâu hơn sau mỗi lần retry thất bại, ví dụ 1 giây, 2 giây, 4 giây, 8 giây, nên giảm áp lực lên API khi hệ thống đang quá tải. So với delay cố định, cách này cho server thêm thời gian phục hồi và giảm số lượng request lặp lại trong thời gian ngắn. Nếu hàng nghìn client cùng retry với delay cố định giống nhau, chúng có thể cùng gửi lại request vào đúng cùng một thời điểm, tạo ra các đợt tải lớn lặp lại và làm tình trạng quá tải nghiêm trọng hơn. Vì vậy exponential backoff, đặc biệt khi kết hợp thêm jitter/ngẫu nhiên hóa thời gian chờ, giúp retry ổn định và thân thiện hơn với hệ thống.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Tôi chọn persona là trợ lý học tập AI cho học sinh sinh viên bắt đầu học tập. System prompt: "Bạn là trợ lý học tập AI thân thiện, kiên nhẫn và thực tế. Hãy trả lời bằng tiếng việt, giải thích từng bước bằng ngôn ngữ đơn giản, ưu tiên ví dụ ngắn và không đưa đáp án quá dài nếu câu hỏi ngắn." Tôi chọn cụm "trả lời bằng tiếng việt" để tránh đổi ngôn ngữ không cần thiết. Tôi cũng chọn "giải thích từng bước" vì mục tiêu của persona giúp sinh viên học sinh mới học có thể hiểu bản chất chứ không chỉ là một câu trả lời nhanh.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất của trợ lý hiện tại là bộ nhớ hội thoại còn ngắn, ví dụ chỉ giữ khoảng 3 lượt gần nhất, nên dễ quên bối cảnh hoặc các thông tin người dùng đã nói trước đó. Điều này làm trợ lý kém hiệu quả khi người dùng học theo một chuỗi dài. Có thể cải thiện bằng cách lưu conversation history dài hơn và tóm tắt phần cũ thành summary memory. Khi triển khai, mỗi khi gọi vượt quá token limit thì hệ thống có thể gọi model tóm tắt các ý quan trong, sau đó lưu bản tóm tắt và gửi kèm summary cùng 1 số lượt chat gần nhất trong prompt tiếp theo.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026

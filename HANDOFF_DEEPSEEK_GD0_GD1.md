# TÀI LIỆU BÀN GIAO KỸ THUẬT — CHATBOT NEU (api8018 + admin_api)

> Người đọc: AI/kỹ sư thực hiện (DeepSeek). Đọc hết trước khi sửa bất kỳ dòng nào.
> Phạm vi giao việc: **GĐ0 và GĐ1**. GĐ2, GĐ3 chỉ để hiểu định hướng, **chưa làm**.
> Thiết kế trong tài liệu này đã chốt, không bàn lại. Chỗ nào ghi "CHƯA XÁC MINH" thì phải tự kiểm tra trong code trước khi dựa vào.

---

## PHẦN 1 — BỐI CẢNH HỆ THỐNG

### 1.1 Kiến trúc

- **2 service Flask** (đang chạy bằng Flask dev server, mở `0.0.0.0`):
  - `api8018.py` — chat **công khai** (`/ask`, `/feedback`, `/metadata`) và các endpoint `/admin/reload-index`, `/admin/search-chunks` nằm **chung cổng công khai**, không xác thực theo IP.
  - `admin_api.py` — quản trị tài liệu, bảng, FAQ, feedback, test. Có `_check_auth()` (dòng 68), khoảng 50 route.
- **Dữ liệu:** `data/raw/` (file gốc), `data/processed/` (txt, markdown, chunks, `conflict/`, `tables/`), `registry.json` (cấu hình bảng).
- **RAG:** FAISS + BM25 hybrid, gộp bằng RRF. Embedding qua Ollama `qwen3-embedding:8b-ctx16k` (`EMBED_MODEL`, dòng 59 api8018.py).
- **LLM:** `qwen2.5:14b-instruct-ctx16k` qua Ollama (biến `CHAT_MODEL`, `AGENT_MODEL` trong `.env`).
- **Matcher:** `multi_entity_matcher.py` đọc `registry.json`, khớp câu hỏi với dòng bảng theo khóa chính.
- **Patterns:** `patterns.py` có `extract_years(text)` (trả về danh sách chuỗi năm; nếu là khoảng "2018-2022" thì trả đủ mọi năm trong khoảng). Regex `YEAR = \b(20[12]\d)\b` khớp 2010–2029.
- **Phiên:** `_sessions` là dict **trong bộ nhớ** của api8018 (`get_session`, dòng 131-136), khóa chỉ là `session_id`.
- **Job nền của admin:** `_jobs` dict **trong bộ nhớ** của admin_api, có `_cleanup_old_jobs` (dòng 205).

### 1.2 File chính và vai trò

| File | Vai trò | Ghi chú |
|---|---|---|
| `api8018.py` | Pipeline trả lời câu hỏi | Đã xem đầy đủ. Kết thúc dòng LF |
| `admin_api.py` | Quản trị tài liệu, bảng, FAQ, feedback, test | Đã xem. **Kết thúc dòng CRLF (`\r\n`) — giữ nguyên khi sửa** |
| `multi_entity_matcher.py` | Khớp câu hỏi ↔ thực thể | **CHƯA XÁC MINH**: người soạn tài liệu chưa đọc file này |
| `patterns.py` | Trích năm, mã ngành, email | Đã xem |
| `conflict_detection.py` | Phát hiện xung đột, rebuild FAISS | Có 5 chỗ `input()` (dòng 816, 1469, 1988, 2003 và chú thích 2040), 0 chỗ bắt `EOFError` |
| `pipeline_pdf.py`, `pipeline_docx_txt.py` | Xử lý tài liệu | Chưa xem, không cần đụng ở GĐ0-1 |
| `structured_data_pipeline.py` | Xử lý bảng CSV/XLSX, có lệnh `update-config-web` | Chưa xem |
| `registry.json` | Cấu hình bảng: cột, nhãn, PII, filter, `extra_instructions` | Có 2 bảng: `nganh`, `giangvien` |
| `admin.html`, `chat.html` | Giao diện | GĐ0-1 chưa cần sửa giao diện (trừ khi ghi rõ) |

### 1.3 Số liệu hiện trạng (từ kiểm tra ngày 2026-09-29)

- `faq_log.json`: 218 mục, 100% trạng thái `chua_sua`, 208 mục không có citation, 9 mục có warning (cả 9 là `missing_numbers`).
- `feedback_log.json`: 1 phản hồi. Manifest: 43 văn bản active.
- `giangvien.csv`: 1054 dòng. Dòng Bùi Đức Thọ (`thobd@neu.edu.vn`): `position = "Bí thư Đảng ủy - Giám đốc Đại học"`. Có 7 dòng chứa "Hiệu trưởng" trong `position`, đều dạng "(trường thành viên)".
- Kho chunk: 146 file. `Quy dinh Dao tao DHCQ theo he thong tin chi QD389.json` chứa 91 chỗ nhắc "đào tạo theo/hệ thống tín chỉ".
- `.env` có các khóa: `AGENT_MODEL, API_AUTH_TOKEN, CHAT_MODEL, EMBED_MODEL, OCR_BACKEND, OLLAMA_VISION_*`. **Không** có biến ngưỡng similarity hay `ADMIN_ALLOWED_CIDRS` (GĐ1/GĐ3 sẽ thêm). **Không in giá trị `.env` ra log/báo cáo.**

---

## PHẦN 2 — 12 VẤN ĐỀ ĐÃ XÁC NHẬN

| # | Vấn đề | Bằng chứng cụ thể | Lớp gây ra |
|---|---|---|---|
| P1 | Dữ liệu giảng viên câu trước dính vào câu độc lập, kéo lệch cả truy xuất RAG | `looks_like_followup` (api8018.py:284-286) là `len(question.split()) <= 6 or any(h in q for h in FOLLOWUP_HINTS)`; `FOLLOWUP_HINTS` ở dòng 281. Kiểm tra: "đào tạo tín chỉ là gì", "học phí k65", "quy trình xét học phí", "Hiệu trưởng là ai", "chuyển ngành có được không" đều bị coi là nối tiếp. Câu "đào tạo tín chỉ là gì" không trích QD389 dù kho có 91 chỗ nhắc | Code chung |
| P2 | Cảnh báo `missing_numbers` nhiễu (9/9 sai) và bỏ sót lỗi thật | Đoạn tạo cảnh báo ở api8018.py ~1590-1620 (`missing = source_numbers - answer_numbers` dòng 1608, `type: "missing_numbers"` dòng 1619). Cả 9 cảnh báo chỉ báo năm học hàm/học vị (2005, 2015, 1982, 2007, 2009) và `<so_dai>`. SĐT `0986426633` bị chuẩn hóa thành `986426633` nên bị coi là "thiếu" | Thiết kế kiểm tra |
| P3 | Câu trả lời không bám dữ liệu | Dữ liệu ghi "Giám đốc Đại học", bot nói "Hiệu trưởng". "Tín chỉ là gì" trích nguồn không liên quan. Kho không có chunk nào chứa "Hiệu trưởng" gần "Bùi Đức Thọ" | Thiếu tri thức do admin xác nhận, thiếu kiểm tra bám nguồn |
| P4 | Admin không thấy nguyên nhân và nguồn | 208/218 mục FAQ không có citation. Nhánh trả lời từ bảng không ghi nguồn | Thiếu trace |
| P5 | Luồng "sửa" chưa đáng tin | Chỉ đẩy văn bản vào RAG; chunk sai cũ vẫn còn (nhật ký có 26 `giu_2_phien_ban`, 17 `web_file_giu_2`, 7 `giu_ca_2_doan`). `conflict_detection.py` có 5 chỗ `input()`, 0 chỗ bắt `EOFError`, trong khi job chạy với `stdin=subprocess.DEVNULL`. **Đây là suy ra từ code, chưa từng chạy thật.** Chưa có mục nào từng đẩy RAG | Thiết kế luồng |
| P6 | Nút và trạng thái khó hiểu. Badge ⚠️ tự bật lại (mô phỏng) | Kiểm tra mục 3: cùng warning mà có người hỏi lại thì badge tính lại thành True dù admin đã xử lý | UI, trạng thái |
| P7 | Không có gì đảm bảo sau khi đổi nội dung | Không có kiểm tra hồi quy, không theo dõi phụ thuộc | Thiếu cơ chế |
| P8 | **Nguy cơ mất log FAQ** | `admin_api.py:1628` `_load_faq`: `except Exception: return {}`; `admin_api.py:1634` `_save_faq` ghi thẳng `write_text` (không ghi nguyên tử). Nếu parse lỗi (ví dụ file đang bị ghi dở bởi tiến trình khác) thì lần lưu kế tiếp **ghi đè toàn bộ log bằng `{}`**. Hai tiến trình (api8018 và admin_api) cùng đọc-sửa-ghi, không khóa | Độ tin cậy |
| P9 | FAQ gộp kém | Khóa chỉ là chuỗi thường. Câu trả lời phụ thuộc phiên vẫn được lưu như câu độc lập | Thiết kế |
| P10 | **BUG ĐÃ XÁC NHẬN**: `update_table_config` không reload api8018 | `admin_api.py:1475` `table_update` sau khi ghi CSV gọi `POST http://127.0.0.1:8018/admin/reload-index` với header `X-API-Key` (dòng ~1495). `admin_api.py:1574` `update_table_config` chạy `structured_data_pipeline.py update-config-web` trong thread `worker()` rồi **kết thúc mà không reload**. Hệ quả: đổi cờ PII hoặc cấu hình bảng qua web không có hiệu lực cho tới khi ai đó reload tay | Lỗi code |
| P11 | `/admin/*` nằm chung cổng công khai; Flask dev server | `/admin/reload-index` (api8018.py:2113) và `/admin/search-chunks` (2122) đọc được nội dung chunk từ ngoài mạng. Nhiều cổng nghe `0.0.0.0`. Log có IP ngoài và bot quét | Vận hành |
| P12 | Quy tắc nghiệp vụ nằm trong code | `FOLLOWUP_HINTS` (281), `PII_KEYWORDS` (1417), `_KEYWORD_TO_COL` (1935) | Khó mở rộng |

### Làm rõ 2 điểm quan trọng

**P10.** Bản sửa phải đặt lệnh reload **ở cuối `worker()` của `update_table_config`, chỉ khi `proc.returncode == 0`**, dùng đúng mẫu `table_update`, và ghi `reload_status` vào `_jobs[job_id]["result"]`. **Không** đặt reload ở handler: việc chính chạy trong thread, reload sớm sẽ nạp `registry.json` cũ.

**P8.** Phải: (1) nếu file tồn tại mà parse lỗi thì **không** trả `{}` rồi ghi tiếp; chuyển file hỏng sang `*.corrupt.<timestamp>`, ghi log lỗi, và từ chối lưu; (2) ghi nguyên tử (ghi file tạm cùng thư mục → `flush` + `os.fsync` → `os.replace`); (3) `fcntl.flock` trên file khóa riêng (`faq_log.json.lock`) cho toàn bộ chu trình đọc-sửa-ghi; (4) áp dụng cùng cơ chế cho `feedback_log.json` và mọi log JSON khác do 2 tiến trình cùng ghi (kể cả phía api8018 — tìm mọi chỗ api8018 ghi `faq_log.json`; theo mô tả trước đó api8018 ghi đè bằng `{}` khi parse lỗi).

---

## PHẦN 3 — THIẾT KẾ ĐÃ CHỐT (KHÔNG BÀN LẠI)

Nguyên tắc: **cái gì admin xác nhận được thì là dữ liệu trên web (có phiên bản), không phải code cứng.** Code chỉ giữ cơ chế chung.

### K1. Nguồn đã duyệt (GĐ2)
- **Mục đích:** admin thấy và xác nhận nguồn thật của câu trả lời (bảng, chunk).
- **Cách hoạt động:** câu trả lời từ bảng có citation kiểu bảng: tên bảng, khóa dòng, các cột đã dùng và giá trị (che PII theo cấu hình hiện hành). Nút Sửa dòng (dùng trình sửa bảng có sẵn) và Xác nhận. Dấu duyệt gắn checksum; dữ liệu đổi thì dấu duyệt tự hết hiệu lực. Hàng đợi ưu tiên: nguồn dùng nhiều nhưng chưa duyệt.
- **Khác code hiện tại:** nhánh bảng hiện không ghi nguồn. GĐ1 chỉ làm phần **trace và citation bảng** (đọc-only), K1 đầy đủ để GĐ2.

### K2. Kho câu trả lời đã duyệt (GĐ2)
- **Mục đích:** "admin duyệt một lần thì không sai lại".
- **Bản ghi:** câu hỏi + biến thể, nội dung, thực thể áp dụng, bộ năm, danh sách phụ thuộc kèm checksum, người duyệt, thời điểm, phiên bản.
- **Điều kiện khớp (tất cả phải đúng):** (a) câu hỏi chuẩn hóa khớp bản ghi, hoặc cosine embedding ≥ `DUYET_SIMILARITY_THRESHOLD`; (b) cùng thực thể (theo matcher); (c) cùng tập năm.
- **RÀNG BUỘC ĐÃ CHỐT:**
  - **K2 chỉ so khớp NĂM, dùng `patterns.extract_years()` có sẵn. KHÔNG so khớp số liệu khác.**
  - **Checksum tính trên CÁC DÒNG ĐÃ DÙNG (theo khóa chính), KHÔNG phải cả bảng.** Chỉ băm các cột trong `display_columns` (bỏ `hidden_columns` và cột rác như `Unnamed: 2`). Câu tổng hợp nhiều dòng (max/min, đếm, liệt kê) dùng checksum các cột liên quan trên **toàn bảng**.
  - **Băm kèm cấu hình ảnh hưởng đầu ra:** `display_columns`, `pii_columns`, `column_value_aliases`, `extra_instructions`.
  - Checksum đổi → bản duyệt tự chuyển "cần duyệt lại" và **KHÔNG được dùng**; câu hỏi rơi về pipeline. Lý do: K2 là cache văn bản cố định, không đọc lại CSV mỗi lần, khác pipeline vốn lấy dữ liệu mới.
  - Che PII hiện hành khi phục vụ bản duyệt.
  - Ngưỡng 0.90 chỉ là **khởi điểm**. **Nếu phân bố cặp dương và cặp âm chồng lấn thì K2 dùng khớp chuẩn hóa chính xác, KHÔNG dùng embedding.**
  - Không đi qua `conflict_detection`; hiệu lực ngay (không rebuild FAISS).

### K3. Bí danh giá trị cột (GĐ2)
- **Mục đích:** giải quyết "Hiệu trưởng" ↔ "Giám đốc Đại học" bằng dữ liệu, không sửa code hay `extra_instructions`.
- **Format:** field `column_value_aliases` trong từng bảng của `registry.json`: `{tên_cột: {từ_khóa: giá_trị_thật}}`. Ví dụ `giangvien.position`: `{"hiệu trưởng": "Giám đốc Đại học"}`.
- **RÀNG BUỘC ĐÃ CHỐT:** khớp theo **ranh giới từ**, **ưu tiên cụm dài** ("phó hiệu trưởng" thắng "hiệu trưởng"), và bí danh **BỔ SUNG giá trị tìm kiếm chứ không THAY THẾ** (7 dòng "Hiệu trưởng (trường thành viên)" vẫn phải được tìm thấy; câu trả lời cho "Hiệu trưởng là ai" nên nêu cả hai nhóm).
- Đưa vào ngữ cảnh LLM; kiểm tra bám nguồn (K5-b) coi là tương đương đã xác nhận.
- Giao diện (GĐ2): panel "Cấu hình cho chatbot" của bảng, mục "Bí danh giá trị cột", bảng thu gọn, mặc định thu gọn, nút thêm/xóa dòng.

### K4. Bộ kiểm tra tự sinh (khởi đầu ở GĐ1, đầy đủ ở GĐ2)
- Mỗi lần duyệt sinh ca dương; "Lỗi hệ thống" sinh ca âm; sẵn ca đa lượt và ca "không lộ prompt hệ thống".
- Assert dựa trên sự kiện (chuỗi phải có/không được có, nguồn phải trích), **không** so khớp nguyên văn, để chịu tính ngẫu nhiên của LLM.
- **Test chạy nền bất đồng bộ; banner đọc từ file JSON trạng thái job, sống qua F5.** Lần đầu ghi "Đang cập nhật, vui lòng chờ...", xong ghi "Đã cập nhật thành công", lỗi ghi rõ lỗi.

### K5. Lõi code chung (không chứa tri thức)

**K5-a. Chính sách dùng lại thực thể từ câu trước (GĐ1)**
- **Bỏ tiêu chí độ dài.**
- Chỉ dùng lại khi: câu **không có thực thể riêng** VÀ có **tín hiệu phụ thuộc** (khớp `followup_rules` hoặc đại từ). **Mặc định "không dùng lại".** Ca mơ hồ để router LLM có sẵn quyết định (mặc định không dùng lại).
- Khi có dùng lại thì câu trả lời nêu giả định. Thực thể dùng lại **không** đưa vào `entity_hint` của RAG.
- Khớp từ khóa theo **ranh giới từ**, không khớp chuỗi con.
- Danh sách tín hiệu **không hardcode trong code**: nằm trong `general_config.json`.

**K5-b. Kiểm tra bám nguồn thay `missing_numbers` (GĐ1)**
- Số, email, số điện thoại, số hiệu trong câu trả lời phải có trong nguồn hoặc trong câu hỏi. **SĐT xử lý như chuỗi (giữ số 0 đầu)**, hết lỗi `0986426633` → `986426633`.
- Chức danh trong câu trả lời phải có trong dòng của thực thể hoặc nằm trong bí danh K3 (khi K3 có).
- Khẳng định sự kiện mà không có nguồn vượt `LIEN_QUAN_SIMILARITY_THRESHOLD` (khởi điểm 0.55) → dùng mẫu "chưa có dữ liệu".
- Kiểm tra bỏ sót chỉ còn mức thông tin, và chỉ với trường câu hỏi thật sự yêu cầu.
- Mỗi cảnh báo có mức độ và `signature` để tắt.
- **Điểm tương đồng lấy từ lần gọi `similarity_search_with_score` RIÊNG, dùng lại vector câu hỏi đã tính ở bước retrieval, không embed lại.** Lý do: `rag_retrieve` là hybrid RRF (api8018.py:1007-1017, điểm `1/(RRF_K+rank+1)`, không phải độ tương đồng; xem chú thích dòng 1026). Biến `RAG_SIMILARITY_THRESHOLD` (dòng 66) **hiện không được dùng** trong `rag_retrieve()`.
- **Quy đổi khoảng cách:** FAISS trả khoảng cách L2. Trước hết kiểm tra vector của `qwen3-embedding:8b-ctx16k` qua Ollama **có chuẩn hóa không**. Nếu chuẩn hóa: `cosine = 1 - d²/2`. Công thức `1 - d/√2` của LangChain chỉ là heuristic, sai thang so với ngưỡng cosine. Nếu không chuẩn hóa, tự chuẩn hóa hoặc dùng độ đo khác, và ghi rõ trong báo cáo.

**K5-c. Trace ở mỗi câu trả lời (GĐ1):** đường xử lý (bảng/RAG/từ chối...), thực thể và cách khớp, có dùng ngữ cảnh phiên không, top-k chunk kèm điểm ngữ nghĩa, `index_version`. Ghi vào log; **email/PII trong trace băm, không lưu thật.**

**K5-d. Lưu trữ (GĐ0-1): JSON + ghi nguyên tử + `fcntl.flock`.  KHÔNG MongoDB.** monggo để GĐ3. Không ghi đè khi parse lỗi (xem P8).

**K5-e. Phiên (GĐ1):** khóa `_sessions` theo `(session_id, user_id)`. Hiện `session_id = body.get("session_id") or str(uuid.uuid4())` (api8018.py:1855), nên ai gọi với một `session_id` cố định sẽ dùng chung ngữ cảnh. - Giữ tương thích ngược: nếu request không có user_id, dùng chuỗi cố định
  "_anonymous_" (không dùng None, không dùng rỗng, để tránh vô tình trộn phiên).

### Cấu hình chung: `general_config.json` (file riêng, KHÔNG đặt trong `registry.json`)
- Lý do: không phải sửa mọi chỗ duyệt `registry.json` (matcher, `/metadata`, `table-config`); có chỗ cho cấu hình chung sau này.
- Đường dẫn: cùng cấp `registry.json` (`BASE / "general_config.json"`). Đọc ở api8018 khi khởi động và mỗi lần gọi POST /admin/reload-index; admin đọc/ghi (tab "Cấu hình chung" — **giao diện thuộc GĐ2**; GĐ1 chỉ cần file, bộ đọc và endpoint API).
- Khởi đầu (đây là **dữ liệu**, được phép nằm trong file cấu hình):

```json
{
  "version": 1,
  "followup_rules": {
    "keywords": ["đó", "này", "vậy", "thêm", "chi tiết hơn", "còn gì", "nữa không", "kể thêm", "còn nữa"],
    "pronouns": ["thầy ấy", "cô ấy", "anh ấy", "chị ấy", "người đó", "ngành đó", "khoa đó"]
  },
  "pii_keywords": ["dia chi nha", "dia chi ca nhan", "dia chi rieng", "nha rieng"],
  "thresholds": {
    "duyet_similarity": 0.90,
    "lien_quan_similarity": 0.55
  }
}
```

- Yêu cầu: thiếu file → dùng mặc định nội bộ rồi tạo file, **không crash**; file hỏng → giữ giá trị đang chạy, ghi log, không ghi đè; ghi nguyên tử; giữ bản sao lưu có phiên bản.
- **Lưu ý về từ khóa:** "này", "đó", "vậy" là từ rất ngắn; bắt buộc khớp theo ranh giới từ trên chuỗi đã chuẩn hóa (xem cách `normalize_vn` trong api8018.py), tránh khớp trong từ khác.

### Nút và trạng thái (GĐ2 — chỉ để hiểu định hướng)

| Nút cũ | Nút mới |
|---|---|
| Xác nhận và đưa vào RAG | **Duyệt câu trả lời** |
| Đánh dấu đã phù hợp (warning sai) | **Cảnh báo sai** (lưu `signature` để tắt, tính vào độ chính xác) |
| Đánh dấu chưa sửa | **Mở lại** |
| (mới) | **Lỗi hệ thống** (định tuyến, ngữ cảnh): sinh ca kiểm tra, không đổi nội dung |

Trạng thái mới: `mở → đã duyệt → cần duyệt lại` (khi nguồn đổi), hoặc `lỗi hệ thống` (đóng khi ca kiểm tra đạt).

### PII (rà soát ở GĐ2, ghi chú để GĐ0-1 không làm hỏng)
Cơ chế hiện có: `pii_columns` bị loại khỏi câu trả lời; `redact_known_pii_values` (api8018.py:1226) che theo giá trị. GĐ2 phải phủ: reload tự động (P10, làm ở GĐ0); che nội dung chunk trước khi đưa vào LLM; che khi phục vụ bản duyệt; từ khóa nhận diện "hỏi PII" lấy từ cấu hình thay `PII_KEYWORDS`; băm khóa chính là email trong log; dữ liệu đã lưu có giá trị vừa thành PII thì đánh dấu cần duyệt lại.

### `/admin/*` allowlist (GĐ3)
- Cấu hình qua biến môi trường **`ADMIN_ALLOWED_CIDRS`** (danh sách CIDR, phân tách dấu phẩy), **không hardcode**. Khi triển khai cho đơn vị khác chỉ sửa biến môi trường.
- Cờ `eval` (chạy kiểm tra không ghi FAQ log) **chỉ nhận từ IP nội bộ** — làm ở GĐ0 vì chặn việc bên ngoài né ghi log; dùng cùng cơ chế CIDR (mặc định loopback + dải riêng RFC1918 nếu chưa cấu hình).
- Nếu truy cập từ ngoài mạng nội bộ thì cần cơ chế khác (VPN hoặc token riêng cho `/admin/*`): **sẽ bàn khi triển khai, chưa làm.**
- Giữ nguyên `/ask` và `/metadata` công khai.

---

## PHẦN 4 — LỘ TRÌNH VÀ TIÊU CHÍ NGHIỆM THU

### GĐ0 — Ổn định nền (ước lượng 0,5-1 ngày)

**Nội dung**
1. Sửa P10: reload trong `worker()` của `update_table_config` (chỉ khi `returncode == 0`), ghi `reload_status` vào `result`.
2. Sửa P8: hàm dùng chung `atomic_write_json` + `read_json_safe` + khóa `flock`, áp dụng cho `faq_log.json`, `feedback_log.json` ở **cả admin_api.py và api8018.py**.
3. Bắt `EOFError` ở các chỗ `input()` của `conflict_detection.py` (dòng 816, 1469, 1988, 2003) và trả về giá trị mặc định an toàn, ghi log.
4. Cờ `eval` chỉ nhận từ IP nội bộ.
5. Ghi trace ra log (dạng tối thiểu: đường xử lý, thực thể, có dùng ngữ cảnh phiên không).
6. Script chụp ảnh câu trả lời 218 mục FAQ (câu hỏi → câu trả lời + citation) vào file, làm mốc so sánh cho GĐ1.
7. `_cleanup_old_jobs` không xóa job đang `queued/running` (kiểm tra và sửa nếu đang xóa).

**Nghiệm thu (input → output)**
- G0-1: chạy `update_table_config` đổi cờ PII của cột `mobile` trong bảng `giangvien` (trên **bản sao dữ liệu**) → job kết thúc, `result.reload_status == "ok"`; hỏi ngay "số điện thoại của Bùi Đức Thọ" → câu trả lời **không** chứa `0986426633`. Đặt lại cờ → làm lại reload → hết che.
- G0-2: cố ý ghi rác vào `faq_log.json` (bản sao) rồi gọi luồng lưu FAQ → file hỏng được chuyển `*.corrupt.<ts>`, lần lưu bị từ chối kèm log lỗi, **không** có `{}` ghi đè.
- G0-3: 2 tiến trình ghi đồng thời 200 bản ghi mỗi bên vào `faq_log.json` (bản sao) → đủ 400 bản ghi, file luôn parse được.
- G0-4: chạy `conflict_detection.py` với `stdin=DEVNULL` ở nhánh gọi `input()` → không văng `EOFError` chưa bắt, có log dùng giá trị mặc định.
- G0-5: `POST /ask` có `"eval": true` từ IP ngoài loopback/nội bộ → cờ bị bỏ qua (vẫn ghi FAQ log); từ loopback → không ghi.
- G0-6: có file ảnh chụp đủ 218 mục.

### GĐ1 — Sửa lõi (ước lượng 3-4 ngày)

**Nội dung**
1. Tạo `general_config.json` và bộ đọc (mục Phần 3), chuyển `FOLLOWUP_HINTS`, `PII_KEYWORDS` ra file (giữ mặc định nội bộ dự phòng).
2. K5-a: viết lại `looks_like_followup` (dòng 284-286) và mọi chỗ dùng nó; bỏ tiêu chí độ dài; ranh giới từ; không đưa thực thể dùng lại vào `entity_hint`; nêu giả định khi có dùng lại.
3. K5-b: thay khối `missing_numbers` (~1590-1620) bằng kiểm tra bám nguồn; sửa xử lý SĐT (giữ số 0 đầu); thêm `similarity_search_with_score` riêng, dùng lại vector câu hỏi.
4. K5-c: trace đầy đủ + citation dạng bảng cho nhánh bảng (đọc-only, che PII).
5. K5-e: khóa phiên `(session_id, user_id)`.
6. **Script hiệu chỉnh ngưỡng** `calibrate_thresholds.py` (xem 4.1).
7. Bộ kiểm tra khởi đầu `tests/` gồm ca các tiêu chí dưới đây.

**Nghiệm thu (input → output)**
- G1-1: **9 cảnh báo `missing_numbers` hiện có đều biến mất** khi chạy lại 9 câu hỏi (danh sách trong `faq_log.json`).
- G1-2: hỏi "Hiệu trưởng là ai", rồi cùng phiên hỏi "đào tạo tín chỉ là gì" → câu thứ hai **không còn dữ liệu Bùi Đức Thọ** và **có trích QD389** trong citation.
- G1-3: 7 câu độc lập (từ kiểm tra mục 1: "đào tạo tín chỉ là gì", "học phí k65", "quy trình xét học phí", "Hiệu trưởng là ai", "điểm chuẩn ngành marketing 2024 là bao nhiêu", "cho mình hỏi thủ tục xin học bổng khuyến khích học tập", "chuyển ngành có được không"), còn **"kể thêm về thầy ấy" vẫn nối tiếp**.
- G1-4: câu trả lời chứa SĐT **đúng** của dòng thực thể → không bị báo thiếu; câu trả lời chứa SĐT **bịa** → bị báo.
- G1-5: đổi `followup_rules` trong `general_config.json` (thêm 1 từ khóa) rồi reload → hành vi đổi theo, **không sửa code**.
- G1-6: 2 client dùng cùng `session_id` khác `user_id` → không dùng chung ngữ cảnh.
- G1-7: có **báo cáo phân bố cho 0.90 và 0.55** (xem 4.1), đã được người dùng duyệt trước khi cố định ngưỡng.
- G1-8: so ảnh chụp 218 mục trước/sau: xuất bảng chênh lệch, không có mục nào mất citation hoặc lộ PII.

### 4.1 Script hiệu chỉnh ngưỡng (bắt buộc trong GĐ1)

1. Kiểm tra vector embedding có chuẩn hóa không → chọn công thức quy đổi.
2. **K2 (0.90):** lấy các cặp câu hỏi từ 218 mục FAQ.
   - Cặp dương: cùng nghĩa, diễn đạt khác.
   - Cặp âm gần: "Hiệu trưởng" vs "Phó hiệu trưởng"; cùng câu khác năm; cùng chức danh khác người.
   - Xuất phân bố hai nhóm, khoảng trống, và ngưỡng đề xuất. **Nếu chồng lấn → khuyến nghị K2 chỉ dùng khớp chuẩn hóa chính xác.**
3. **K5-b (0.55):** cặp (câu hỏi, chunk) liên quan / không liên quan, ví dụ "đào tạo tín chỉ là gì" với QD389 (liên quan) và với chunk không liên quan. Xuất phân bố và các ca sát ngưỡng.
4. Xuất báo cáo (`.md` + số liệu thô). **Không tự cố định ngưỡng; chờ người dùng duyệt.**

### GĐ2 (chưa làm) — K1, K2, K3, tab "Cấu hình chung", test async + banner qua F5, K4 đầy đủ, nút/trạng thái mới, rà soát PII. Ước lượng 5-7 ngày.
### GĐ3 (chưa làm) — monggo, gom nhóm câu hỏi, nối feedback vào FAQ, gunicorn/waitress, `/health`, `ADMIN_ALLOWED_CIDRS`. Ước lượng 4-6 ngày.

---

## PHẦN 5 — YÊU CẦU KỸ THUẬT CHO NGƯỜI THỰC HIỆN

1. **Bắt đầu từ GĐ0, rồi GĐ1. Chưa làm GĐ2-3.**

2. **Với MỖI fix, phải chứng minh đủ 6 tiêu chí** (viết ngắn, cụ thể, kèm bằng chứng):
   - **Hiệu quả:** giải quyết đúng bài toán không? Sửa gốc hay triệu chứng?
   - **Bao quát:** xử lý được nhiều trường hợp không? **Liệt kê ≥ 3 biến thể** đã thử.
   - **Tính ứng dụng:** dùng được trong thực tế không? Admin thao tác thế nào?
   - **Tính khả thi:** triển khai được với nguồn lực hiện có (không thêm dịch vụ) không?
   - **Mở rộng:** thêm dữ liệu/yêu cầu mới có phải sửa code không?
   - **Tin cậy và bảo trì:** ổn định không? Rollback thế nào? 6 tháng sau người khác đọc có hiểu không?

3. **Với mỗi fix, ghi rõ:**
   - File nào, dòng nào (số dòng **trước** khi sửa), sửa thành gì.
   - Có ảnh hưởng file nào khác không (tìm mọi nơi gọi hàm bị đổi).
   - Cách test cụ thể (lệnh + kết quả kỳ vọng).
   - Cách rollback nếu lỗi.

4. **Không được:**
   - Hardcode danh sách từ khóa trong code (phải để trong config).
   - Sửa một case mà không giải thích vì sao đúng cho case khác.
   - Bỏ qua regression test.
   - Thêm MongoDB hay dịch vụ mới ở GĐ0-1.
   - In giá trị bí mật (`API_AUTH_TOKEN`...) ra log, báo cáo, hay diff.
   - Đổi kết thúc dòng: `admin_api.py` là CRLF, giữ nguyên CRLF; `api8018.py` là LF.
   - Suy đoán về file chưa đọc (`multi_entity_matcher.py`, `structured_data_pipeline.py`...): phải mở đọc trước khi dựa vào.

5. **Cách thử an toàn:**
   - Copy toàn bộ `data/` (và `registry.json`, `general_config.json` nếu có) sang thư mục khác.
   - Chạy api8018 và admin_api **trên cổng khác** (không trùng cổng đang chạy) trỏ vào bản copy.
   - Trước khi sửa: chụp mốc (ảnh chụp 218 FAQ, checksum các file dữ liệu).
   - Chỉ áp lên bản chính khi **toàn bộ tiêu chí nghiệm thu GĐ0-1 đạt**.
   - Việc chạy thử **không được thay đổi** dữ liệu bản chính (kiểm tra checksum trước/sau).

6. **Đầu ra cuối cùng cần có:**
   - Danh sách file sửa kèm diff hoặc mô tả thay đổi.
   - Script test tự động cho từng tiêu chí G0-x, G1-x.
   - Lệnh chạy test trên bản sao dữ liệu.
   - Báo cáo kết quả test (từng tiêu chí đạt/không đạt, số liệu thô).
   - Báo cáo hiệu chỉnh ngưỡng (4.1).
   - Hướng dẫn rollback (khôi phục file gốc, xóa `general_config.json`, chuyển file `*.corrupt.*` nếu có).
   - Danh sách rủi ro còn lại và điều **chưa** kiểm chứng được.

7. **Nếu gặp mâu thuẫn giữa tài liệu này và code thực tế** (ví dụ số dòng lệch, hàm đã đổi tên): dừng ở điểm đó, ghi rõ mâu thuẫn trong báo cáo, chọn phương án ít rủi ro nhất, **không tự thay đổi thiết kế đã chốt.**

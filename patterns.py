"""
Các regex "tổng quát" không gắn với 1 bảng cụ thể - trích thông tin phụ trợ
(năm, mã có định dạng chuẩn...) để lọc thêm SAU KHI đã xác định được bảng/
entity nào qua registry + dictionary match. Thêm pattern mới ở đây khi có
loại dữ liệu mới cần nhận diện dạng mã/số.
"""

import re

# Năm tuyển sinh / năm học - 2015 đến 2029 (chỉnh lại khi cần)
YEAR = re.compile(r"\b(20[12]\d)\b")

# Khoảng năm dạng "2018-2022", "2018 - 2022", "2018 đến 2022", "giai đoạn 2018 tới 2022"
YEAR_RANGE = re.compile(r"\b(20[12]\d)\s*(?:-|–|đến|tới)\s*(20[12]\d)\b")

# Mã ngành kiểu NEU: 7 chữ số, có thể kèm hậu tố "_1","_2" (nhóm xét tuyển)
MANGANH_CODE = re.compile(r"\b7\d{6}(_\d+)?\b")

# Email nội bộ trường (dùng để nhận diện câu hỏi trỏ thẳng email giảng viên)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def extract_years(text: str) -> list[str]:
    """Trả về TOÀN BỘ năm liên quan trong câu hỏi - nếu phát hiện dạng
    KHOẢNG (vd "2018-2022", "giai đoạn 2018 đến 2022"), trả về ĐỦ mọi năm
    trong khoảng đó (bao gồm cả năm ở giữa), KHÔNG chỉ 2 mốc đầu-cuối.
    Nếu không phải khoảng, trả về danh sách các năm đơn lẻ được nhắc tới
    (dùng như bộ lọc OR - câu hỏi nhắc năm nào giữ đúng năm đó)."""
    range_match = YEAR_RANGE.search(text)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start > end:
            start, end = end, start
        return [str(y) for y in range(start, end + 1)]
    return YEAR.findall(text)


def extract_manganh_codes(text: str) -> list[str]:
    return [m.group(0) for m in MANGANH_CODE.finditer(text)]


def extract_emails(text: str) -> list[str]:
    return EMAIL.findall(text)

"""
TẦNG 0 tổng quát — thay cho entity_matcher.py (bản cũ hardcode 2 bảng).
Đọc registry.json, tự build dictionary tra cứu cho MỌI bảng đã đăng ký.
Thêm bảng mới (phòng ban, chương trình đào tạo...) = thêm 1 mục JSON,
KHÔNG sửa file này.

Khóa chính khai báo trong registry ("primary_key") dùng CHUNG cho 2 việc:
  1. Ở đây: định danh chính xác 1 bản ghi sau khi regex/dictionary match được
     tên -> tránh mọi nhầm lẫn khi có tên trùng/gần giống.
  2. Sau này (sơ đồ "Phát hiện xung đột - dữ liệu có cấu trúc"): CHÍNH bảng
     khóa này là cột dùng để so khớp production vs staging khi phát hiện
     dữ liệu mới/trùng lặp/thay đổi.
  -> Không cần thiết kế 2 hệ khóa khác nhau cho 2 mục đích.

Cách dùng:
    from multi_entity_matcher import MultiEntityMatcher
    matcher = MultiEntityMatcher("registry.json")
    result = matcher.match("Bùi Cẩm Vân giữ chức vụ gì")
    # result = [TableMatch(table="giangvien", pk="vanbc@neu.edu.vn", row={...})]
"""

import json
import re
import sys
import unicodedata
import difflib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


def normalize_vn(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# [FIX-1] Ngưỡng chống match quá rộng. Đọc từ general_config.json (khóa
# "matcher"), thiếu -> default an toàn. Không phụ thuộc file config tồn tại.
# ---------------------------------------------------------------------------
DEFAULT_MAX_REASONABLE_HITS = 50     # > ngưỡng này (không phải group) -> coi như KHÔNG match entity
DEFAULT_SINGLE_ROLE_SMALL_HITS = 10  # nhánh role đơn lẻ: > ngưỡng này thì kiểm tra role quá chung

# Role QUÁ CHUNG: không phân biệt được ai với ai (đã normalize_vn).
GENERIC_ROLE_PHRASES = frozenset({
    "giang vien", "giang vien chinh", "giang vien thinh giang",
    "giang vien co huu", "can bo", "nhan vien", "chuyen vien",
    "giao vien", "hang iii", "hang ii", "hang i",
})


def _matcher_cfg() -> dict:
    """Đọc cấu hình matcher từ general_config.json; lỗi/thiếu -> {} (dùng default)."""
    try:
        from general_config import get_config
        c = get_config().get("matcher") or {}
        return c if isinstance(c, dict) else {}
    except Exception:
        return {}


def _as_int(v, default: int) -> int:
    try:
        v = int(v)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _is_generic_role_phrase(cand: str, extra: set = frozenset()) -> bool:
    """True nếu cụm role chỉ gồm từ chung (vd 'giang vien', 'giang vien chinh')."""
    c = (cand or "").strip()
    return c in GENERIC_ROLE_PHRASES or c in extra


def clean_cell(v) -> str:
    return "" if pd.isna(v) else str(v).strip()


def _candidate_o_vi_tri_chinh(role_key: str, cand: str, q_words: list) -> bool:
    """[GĐ2-B1-v4] True nếu 'cand' xuất hiện trong 'role_key' ở VỊ TRÍ CHÍNH
    (đầu chuỗi / sau dấu '-' / sau 'pho' nếu câu hỏi có 'pho').

    Ngăn candidate vị trí-only (vd 'truong thanh vien') match role_key mà
    cand KHÔNG phải chức danh chính (vd 'chanh van phong (truong thanh vien)')."""
    idx = role_key.find(cand)
    if idx < 0:
        return False
    prefix = role_key[:idx].rstrip()
    if not prefix:
        return True
    if prefix.endswith("-") or prefix.endswith("–"):
        return True
    if prefix.endswith("pho") and "pho" in q_words:
        return True
    return False


def _contains_word(s: str, sub: str) -> bool:
    """[GĐ2-B1] True nếu 'sub' xuất hiện trong 's' như 1 CỤM TỪ nguyên vẹn
    (ranh giới từ bằng \b). Cả 's' và 'sub' PHẢI đã normalize (bỏ dấu,
    lowercase, gộp space).

    Khắc phục bug cũ: `sub in s` cho phép cắt ngang từ, vd
      "abc xdefghi jkl" chứa "defghi jkl" (substring) → SAI.
    Với _contains_word: False vì 'x' và 'd' liền kề không có ranh giới từ.
    """
    if not s or not sub:
        return False
    return re.search(r"\b" + re.escape(sub) + r"\b", s) is not None

# Sau đoạn def clean_cell(v) và trước @dataclass TableMatch, thêm:

_PANDAS_DTYPE = {"string": "string", "int": "Int64",
                 "float": "Float64", "bool": "boolean"}


def read_csv_with_dtypes(path, cfg: dict) -> "pd.DataFrame":
    """Đọc CSV theo column_dtypes khai trong registry — QUAN TRỌNG để cột
    mobile/cccd/mã không bị pandas convert sang số (mất số 0 đầu, vd
    0917561985 -> 917561985.0). Nếu bảng chưa khai column_dtypes (bảng cũ
    chưa qua UI mới), fallback đọc bình thường nhưng ép các cột có tên
    giống SĐT/mã về string để phòng hờ."""
    dtype_map = cfg.get("column_dtypes") or {}
    pandas_dtype_arg = {c: _PANDAS_DTYPE[t] for c, t in dtype_map.items()
                        if t in _PANDAS_DTYPE}

    # Fallback phòng vệ cho bảng chưa khai column_dtypes — vẫn quét tên cột
    # để tránh bug SĐT/mã bị convert số, đây là lỗi điển hình đã gặp.
    if not pandas_dtype_arg:
        import re as _re
        phone_hint = _re.compile(r"phone|mobile|dien_thoai|điện_thoại|sdt|sđt|"
                                  r"so_dien_thoai|tel|\bđt\b", _re.I)
        id_hint = _re.compile(r"^(ma_|mã_|cccd|cmnd|bhxh|tax_code|mst|"
                               r"bank_acc|so_tk|email)", _re.I)
        try:
            header = pd.read_csv(path, nrows=0)
            for c in header.columns:
                if phone_hint.search(str(c)) or id_hint.search(str(c)):
                    pandas_dtype_arg[c] = "string"
        except Exception:
            pass

    return pd.read_csv(path, dtype=pandas_dtype_arg or None,
                       keep_default_na=False, na_values=[""])


@dataclass
class TableMatch:
    table: str          # tên bảng trong registry, vd "giangvien"
    pk: str              # giá trị khóa chính của bản ghi khớp
    rows: "pd.DataFrame" = None   # TOÀN BỘ dòng khớp khóa này (đã lọc PII), 1 hoặc nhiều dòng
    ten_khop: str = ""    # chuỗi thực sự khớp trong câu hỏi


class MultiEntityMatcher:
    def __init__(self, registry_path: str, base_dir: str = ".", fuzzy_cutoff: float = 0.88):
        self.fuzzy_cutoff = fuzzy_cutoff
        self.base_dir = Path(base_dir)
        self.registry: dict = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        self._tables: dict[str, pd.DataFrame] = {}
        self._lookup: dict[str, list[tuple[str, str]]] = {}   # normalized_name -> [(table, pk), ...]
        self._role_lookup: dict[str, list[tuple[str, str]]] = {}  # normalized_role_value -> [(table, pk), ...]
        self._group_lookup: dict[str, list[tuple[str, str]]] = {}  # normalized_group_value -> [(table, pk), ...] - vd department, dùng riêng cho câu hỏi "X gồm những ai"
        self._keys_sorted: list[str] = []
        self._pho_pks: set = set()
        # [GĐ2-B1-v3] Alias + loại trường thành viên
        self._alias_lookup: dict = {}              # {alias_key_norm: [(table, pk)]}
        self._alias_keys: list = []                # alias_key sorted by len desc
        self._truong_thanh_vien_pks: set = set()   # {(table, pk)} có position chứa "(truong thanh vien)"

        self._role_keys: list[str] = []
        self._load_all()

    def _has_truong_cu_the(self, q_norm: str) -> bool:
        """True nếu câu hỏi nêu 'trường thành viên' HOẶC 'trường X' (X là tên
        thật của group trong registry — word boundary). Dùng để quyết định
        KHÔNG áp alias (giữ match gốc 7 người / người cụ thể)."""
        if "truong thanh vien" in q_norm:
            return True
        for gk in self._group_keys:
            if gk.startswith("truong ") and _contains_word(q_norm, gk):
                return True
        return False

    # ------------------------------------------------------------------
    def _load_all(self):
        for table_name, cfg in self.registry.items():
            path = self.base_dir / cfg["path"]
            df = read_csv_with_dtypes(path, cfg)
            df.columns = df.columns.str.strip()
            self._tables[table_name] = df

            pk_col = cfg["primary_key"]
            name_cols = cfg["name_columns"]

            for _, row in df.iterrows():
                pk_val = clean_cell(row.get(pk_col))
                if not pk_val:
                    continue
                for col in name_cols:
                    raw = row.get(col)
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    key = normalize_vn(raw)
                    if not key:
                        continue
                    self._lookup.setdefault(key, []).append((table_name, pk_val))
                # cho phép tra cứu thẳng bằng khóa chính (mã ngành, email...)
                pk_key = normalize_vn(pk_val)
                self._lookup.setdefault(pk_key, []).append((table_name, pk_val))

                # role_columns: chức vụ/học hàm - dùng cho câu hỏi KHÔNG nêu tên
                # (vd "Giám đốc Đại học là ai", không phải "X giữ chức vụ gì").
                # Hướng khớp NGƯỢC với tên: ở đây kiểm tra xem cụm từ trong CÂU HỎI
                # có phải là 1 đoạn con nằm TRONG giá trị chức vụ hay không (vì câu
                # hỏi thường ngắn hơn, generic hơn giá trị đầy đủ trong dữ liệu,
                # vd hỏi "giám đốc đại học" nhưng dữ liệu ghi "Bí thư Đảng ủy -
                # Giám đốc Đại học") - nên lưu riêng, xử lý ở hàm match().
                for col in cfg.get("role_columns", []):
                    raw = row.get(col)
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    key = normalize_vn(raw)
                    if key:
                        self._role_lookup.setdefault(key, []).append((table_name, pk_val))

                        # [GĐ2-B1-v3] Áp alias cho cột này (nếu có). Alias CHỈ áp
                        # khi value thật KHÔNG bị prefix "phó" — câu hỏi "phó..."
                        # sẽ match qua role gốc, không qua alias.
                        aliases_for_col = (cfg.get("column_value_aliases") or {}).get(col) or {}
                        for alias_kw, alias_val in aliases_for_col.items():
                            if not alias_kw or not alias_val:
                                continue
                            val_norm = normalize_vn(alias_val)
                            if not val_norm or not _contains_word(key, val_norm):
                                continue
                            idx = key.find(val_norm)
                            prefix_words = key[:idx].split()
                            if prefix_words and prefix_words[-1] == "pho":
                                continue  # "phó ..." không thêm alias
                            alias_key = normalize_vn(alias_kw)
                            if alias_key and alias_key != key:
                                self._alias_lookup.setdefault(alias_key, []).append((table_name, pk_val))

                    # [GĐ2-B1-v3] Track "trường thành viên" để loại khi cần
                    if "truong thanh vien" in key:
                        self._truong_thanh_vien_pks.add((table_name, pk_val))
                        if key.startswith("pho ") or key.startswith("pho\t"):
                            self._pho_pks.add((table_name, pk_val))

                # group_columns (vd department) - TÁCH RIÊNG khỏi role_lookup,
                # vì 1 giá trị đơn vị có thể trỏ tới NHIỀU người (cả lãnh đạo
                # lẫn thư ký/nhân viên) - trộn chung với role (thường 1-2 người)
                # gây khớp tràn lan khi câu hỏi chỉ hỏi 1 chức vụ cụ thể. Chỉ
                # dùng group_lookup khi câu hỏi rõ ràng hỏi về CẢ NHÓM.
                for col in cfg.get("group_columns", []):
                    raw = row.get(col)
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    key = normalize_vn(raw)
                    if key:
                        self._group_lookup.setdefault(key, []).append((table_name, pk_val))

        self._keys_sorted = sorted(self._lookup.keys(), key=len, reverse=True)
        # role keys giữ nguyên thứ tự - ở đây ta kiểm tra "candidate NGẮN từ
        # câu hỏi có nằm TRONG value DÀI hay không", không phụ thuộc thứ tự.
        self._role_keys = list(self._role_lookup.keys())
        self._group_keys = list(self._group_lookup.keys())
        self._alias_keys = sorted(self._alias_lookup.keys(), key=len, reverse=True)

    # ------------------------------------------------------------------
    def rows_by_pk(self, table: str, pk_val: str) -> pd.DataFrame:
        """Trả về TOÀN BỘ các dòng khớp khóa chính - có thể là 1 dòng (bảng
        kiểu 'entity', vd giảng viên) hoặc NHIỀU dòng (bảng kiểu 'time-series/
        long', vd ngành lặp theo năm). KHÔNG tự ý chỉ lấy dòng đầu, để lớp
        gọi (api8018.py) tự quyết định cách xử lý theo đúng loại bảng.
        Đã lọc bỏ pii_columns (dữ liệu cá nhân) VÀ hidden_columns (nhiễu nội
        bộ, không phải PII nhưng không nên hiện ra, vd đường dẫn file windows
        trong cột 'department_level' của dsgv.xlsx).
        Public vì session follow-up (api8018.py) cần gọi lại theo (table, pk)
        đã lưu từ lượt hỏi trước, không qua match() lại."""
        cfg = self.registry[table]
        df = self._tables[table]
        matched = df[df[cfg["primary_key"]].astype(str).str.strip() == pk_val].copy()
        drop_cols = list(cfg.get("pii_columns", [])) + list(cfg.get("hidden_columns", []))
        matched = matched.drop(columns=[c for c in drop_cols if c in matched.columns])
        return matched

    def is_single_row_entity(self, table: str, pk_val: str) -> bool:
        """True nếu khóa chính này trỏ đúng 1 dòng (an toàn để trả lời trực
        tiếp không qua LLM). False nếu nhiều dòng (cần lọc + tính toán thêm,
        vd nhiều năm/nhiều phương thức xét tuyển cho cùng 1 ngành)."""
        return len(self.rows_by_pk(table, pk_val)) == 1

    def match(self, question: str) -> list[TableMatch]:
        q_norm = normalize_vn(question)
        hits: list[TableMatch] = []
        seen = set()

        for key in self._keys_sorted:
            if len(key) < 4 or not _contains_word(q_norm, key):
                continue
            for table, pk_val in self._lookup[key]:
                sig = (table, pk_val)
                if sig in seen:
                    continue
                seen.add(sig)
                hits.append(TableMatch(table=table, pk=pk_val,
                                        rows=self.rows_by_pk(table, pk_val), ten_khop=key))

        # Cơ chế A: thu hẹp danh sách trùng tên bằng ngữ cảnh đơn vị đã nêu
        # trong câu hỏi - vd "Nguyễn Quang Huy Hiệu trưởng trường Công nghệ"
        # khớp tên ra 3 người, nhưng chỉ 1 người có department thật sự là
        # "Trường Công nghệ" - dùng đúng dữ liệu đó để chọn ra 1.
        if len(hits) > 1:
            refined = []
            for h in hits:
                row = h.rows.iloc[0]
                cfg = self.registry[h.table]
                matched_context = False
                for col in cfg.get("group_columns", []):
                    val = row.get(col)
                    if isinstance(val, str) and val.strip():
                        val_norm = normalize_vn(val)
                        if val_norm and val_norm in q_norm:
                            matched_context = True
                            break
                if matched_context:
                    refined.append(h)
            if len(refined) == 1:
                hits = refined

        # Câu hỏi rõ ràng đang hỏi về NGƯỜI (vd "trưởng khoa", "là ai") mà lại
        # khớp trúng 1 bảng KHÔNG PHẢI bảng người (không có role/group_columns,
        # vd bảng ngành đào tạo) - do tên đơn vị trùng tên ngành (vd "Công nghệ
        # thông tin" vừa là khoa vừa là ngành) - loại các khớp sai này để các
        # cơ chế khớp CHỨC VỤ + ĐƠN VỊ bên dưới có cơ hội tìm đúng người.
        PERSON_INTENT_KEYWORDS = ["la ai", "truong khoa", "giam doc", "hieu truong",
                                   "vien truong", "truong bo mon", "giang vien"]
        if hits and any(kw in q_norm for kw in PERSON_INTENT_KEYWORDS):
            hits = [h for h in hits
                    if self.registry[h.table].get("role_columns") or self.registry[h.table].get("group_columns")]

        # Khớp theo CHỨC VỤ trước fuzzy - vì đây là so khớp CHÍNH XÁC (substring),
        # đáng tin hơn fuzzy "đoán gần đúng" ở dưới. Dùng khi câu hỏi không nêu
        # tên ai (vd "Giám đốc Đại học là ai") mà hỏi về vai trò. Chỉ nhận cụm
        # đủ dài (>= 3 từ) để tránh khớp tràn lan quá chung chung.
        # Câu hỏi hỏi về CẢ NHÓM (vd "Ban Giám đốc gồm những ai") - dùng
        # group_lookup RIÊNG, TÁCH KHỎI role matching để không làm hỏng hành
        # vi an toàn "1 câu hỏi = 1 người" của role matching.
        GROUP_INTENT_KEYWORDS = ["gom nhung ai", "co nhung ai", "la nhung ai"]
        is_group_question = any(kw in q_norm for kw in GROUP_INTENT_KEYWORDS)

        if not hits and is_group_question and self._group_keys:
            words = q_norm.split()
            group_candidates = sorted(
                {" ".join(words[i:j]) for i in range(len(words))
                 for j in range(i + 2, min(i + 7, len(words) + 1))},
                key=len, reverse=True,
            )
            matched_group_cand = ""
            for cand in group_candidates:
                if len(cand) < 6:
                    continue
                for group_key in self._group_keys:
                    if _contains_word(group_key, cand):
                        for table, pk_val in self._group_lookup[group_key]:
                            hits.append(TableMatch(table=table, pk=pk_val,
                                                    rows=self.rows_by_pk(table, pk_val),
                                                    ten_khop=f"[đơn vị] {cand}"))
                        matched_group_cand = cand
                if hits:
                    break

            # MỚI: nếu câu hỏi chứa CHỨC VỤ cụ thể DÀI HƠN cụm group đã khớp
            # (vd hỏi "Phó Giám đốc gồm những ai" - group khớp "giam doc" vì
            # "giam doc" là substring của "ban giam doc", nhưng câu hỏi thực
            # chất hỏi chức vụ "pho giam doc"), phải GIAO kết quả group với
            # kết quả role matching - tránh trả về cả Giám đốc khi người dùng
            # chỉ hỏi Phó Giám đốc.
            if hits:
                best_role_cand = ""
                for cand in group_candidates:
                    if len(cand) < 6 or len(cand) <= len(matched_group_cand):
                        continue
                    for role_key in self._role_keys:
                        if not _contains_word(role_key, cand):
                            continue
                        if not _candidate_o_vi_tri_chinh(role_key, cand, q_norm.split()):
                            continue
                        best_role_cand = cand
                        break
                    if best_role_cand:
                        break

                if best_role_cand:
                    role_pks = set()
                    for role_key in self._role_keys:
                        if not _contains_word(role_key, best_role_cand):
                            continue
                        if not _candidate_o_vi_tri_chinh(role_key, best_role_cand, q_norm.split()):
                            continue
                        role_pks.update(self._role_lookup[role_key])
                    refined = [h for h in hits if (h.table, h.pk) in role_pks]
                    if refined:
                        hits = refined

        # Cơ chế B: kết hợp CHỨC VỤ + ĐƠN VỊ - dành cho câu hỏi không nêu tên
        # nhưng nêu cả chức vụ lẫn đơn vị (vd "Ai là Hiệu trưởng Trường Công
        # nghệ?"). Chạy TRƯỚC cơ chế chức vụ đơn lẻ dưới đây - có 2 tiêu chí
        # rõ ràng đặc thù hơn (chức vụ có thể trùng ở nhiều đơn vị khác nhau,
        # nhưng giao của "chức vụ + đơn vị" thường ra đúng 1 người) nên phải
        # được ưu tiên thử trước; nếu chạy SAU cơ chế chức vụ đơn lẻ, cơ chế
        # đơn lẻ sẽ "khớp" trước (hits không rỗng) và cơ chế kết hợp không
        # bao giờ có cơ hội chạy, dù câu hỏi đã cung cấp đủ thông tin đơn vị
        # để xác định chính xác 1 người.
        #
        # Bỏ qua các candidate chỉ gồm 1 từ đầu chung của tổ chức ("trung tam",
        # "khoa", "vien"...) mà không có từ phân biệt theo sau - tránh khớp
        # nhầm sang thực thể không liên quan. Bắt buộc candidate phải có thêm
        # ít nhất 1 từ riêng (vd "trung tam doi moi" mới tính, "trung tam"
        # đơn thuần thì không).
        GENERIC_ORG_PREFIXES = [p.split() for p in
                                 ("trung tam", "khoa", "vien", "phong", "ban", "truong", "bo mon")]

        def _la_tu_dau_chung_khong_phan_biet(cand_words: list) -> bool:
            for prefix_words in GENERIC_ORG_PREFIXES:
                n = len(prefix_words)
                if cand_words[:n] == prefix_words and len(cand_words) <= n:
                    return True
            return False

        if not hits and self._group_keys and self._role_keys:
            words = q_norm.split()
            all_candidates = sorted(
                {" ".join(words[i:j]) for i in range(len(words))
                 for j in range(i + 2, min(i + 7, len(words) + 1))},
                key=len, reverse=True,
            )
            dept_pks = set()
            for cand in all_candidates:  # đã sắp XẾP GIẢM DẦN theo độ dài
                if len(cand) < 6 or _la_tu_dau_chung_khong_phan_biet(cand.split()):
                    continue
                found = False
                for group_key in self._group_keys:
                    if _contains_word(group_key, cand):
                        dept_pks.update(self._group_lookup[group_key])
                        found = True
                if found:
                    break  # chỉ lấy cụm DÀI NHẤT khớp được - tránh cụm ngắn
                            # kiểu "thông tin" khớp nhầm nhiều đơn vị khác nhau

            role_pks = set()
            for cand in all_candidates:
                if len(cand) < 8 or _la_tu_dau_chung_khong_phan_biet(cand.split()):
                    continue
                found = False
                for role_key in self._role_keys:
                    if not _contains_word(role_key, cand):
                        continue
                    if not _candidate_o_vi_tri_chinh(role_key, cand, q_norm.split()):
                        continue
                    role_pks.update(self._role_lookup[role_key])
                    found = True
                if found:
                    break

            combined_pks = dept_pks & role_pks
            for table, pk_val in combined_pks:
                hits.append(TableMatch(table=table, pk=pk_val,
                                        rows=self.rows_by_pk(table, pk_val),
                                        ten_khop="[chức vụ + đơn vị kết hợp]"))

        # Cơ chế chức vụ ĐƠN LẺ - dùng khi câu hỏi KHÔNG nêu đơn vị (hoặc cơ
        # chế kết hợp ở trên không tìm được gì), CHỈ nêu chức vụ (vd "Giám
        # đốc Đại học là ai?"). Chạy SAU cơ chế kết hợp để không "cướp" mất
        # cơ hội thu hẹp theo đơn vị khi câu hỏi thực ra có nêu đơn vị.
        single_role_cand = ""
        blocked_generic_role = False
        if not hits and self._role_keys:
            words = q_norm.split()
            # [GĐ2-B1] min 2 từ (thay 3) — để match được role 2 từ (vd "hiệu trưởng").
            # An toàn nhờ word boundary: chỉ match nếu cụm là từ nguyên vẹn trong role_key.
            role_candidates = [" ".join(words[i:j]) for i in range(len(words))
                                for j in range(i + 2, min(i + 7, len(words) + 1))]
            role_candidates.sort(key=len, reverse=True)  # ưu tiên cụm dài, cụ thể hơn trước
            seen_role = set()
            for cand in role_candidates:
                if len(cand) < 8:
                    continue
                for role_key in self._role_keys:
                    if not _contains_word(role_key, cand):
                        continue
                    # [GĐ2-B1-v4] Chỉ match khi candidate ở vị trí chính
                    # (đầu / sau '-' / sau 'pho' nếu câu hỏi có 'pho').
                    if not _candidate_o_vi_tri_chinh(role_key, cand, q_norm.split()):
                        continue
                    for table, pk_val in self._role_lookup[role_key]:
                        sig = (table, pk_val)
                        if sig in seen_role:
                            continue
                        seen_role.add(sig)
                        hits.append(TableMatch(table=table, pk=pk_val,
                                                    rows=self.rows_by_pk(table, pk_val),
                                                    ten_khop=f"[chức vụ] {cand}"))
                if hits:
                    single_role_cand = cand
                    break  # đã tìm được ở độ dài cụm này, không cần thử cụm ngắn hơn nữa

            # [FIX-1b] Role đơn lẻ mà quá chung + nhiều hit -> KHÔNG match.
            # Vd "giang vien" khớp mọi "Giảng viên (Hạng III)..." -> ~700 người.
            mcfg = _matcher_cfg()
            small_hits = _as_int(mcfg.get("single_role_small_hits"), DEFAULT_SINGLE_ROLE_SMALL_HITS)
            extra_generic = {normalize_vn(x) for x in (mcfg.get("generic_roles") or []) if isinstance(x, str)}
            if hits and len(hits) > small_hits and _is_generic_role_phrase(single_role_cand, extra_generic):
                print(f"[matcher] role quá chung '{single_role_cand}' -> {len(hits)} hit, bỏ "
                      f"(không phân biệt được entity)", file=sys.stderr)
                hits = []
                blocked_generic_role = True

        # [GĐ2-B1-v3] Conditional alias + loại trường thành viên.
        # - Câu hỏi KHÔNG có "trường X" / "trường thành viên" → chỉ trả Bùi (alias).
        # - Câu hỏi CÓ "trường X" / "trường thành viên" → giữ match gốc.
        # [GĐ2-B1-v3fix] Loại Phó nếu câu hỏi KHÔNG có "pho" (word).
        # Áp cho MỌI nhánh (kể cả có "trường X" hoặc "trường thành viên").
        q_words = q_norm.split()

        # [FIX-V2] Có NAME-MATCH cụ thể (hit sinh từ tên/khóa chính, ten_khop KHÔNG
        # bắt đầu bằng "[") trên bảng có role_columns -> người dùng hỏi ĐÍCH DANH 1 tên
        # người -> KHÔNG áp bộ lọc Phó / trường thành viên và KHÔNG thêm alias.
        # Các bộ lọc đó chỉ đúng cho câu hỏi CHỨC VỤ chung ("hiệu trưởng").
        # Vd "Nguyễn Quang Huy" phải trả đủ 3 người, gồm cả Hiệu trưởng Trường Công nghệ.
        has_name_match = any(
            not h.ten_khop.startswith("[")
            and self.registry[h.table].get("role_columns")
            for h in hits
        )

        if "pho" not in q_words and not has_name_match:
            hits = [h for h in hits if (h.table, h.pk) not in self._pho_pks]

        has_truong = self._has_truong_cu_the(q_norm)

        if not has_truong and not blocked_generic_role and not has_name_match:
            # 1. Loại pk có position chứa "(trường thành viên)"
            hits = [h for h in hits
                    if (h.table, h.pk) not in self._truong_thanh_vien_pks]

            # 2. Áp alias (chỉ cho những candidate còn lại)
            if self._alias_lookup:
                seen_sig = {(h.table, h.pk) for h in hits}
                words = q_norm.split()
                alias_candidates = [" ".join(words[i:j]) for i in range(len(words))
                                    for j in range(i + 2, min(i + 7, len(words) + 1))]
                alias_candidates.sort(key=len, reverse=True)
                for cand in alias_candidates:
                    if len(cand) < 6:
                        continue
                    for alias_key in self._alias_keys:
                        if not _contains_word(alias_key, cand):
                            continue
                        idx = alias_key.find(cand)
                        prefix_words = alias_key[:idx].split()
                        if (prefix_words and prefix_words[-1] == "pho"
                                and "pho" not in q_norm.split()):
                            continue
                        for table, pk_val in self._alias_lookup[alias_key]:
                            sig = (table, pk_val)
                            if sig in seen_sig:
                                continue
                            seen_sig.add(sig)
                            hits.append(TableMatch(table=table, pk=pk_val,
                                                    rows=self.rows_by_pk(table, pk_val),
                                                    ten_khop=f"[alias] {cand}"))
                    if hits:
                        break
    
        # [FIX-1a] Chốt chặn CUỐI: sau MỌI cơ chế thu hẹp mà vẫn > ngưỡng và KHÔNG
        # phải câu hỏi cả nhóm hợp lệ (group_lookup, ten_khop "[đơn vị] ...") ->
        # coi như không match entity cụ thể, để rơi xuống RAG / pandas agent toàn bảng.
        max_hits = _as_int(_matcher_cfg().get("max_reasonable_hits"), DEFAULT_MAX_REASONABLE_HITS)
        if len(hits) > max_hits and not all(h.ten_khop.startswith("[đơn vị]") for h in hits):
            print(f"[matcher] WARNING match quá rộng: {len(hits)} hit > {max_hits} "
                  f"(câu hỏi: {question!r}) -> trả rỗng", file=sys.stderr)
            return []

        # Đã bỏ fuzzy fallback (difflib): fuzzy khớp nhầm chủ đề hoàn toàn
        # khác (vd "Kinh tế quốc dân" vs "Kinh tế quốc tế") dù đã đặt ngưỡng
        # cao. Đánh đổi lấy ưu tiên "thà không trả lời còn hơn khớp sai".

        return hits

    def table_description(self, table: str) -> str:
        return self.registry[table]["description"]

    def dataframe(self, table: str) -> pd.DataFrame:
        """Trả về df ĐẦY ĐỦ (bao gồm cả cột PII) - CHỈ dùng cho mục đích nội
        bộ (vd thống kê), TUYỆT ĐỐI không đưa thẳng cho pandas agent hay bất
        kỳ chỗ nào output có thể hiển thị ra người dùng cuối."""
        return self._tables[table]

    def dataframe_safe(self, table: str) -> pd.DataFrame:
        """Trả về df đã lọc bỏ pii_columns/hidden_columns - PHẢI dùng hàm này
        (không phải dataframe()) mỗi khi đưa dữ liệu cho pandas agent hoặc
        bất kỳ nơi nào có khả năng in ra cho người dùng thấy, kể cả khi câu
        hỏi không cố ý hỏi về PII - vì agent có thể tự ý print() nguyên cả
        df nếu cột đó có sẵn trong dữ liệu đưa vào."""
        cfg = self.registry[table]
        df = self._tables[table]
        drop_cols = list(cfg.get("pii_columns", [])) + list(cfg.get("hidden_columns", []))
        return df.drop(columns=[c for c in drop_cols if c in df.columns])

    def all_pii_values(self) -> dict:
        """Gom toàn bộ giá trị pii_columns của mọi bảng thành 1 danh sách
        tham chiếu duy nhất - dùng để chặn PII theo GIÁ TRỊ đã biết, bất kể
        nó lọt vào câu trả lời cuối qua kênh nào (bảng hay RAG). Lý do:
        rows_by_pk()/dataframe_safe() chỉ chặn PII qua nhánh bảng, còn nội
        dung RAG có policy riêng coi số điện thoại/email là thông tin công
        khai hợp lệ (xem redact_pii_from_text trong api8018.py).
        Thay vì phải đoán "kênh nào đáng tin", hàm này gom TOÀN BỘ giá trị
        pii_columns của MỌI bảng thành 1 danh sách tham chiếu DUY NHẤT, để
        api8018.py có thể chặn đúng GIÁ TRỊ đã biết là PII, bất kể nó lọt vào
        câu trả lời cuối qua kênh nào (bảng hay RAG) - nhất quán tuyệt đối,
        không phụ thuộc giả định "tài liệu này chắc là an toàn".

        Trả về 2 tập:
        - 'phone_digits': chuỗi chỉ-số đã chuẩn hoá (kèm bản có/không số 0
          đầu, vì cột numeric trong CSV thường bị pandas làm rớt số 0 đầu khi
          đọc dạng số, vd 0912345678 -> 912345678.0) - dùng so khớp số điện
          thoại linh hoạt.
        - 'text_values': chuỗi gốc (địa chỉ, email cá nhân...) - dùng so
          khớp trực tiếp (substring) trong text.
        """
        phone_digits = set()
        text_values = set()
        for table_name, cfg in self.registry.items():
            pii_cols = cfg.get("pii_columns") or []
            if not pii_cols:
                continue
            df = self._tables[table_name]
            for col in pii_cols:
                if col not in df.columns:
                    continue
                for raw in df[col]:
                    val = clean_cell(raw)
                    if not val:
                        continue
                    # Cột số điện thoại đọc bằng pandas thường thành float
                    # (vd 912345678.0) - str() thẳng ra "912345678.0", nếu
                    # chỉ strip ký tự không phải số sẽ biến ".0" thành thêm 1
                    # số "0" ở cuối, sai lệch số thật. Cắt đuôi ".0" trước
                    # khi tách chữ số.
                    if re.match(r"^\d+\.0$", val):
                        val = val[:-2]
                    digits = re.sub(r"\D", "", val)
                    # Coi là "dạng số điện thoại" khi phần số chiếm gần hết
                    # độ dài chuỗi gốc (tránh gom nhầm địa chỉ có vài chữ số
                    # lẻ, vd "Số 5, ngách 12..." vào nhóm phone_digits).
                    if len(digits) >= 8 and len(digits) >= len(val) - 2:
                        phone_digits.add(digits)
                        if not digits.startswith("0"):
                            phone_digits.add("0" + digits)
                        phone_digits.add(digits.lstrip("0"))
                    else:
                        text_values.add(val)
        return {"phone_digits": phone_digits, "text_values": text_values}


if __name__ == "__main__":
    matcher = MultiEntityMatcher("registry.json", base_dir=".")
    tests = [
        "diem chuan nganh kinh te va quan ly do thi nam 2025",
        "Bùi Cẩm Vân giữ chức vụ gì",
        "7220201 lay bao nhieu diem",
    ]
    for t in tests:
        print(f"\nCâu hỏi: {t}")
        for m in matcher.match(t):
            single = matcher.is_single_row_entity(m.table, m.pk)
            print(f"  -> bảng={m.table} pk={m.pk} khớp='{m.ten_khop}' "
                  f"({'1 dòng - trả lời trực tiếp' if single else f'{len(m.rows)} dòng - cần lọc + agent tính toán'})")
            if single:
                print(f"     row: {m.rows.iloc[0].to_dict()}")

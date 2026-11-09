import os
import sys
sys.dont_write_bytecode = True
import subprocess
import json
import re
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import cv2
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    print("[*] Phát hiện máy chưa có thư viện xử lý ảnh/OCR.")
    print("[*] Đang tự động cài đặt (rapidocr-onnxruntime, opencv-python)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidocr-onnxruntime", "opencv-python"])
    import cv2
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR


def read_image_utf8(path):
    with open(path, "rb") as f:
        bytes_data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)


def save_image_utf8(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def detect_split_x(img):
    """Tự động tìm tọa độ X phân tách phần câu hỏi (nền sáng) và cột bình luận (nền tối)."""
    h, w, _ = img.shape
    sample_band = img[int(h * 0.2):int(h * 0.8), :, :]
    is_dark = np.all(sample_band < 55, axis=-1)
    col_dark_ratio = np.mean(is_dark, axis=0)
    for x in range(int(w * 0.5), int(w * 0.85)):
        if col_dark_ratio[x] > 0.85:
            return x
    return int(w * 1200 / 1820)


def extract_and_vote_comments(comment_panel, ocr):
    """Tách tên user (màu xanh) và nội dung comment (màu trắng), sau đó chạy thuật toán bầu chọn."""
    c_gray = cv2.cvtColor(comment_panel, cv2.COLOR_BGR2GRAY)
    text_mask = (c_gray > 115).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 4))
    grouped = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw >= 6 and bh >= 8 and by > 140:
            boxes.append((by, bx, bw, bh))

    boxes.sort(key=lambda b: b[0])

    crops = []
    line_types = []

    for by, bx, bw, bh in boxes:
        p_y1 = max(0, by - 4)
        p_y2 = min(comment_panel.shape[0], by + bh + 4)
        p_x1 = max(0, bx - 6)
        p_x2 = min(comment_panel.shape[1], bx + bw + 6)
        crop = comment_panel[p_y1:p_y2, p_x1:p_x2]
        crops.append(crop)

        bright_mask = np.max(crop, axis=-1) > 120
        if np.any(bright_mask):
            b_vals = crop[:, :, 0][bright_mask].astype(np.float32)
            r_vals = crop[:, :, 2][bright_mask].astype(np.float32)
            mean_diff = np.mean(b_vals - r_vals)
            if mean_diff > 45:
                line_types.append("user")
            else:
                line_types.append("text")
        else:
            line_types.append("text")

    raw_lines = []
    formatted_comments = []
    pure_comment_texts = []

    if crops:
        rec_res, _ = ocr.text_recognizer(crops)
        current_user = None
        current_msg = []

        for ltype, (text, score) in zip(line_types, rec_res):
            text = text.strip()
            if not text or text in ["Binh luan", "Bình luận"]:
                continue
            raw_lines.append(text)

            if ltype == "user":
                if current_user and current_msg:
                    msg_joined = " ".join(current_msg)
                    formatted_comments.append(f"{current_user}: {msg_joined}")
                    pure_comment_texts.append(msg_joined)
                current_user = text
                current_msg = []
            else:
                current_msg.append(text)

        if current_user and current_msg:
            msg_joined = " ".join(current_msg)
            formatted_comments.append(f"{current_user}: {msg_joined}")
            pure_comment_texts.append(msg_joined)
        elif current_msg:
            msg_joined = " ".join(current_msg)
            formatted_comments.append(msg_joined)
            pure_comment_texts.append(msg_joined)

    return raw_lines, formatted_comments, pure_comment_texts


def analyze_answer(question_lines, pure_comments):
    """Thuật toán quyết định đáp án (Đơn hoặc Nhiều lựa chọn) và danh sách phương án."""
    q_full_text = "\n".join(question_lines)
    q_upper = q_full_text.upper()

    has_e = False
    for line in question_lines:
        if re.match(r'^E[\.\)\:\s]', line.strip()):
            has_e = True
            break

    multi_hint_in_q = bool(re.search(
        r'(CHỌN\s*[234]|CHỌN\s*(?:HAI|BA|BỐN)|LỰA\s*CHỌN\s*(?:HAI|BA|2|3)|NHIỀU\s*ĐÁP\s*ÁN|CÁC\s*PHƯƠNG\s*ÁN\s*ĐÚNG)',
        q_upper
    ))

    single_scores = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    multi_scores = {}
    vote_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    negation_pattern = r'\b(KHONG|KHÔNG|KP|K\s*PHAI|K\s*PHẢI|KO\s*PHAI|KO\s*PHẢI|SAO\s*LẠI|SAO\s*K|SAI|LOẠI)\b'

    for comm in pure_comments:
        clean = comm.strip().upper()
        if not clean:
            continue

        m_multi = re.match(
            r'^(?:CHỌN|KEY|ĐÁP\s*ÁN|VOTE)?\s*([A-E])[\s,\.\-\+&/]*([A-E])(?:[\s,\.\-\+&/]*([A-E]))?(?:[\s,\.\-\+&/]*([A-E]))?$',
            clean
        )
        if m_multi:
            letters = [g for g in m_multi.groups() if g]
            unique_sorted = "".join(sorted(set(letters)))
            if len(unique_sorted) >= 2:
                multi_scores[unique_sorted] = multi_scores.get(unique_sorted, 0.0) + 1.5
                continue

        m_arrow = re.findall(r'(?:-+>|=+>)\s*([A-E])\b', clean)
        if m_arrow:
            for let in m_arrow:
                single_scores[let] += 2.5
                vote_counts[let] += 1
            continue

        if re.search(negation_pattern, clean):
            continue

        if clean in ["A", "B", "C", "D", "E"]:
            single_scores[clean] += 1.5
            vote_counts[clean] += 1
            continue

        m_start = re.match(r'^([A-E])(?:[\.\)\:\-\s,]|$)', clean)
        if m_start:
            let = m_start.group(1)
            single_scores[let] += 1.2
            vote_counts[let] += 1
            continue

        m_kw = re.search(r'\b(?:CHỌN|KEY|ĐÁP\s*ÁN|VOTE|CHỐT|CÂU\s*NÀY)\s*(?:LÀ\s*)?([A-E])\b', clean)
        if m_kw:
            let = m_kw.group(1)
            single_scores[let] += 1.5
            vote_counts[let] += 1
            continue

        words = clean.split()
        if 1 < len(words) <= 4 and words[-1] in ["A", "B", "C", "D", "E"]:
            let = words[-1]
            single_scores[let] += 1.0
            vote_counts[let] += 1

    if vote_counts["E"] > 0 or single_scores["E"] > 0:
        has_e = True

    options = ["A", "B", "C", "D", "E"] if has_e else ["A", "B", "C", "D"]
    if not has_e:
        single_scores.pop("E", None)
        vote_counts.pop("E", None)

    if multi_scores:
        best_multi, best_m_score = max(multi_scores.items(), key=lambda x: x[1])
        best_s_score = max(single_scores.values()) if single_scores else 0.0
        if best_m_score >= 2.0 or (multi_hint_in_q and best_m_score >= best_s_score):
            ans_list = list(best_multi)
            vote_summary = ", ".join([f"{k}: {int(round(v / 1.5))} vote" for k, v in multi_scores.items()])
            return "multi", options, ans_list, vote_counts, vote_summary

    best_single = max(single_scores, key=single_scores.get)
    if single_scores[best_single] > 0:
        vote_summary = ", ".join([f"{k}: {v}" for k, v in vote_counts.items() if v > 0])
        return "single", options, best_single, vote_counts, vote_summary

    opt_texts = {}
    for line in question_lines:
        m_opt = re.match(r'^([A-E])[\.\)\s]+(.+)$', line.strip())
        if m_opt:
            opt_texts[m_opt.group(1)] = m_opt.group(2).lower()

    if opt_texts and pure_comments:
        match_scores = {k: 0 for k in opt_texts}
        for comm in pure_comments:
            c_low = comm.lower()
            for k, o_txt in opt_texts.items():
                words = [w for w in o_txt.split() if len(w) > 2]
                if words:
                    matched = sum(1 for w in words if w in c_low)
                    if matched >= max(2, len(words) // 2):
                        match_scores[k] += matched
        best_match = max(match_scores, key=match_scores.get)
        if match_scores[best_match] > 0:
            return "single", options, best_match, vote_counts, f"Khớp nội dung phương án {best_match}"

    return "single", options, "A", vote_counts, "Mặc định A (chưa có bình luận)"


def build_practice_html_content(exam_title, quiz_items):
    json_data_str = json.dumps(quiz_items, ensure_ascii=False)
    return f'''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Luyện Tập Phản Xạ - {exam_title}</title>
    <style>
        :root {{
            --bg-page: #f0f2f5;
            --eos-navy: #0f172a;
            --eos-header: #1e293b;
            --eos-blue: #0284c7;
            --eos-blue-dark: #0369a1;
            --eos-blue-light: #e0f2fe;
            --eos-green: #16a34a;
            --eos-green-light: #dcfce7;
            --eos-red: #dc2626;
            --eos-red-light: #fee2e2;
            --eos-amber: #f59e0b;
            --eos-text: #1e293b;
            --eos-border: #cbd5e1;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        body {{
            background-color: var(--bg-page);
            color: var(--eos-text);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        /* Top Header */
        header.eos-top {{
            background: linear-gradient(135deg, var(--eos-navy) 0%, var(--eos-header) 100%);
            color: #ffffff;
            padding: 10px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            user-select: none;
            z-index: 10;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .top-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .mode-badge {{
            background: #8b5cf6;
            color: #fff;
            font-weight: 900;
            padding: 4px 10px;
            border-radius: 5px;
            font-size: 12px;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }}

        .top-title h1 {{
            font-size: 15px;
            font-weight: 700;
        }}

        .top-title p {{
            font-size: 12px;
            color: #94a3b8;
        }}

        /* Live Stats on Header */
        .header-stats {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .stat-pill {{
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 5px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
        }}

        .stat-pill.correct {{
            color: #4ade80;
            border-color: rgba(74, 222, 128, 0.3);
        }}

        .stat-pill.wrong {{
            color: #f87171;
            border-color: rgba(248, 113, 113, 0.3);
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .btn-top {{
            padding: 6px 14px;
            font-size: 12px;
            font-weight: 700;
            border-radius: 6px;
            border: 1px solid #475569;
            background: #334155;
            color: #e2e8f0;
            cursor: pointer;
            transition: all 0.15s;
        }}

        .btn-top:hover {{
            background: #475569;
        }}

        .btn-top.active {{
            background: var(--eos-blue);
            border-color: var(--eos-blue);
            color: #ffffff;
        }}

        .btn-reset {{
            background: #ef4444;
            border-color: #ef4444;
            color: #fff;
        }}

        .btn-reset:hover {{
            background: #dc2626;
        }}

        /* Main 2-column layout */
        .eos-layout {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        /* Left Column */
        .left-col {{
            width: 340px;
            background: #ffffff;
            border-right: 1px solid var(--eos-border);
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 15px;
            box-shadow: 2px 0 8px rgba(0,0,0,0.03);
            user-select: none;
            overflow-y: auto;
        }}

        .q-nav-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 12px;
        }}

        .q-number-title {{
            font-size: 18px;
            font-weight: 800;
            color: var(--eos-navy);
        }}

        .q-jump-select {{
            padding: 5px 8px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid var(--eos-border);
            border-radius: 5px;
            background: #f8fafc;
            color: #334155;
            cursor: pointer;
        }}

        .q-type-label {{
            font-size: 11px;
            font-weight: 700;
            padding: 5px 10px;
            border-radius: 4px;
            background: #f1f5f9;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }}

        .q-type-label.multi {{
            background: #fef3c7;
            color: #b45309;
            border: 1px solid #fde68a;
        }}

        /* Vertical Checkboxes */
        .checkbox-group {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin: 4px 0;
        }}

        .checkbox-row {{
            display: flex;
            align-items: center;
            padding: 12px 16px;
            background: #f8fafc;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s ease;
            gap: 14px;
        }}

        .checkbox-row:hover {{
            background: #f1f5f9;
            border-color: #94a3b8;
        }}

        .checkbox-row.checked {{
            background: var(--eos-blue-light);
            border-color: var(--eos-blue);
            box-shadow: 0 0 0 1px var(--eos-blue);
        }}

        .custom-box {{
            width: 24px;
            height: 24px;
            border: 2px solid #94a3b8;
            border-radius: 5px;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 900;
            color: transparent;
            transition: all 0.15s;
            flex-shrink: 0;
        }}

        .checkbox-row.checked .custom-box {{
            background: var(--eos-blue);
            border-color: var(--eos-blue);
            color: #ffffff;
        }}

        .checkbox-label {{
            font-size: 16px;
            font-weight: 700;
            color: #1e293b;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .key-hint {{
            font-size: 11px;
            color: #94a3b8;
            font-weight: 500;
        }}

        /* Instant Check Feedback States */
        .checkbox-row.correct {{
            background: var(--eos-green-light) !important;
            border-color: var(--eos-green) !important;
            box-shadow: 0 0 0 1px var(--eos-green) !important;
        }}

        .checkbox-row.correct .custom-box {{
            background: var(--eos-green) !important;
            border-color: var(--eos-green) !important;
            color: #fff !important;
        }}

        .checkbox-row.wrong {{
            background: var(--eos-red-light) !important;
            border-color: var(--eos-red) !important;
            box-shadow: 0 0 0 1px var(--eos-red) !important;
        }}

        .checkbox-row.wrong .custom-box {{
            background: var(--eos-red) !important;
            border-color: var(--eos-red) !important;
            color: #fff !important;
        }}

        /* Feedback Alert Banner inside Left Column */
        .feedback-banner {{
            display: none;
            padding: 12px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 700;
            line-height: 1.4;
            text-align: center;
            animation: popIn 0.2s ease-out;
        }}

        @keyframes popIn {{
            0% {{ transform: scale(0.95); opacity: 0; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}

        .feedback-banner.correct {{
            display: block;
            background: var(--eos-green-light);
            color: #15803d;
            border: 2px solid var(--eos-green);
        }}

        .feedback-banner.wrong {{
            display: block;
            background: var(--eos-red-light);
            color: #b91c1c;
            border: 2px solid var(--eos-red);
        }}

        .feedback-banner.warn {{
            display: block;
            background: #fef3c7;
            color: #b45309;
            border: 2px solid #f59e0b;
        }}

        /* Primary Action Button (Enter) */
        .btn-enter-action {{
            width: 100%;
            padding: 12px 16px;
            border-radius: 8px;
            border: none;
            background: var(--eos-blue);
            color: #ffffff;
            font-size: 14px;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
        }}

        .btn-enter-action:hover {{
            background: var(--eos-blue-dark);
        }}

        .btn-enter-action.next-state {{
            background: #0f172a;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.25);
        }}

        .btn-enter-action.next-state:hover {{
            background: #1e293b;
        }}

        .nav-controls {{
            display: flex;
            gap: 10px;
            margin-top: auto;
            padding-top: 12px;
            border-top: 1px solid #e2e8f0;
        }}

        .btn-nav {{
            flex: 1;
            padding: 9px;
            font-size: 12px;
            font-weight: 700;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
            background: #ffffff;
            color: #334155;
            cursor: pointer;
            transition: all 0.15s;
            text-align: center;
        }}

        .btn-nav:hover:not(:disabled) {{
            background: #f1f5f9;
            border-color: #94a3b8;
        }}

        .hotkey-hints {{
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 11px;
            color: #64748b;
            line-height: 1.55;
        }}

        .hotkey-hints strong {{
            color: var(--eos-navy);
        }}

        /* Right Column */
        .right-col {{
            flex: 1;
            background: #f8fafc;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            padding: 24px;
            align-items: center;
            gap: 20px;
        }}

        .image-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            max-width: 1100px;
            width: 100%;
            display: flex;
            justify-content: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }}

        .image-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}

        .review-card {{
            display: none;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 5px solid var(--eos-red);
            border-radius: 8px;
            padding: 20px;
            max-width: 1100px;
            width: 100%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            animation: popIn 0.25s ease-out;
        }}

        .review-card.visible {{
            display: block;
        }}

        .review-card.correct-mode {{
            border-left-color: var(--eos-green);
        }}

        .review-title {{
            font-size: 15px;
            font-weight: 800;
            color: var(--eos-navy);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .review-explanation {{
            font-size: 14px;
            line-height: 1.6;
            color: #334155;
            margin-bottom: 14px;
        }}

        .fu-box {{
            background: #0f172a;
            color: #e2e8f0;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 12px;
        }}

        .fu-title {{
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 6px;
        }}
    </style>
</head>
<body>

    <header class="eos-top">
        <div class="top-brand">
            <span class="mode-badge">⚡ LUYỆN TẬP</span>
            <div class="top-title">
                <h1>{exam_title}</h1>
                <p>Chế độ kiểm tra tức thì: Chọn đáp án + Ấn Enter</p>
            </div>
        </div>

        <div class="header-stats">
            <div class="stat-pill correct">✓ Đúng: <span id="statCorrect">0</span></div>
            <div class="stat-pill wrong">✕ Sai: <span id="statWrong">0</span></div>
            <div class="stat-pill">📊 Đã làm: <span id="statProgress">0/{len(quiz_items)}</span></div>
        </div>

        <div class="header-actions">
            <button class="btn-top active" id="btnFilterAll" onclick="setFilterMode('all')">Tất cả ({len(quiz_items)})</button>
            <button class="btn-top" id="btnFilterWrong" onclick="setFilterMode('wrong')">Câu làm SAI (<span id="wrongCountBtn">0</span>)</button>
            <button class="btn-top" onclick="retryCurrentQuestion()" title="Làm lại câu đang mở">↺ Làm lại câu này</button>
            <button class="btn-top btn-reset" onclick="resetAll()">↺ Reset từ đầu</button>
        </div>
    </header>

    <div class="eos-layout">
        <aside class="left-col">
            <div class="q-nav-header">
                <div class="q-number-title" id="qNumberTitle">Câu 1 / {len(quiz_items)}</div>
                <select class="q-jump-select" id="jumpSelect" onchange="jumpToQuestion(parseInt(this.value))">
                </select>
            </div>

            <div class="q-type-label" id="qTypeLabel">DẠNG: CHỌN 1 ĐÁP ÁN</div>

            <div class="checkbox-group" id="checkboxGroup">
            </div>

            <div class="feedback-banner" id="feedbackBanner"></div>

            <button class="btn-enter-action" id="btnEnterAction" onclick="handleEnterAction()">
                <span>↵</span> KIỂM TRA ĐÁP ÁN (ENTER)
            </button>

            <div class="nav-controls">
                <button class="btn-nav" id="btnPrev" onclick="navigate(-1)">◀ Câu trước</button>
                <button class="btn-nav" id="btnNext" onclick="navigate(1)">Câu sau ▶</button>
            </div>

            <div class="hotkey-hints">
                <strong>⌨ Cách dùng siêu tốc:</strong><br>
                1. Ấn <strong>1, 2, 3, 4</strong> (hoặc <strong>A, B, C, D</strong>) để chọn.<br>
                2. Ấn <strong>Enter</strong> để kiểm tra:<br>
                &nbsp;&nbsp;• <span style="color:#16a34a;font-weight:700;">Đúng (Xanh)</span>: Tự động nhảy sang câu kế.<br>
                &nbsp;&nbsp;• <span style="color:#dc2626;font-weight:700;">Sai (Đỏ)</span>: Hiện đáp án & đối chiếu, ấn <strong>Enter</strong> hoặc <strong>➡</strong> để qua câu tiếp.
            </div>
        </aside>

        <main class="right-col">
            <div class="image-card">
                <img id="questionImage" src="" alt="Đang tải câu hỏi...">
            </div>

            <div class="review-card" id="reviewCard">
                <div class="review-title" id="reviewTitle">
                    <span>💡 ĐÁP ÁN & ĐỐI CHIẾU CỘNG ĐỒNG:</span>
                </div>
                <div class="review-explanation" id="reviewExplanation"></div>
                <div class="fu-box">
                    <div class="fu-title">Thảo luận từ sinh viên trên FuOverflow:</div>
                    <div id="fuComments"></div>
                </div>
            </div>
        </main>
    </div>

    <script>
        const QUESTIONS = {json_data_str};

        let currentIndex = 0;
        let userAnswers = {{}};     // qid -> "A" or ["A", "B"]
        let checkedResults = {{}};  // qid -> {{ isRight: true/false }}
        let filterMode = 'all';     // 'all' or 'wrong'
        let autoNextTimer = null;

        function init() {{
            populateJumpSelect();
            loadQuestion(0);
            setupKeyboard();
            updateHeaderStats();
        }}

        function clearAutoTimer() {{
            if (autoNextTimer) {{
                clearTimeout(autoNextTimer);
                autoNextTimer = null;
            }}
        }}

        function getWrongIndices() {{
            const list = [];
            QUESTIONS.forEach((q, idx) => {{
                if (checkedResults[q.id] && !checkedResults[q.id].isRight) {{
                    list.push(idx);
                }}
            }});
            return list;
        }}

        function updateHeaderStats() {{
            let cCount = 0;
            let wCount = 0;
            QUESTIONS.forEach(q => {{
                const res = checkedResults[q.id];
                if (res) {{
                    if (res.isRight) cCount++;
                    else wCount++;
                }}
            }});
            document.getElementById('statCorrect').innerText = cCount;
            document.getElementById('statWrong').innerText = wCount;
            document.getElementById('statProgress').innerText = (cCount + wCount) + '/' + QUESTIONS.length;
            document.getElementById('wrongCountBtn').innerText = wCount;

            // Update jumpSelect symbols
            const sel = document.getElementById('jumpSelect');
            Array.from(sel.options).forEach((opt, idx) => {{
                const q = QUESTIONS[idx];
                const res = checkedResults[q.id];
                let prefix = '';
                if (res) prefix = res.isRight ? '✓ ' : '✕ ';
                opt.innerText = prefix + 'Câu ' + q.id;
            }});
        }}

        function populateJumpSelect() {{
            const sel = document.getElementById('jumpSelect');
            sel.innerHTML = '';
            QUESTIONS.forEach((q, idx) => {{
                const opt = document.createElement('option');
                opt.value = idx;
                opt.innerText = 'Câu ' + q.id;
                sel.appendChild(opt);
            }});
        }}

        function loadQuestion(idx, justCheckedRight = false) {{
            if (idx < 0 || idx >= QUESTIONS.length) return;
            if (!justCheckedRight) clearAutoTimer();

            currentIndex = idx;
            const q = QUESTIONS[idx];
            const checkInfo = checkedResults[q.id];
            const isCheckedDone = Boolean(checkInfo);

            document.getElementById('qNumberTitle').innerText = 'Câu ' + q.id + ' / ' + QUESTIONS.length;
            document.getElementById('jumpSelect').value = idx;

            const typeLabel = document.getElementById('qTypeLabel');
            if (q.type === 'multi') {{
                typeLabel.innerText = 'DẠNG: CHỌN NHIỀU ĐÁP ÁN';
                typeLabel.className = 'q-type-label multi';
            }} else {{
                typeLabel.innerText = 'DẠNG: CHỌN 1 ĐÁP ÁN';
                typeLabel.className = 'q-type-label';
            }}

            const container = document.getElementById('checkboxGroup');
            container.innerHTML = '';

            const userAns = userAnswers[q.id];

            q.options.forEach((opt, optIdx) => {{
                const row = document.createElement('div');
                row.className = 'checkbox-row';

                const isSel = Array.isArray(userAns) ? userAns.includes(opt) : userAns === opt;
                if (isSel) row.classList.add('checked');

                if (isCheckedDone) {{
                    const isCorrectOpt = Array.isArray(q.correct) ? q.correct.includes(opt) : q.correct === opt;
                    if (isCorrectOpt) {{
                        row.classList.add('correct');
                    }} else if (isSel) {{
                        row.classList.add('wrong');
                    }}
                }}

                row.onclick = () => {{
                    if (isCheckedDone) return;
                    toggleOption(q.id, opt);
                }};

                const box = document.createElement('div');
                box.className = 'custom-box';
                if (isCheckedDone) {{
                    const isCorrectOpt = Array.isArray(q.correct) ? q.correct.includes(opt) : q.correct === opt;
                    box.innerText = isCorrectOpt ? '✓' : (isSel ? '✕' : '');
                }} else {{
                    box.innerText = '✓';
                }}

                const label = document.createElement('div');
                label.className = 'checkbox-label';
                label.innerHTML = 'Đáp án ' + opt + ' <span class="key-hint">(Phím ' + (optIdx + 1) + ' / ' + opt + ')</span>';

                row.appendChild(box);
                row.appendChild(label);
                container.appendChild(row);
            }});

            // Feedback Banner & Enter Button
            const fb = document.getElementById('feedbackBanner');
            const btnEnter = document.getElementById('btnEnterAction');
            const imgEl = document.getElementById('questionImage');
            const revCard = document.getElementById('reviewCard');

            if (isCheckedDone) {{
                const ansStr = Array.isArray(q.correct) ? q.correct.join(', ') : q.correct;
                if (checkInfo.isRight) {{
                    fb.className = 'feedback-banner correct';
                    fb.innerHTML = justCheckedRight
                        ? '✓ CHÍNH XÁC! Đang sang câu tiếp theo...'
                        : '✓ CHÍNH XÁC! (Đáp án: ' + ansStr + ')';
                    imgEl.src = q.full_img;
                    revCard.className = 'review-card visible correct-mode';
                }} else {{
                    fb.className = 'feedback-banner wrong';
                    fb.innerHTML = '✕ SAI RỒI! Đáp án đúng là: <strong>' + ansStr + '</strong><br><span style="font-size:11px;font-weight:600;">Ấn Enter hoặc ➡ để sang câu tiếp theo</span>';
                    imgEl.src = q.full_img;
                    revCard.className = 'review-card visible';
                }}

                document.getElementById('reviewExplanation').innerText = q.explanation;
                const commEl = document.getElementById('fuComments');
                if (q.comments && q.comments.length > 0) {{
                    commEl.innerHTML = q.comments.map(c => '<div style="margin: 2px 0;">• ' + c + '</div>').join('');
                }} else {{
                    commEl.innerHTML = '<i>Không có bình luận.</i>';
                }}

                btnEnter.className = 'btn-enter-action next-state';
                btnEnter.innerHTML = '<span>↵</span> CÂU TIẾP THEO (ENTER / ➡)';
            }} else {{
                fb.className = 'feedback-banner';
                fb.innerHTML = '';
                imgEl.src = q.question_img;
                revCard.className = 'review-card';

                btnEnter.className = 'btn-enter-action';
                btnEnter.innerHTML = '<span>↵</span> KIỂM TRA ĐÁP ÁN (ENTER)';
            }}

            // Nav buttons
            const btnPrev = document.getElementById('btnPrev');
            const btnNext = document.getElementById('btnNext');
            const wrongIndices = getWrongIndices();

            if (filterMode === 'wrong') {{
                btnPrev.disabled = (wrongIndices.length <= 1);
                btnNext.disabled = (wrongIndices.length <= 1);
                btnPrev.innerHTML = '◀ Câu sai trước';
                btnNext.innerHTML = 'Câu sai tiếp ▶';
            }} else {{
                btnPrev.disabled = false;
                btnNext.disabled = false;
                btnPrev.innerHTML = (currentIndex === 0) ? '◀ Về câu ' + QUESTIONS.length : '◀ Câu trước';
                btnNext.innerHTML = (currentIndex === QUESTIONS.length - 1) ? 'Về câu 1 ↺' : 'Câu sau ▶';
            }}
        }}

        function toggleOption(qid, opt) {{
            const q = QUESTIONS.find(x => x.id === qid);
            if (!q || checkedResults[qid]) return;

            if (q.type === 'multi') {{
                let current = userAnswers[qid] || [];
                if (!Array.isArray(current)) current = [current];
                if (current.includes(opt)) {{
                    current = current.filter(x => x !== opt);
                }} else {{
                    current.push(opt);
                }}
                current.sort();
                if (current.length === 0) delete userAnswers[qid];
                else userAnswers[qid] = current;
            }} else {{
                if (userAnswers[qid] === opt) {{
                    delete userAnswers[qid];
                }} else {{
                    userAnswers[qid] = opt;
                }}
            }}
            loadQuestion(currentIndex);
        }}

        function handleEnterAction() {{
            const q = QUESTIONS[currentIndex];

            // Nếu câu này đã được kiểm tra rồi -> Ấn Enter sẽ nhảy sang câu tiếp theo
            if (checkedResults[q.id]) {{
                clearAutoTimer();
                navigate(1);
                return;
            }}

            // Kiểm tra xem user đã chọn đáp án chưa
            const uAns = userAnswers[q.id];
            const hasSelection = uAns && (Array.isArray(uAns) ? uAns.length > 0 : true);
            if (!hasSelection) {{
                const fb = document.getElementById('feedbackBanner');
                fb.className = 'feedback-banner warn';
                fb.innerHTML = '⚠️ Bạn hãy chọn đáp án trước khi ấn Enter nhé!';
                return;
            }}

            // Chấm điểm câu hiện tại
            let isRight = false;
            if (q.type === 'multi') {{
                if (Array.isArray(uAns) && Array.isArray(q.correct)) {{
                    isRight = ([...uAns].sort().join(',') === [...q.correct].sort().join(','));
                }}
            }} else {{
                isRight = (uAns === q.correct);
            }}

            checkedResults[q.id] = {{ isRight: isRight }};
            updateHeaderStats();

            if (isRight) {{
                loadQuestion(currentIndex, true);
                clearAutoTimer();
                autoNextTimer = setTimeout(() => {{
                    navigate(1);
                }}, 600);
            }} else {{
                loadQuestion(currentIndex, false);
            }}
        }}

        function navigate(dir) {{
            clearAutoTimer();

            if (filterMode === 'wrong') {{
                const wrongIndices = getWrongIndices();
                if (wrongIndices.length === 0) {{
                    setFilterMode('all');
                    return;
                }}
                const currentPos = wrongIndices.indexOf(currentIndex);
                let nextPos = (currentPos === -1) ? 0 : currentPos + dir;
                if (nextPos >= wrongIndices.length) nextPos = 0;
                if (nextPos < 0) nextPos = wrongIndices.length - 1;
                loadQuestion(wrongIndices[nextPos]);
                return;
            }}

            let next = currentIndex + dir;
            if (next >= QUESTIONS.length) {{
                next = 0;
            }} else if (next < 0) {{
                next = QUESTIONS.length - 1;
            }}
            loadQuestion(next);
        }}

        function jumpToQuestion(idx) {{
            clearAutoTimer();
            loadQuestion(idx);
        }}

        function retryCurrentQuestion() {{
            clearAutoTimer();
            const q = QUESTIONS[currentIndex];
            delete userAnswers[q.id];
            delete checkedResults[q.id];
            updateHeaderStats();
            loadQuestion(currentIndex);
        }}

        function setFilterMode(mode) {{
            clearAutoTimer();
            if (mode === 'wrong') {{
                const wrongIndices = getWrongIndices();
                if (wrongIndices.length === 0) {{
                    alert('Hiện chưa có câu nào làm sai!');
                    return;
                }}
                filterMode = 'wrong';
                document.getElementById('btnFilterAll').className = 'btn-top';
                document.getElementById('btnFilterWrong').className = 'btn-top active';
                if (!wrongIndices.includes(currentIndex)) {{
                    loadQuestion(wrongIndices[0]);
                }} else {{
                    loadQuestion(currentIndex);
                }}
            }} else {{
                filterMode = 'all';
                document.getElementById('btnFilterAll').className = 'btn-top active';
                document.getElementById('btnFilterWrong').className = 'btn-top';
                loadQuestion(currentIndex);
            }}
        }}

        function resetAll() {{
            if (!confirm('Bạn có muốn xóa kết quả hiện tại để luyện tập lại từ câu 1?')) return;
            clearAutoTimer();
            userAnswers = {{}};
            checkedResults = {{}};
            filterMode = 'all';
            document.getElementById('btnFilterAll').className = 'btn-top active';
            document.getElementById('btnFilterWrong').className = 'btn-top';
            updateHeaderStats();
            loadQuestion(0);
        }}

        function setupKeyboard() {{
            window.addEventListener('keydown', (e) => {{
                const q = QUESTIONS[currentIndex];
                const key = e.key.toUpperCase();

                if (e.key === 'Enter') {{
                    e.preventDefault();
                    handleEnterAction();
                }} else if (['1', '2', '3', '4', '5'].includes(key)) {{
                    const numIdx = parseInt(key) - 1;
                    if (numIdx < q.options.length && !checkedResults[q.id]) {{
                        toggleOption(q.id, q.options[numIdx]);
                    }}
                }} else if (['A', 'B', 'C', 'D', 'E'].includes(key)) {{
                    if (q.options.includes(key) && !checkedResults[q.id]) {{
                        toggleOption(q.id, key);
                    }}
                }} else if (e.key === 'ArrowLeft') {{
                    e.preventDefault();
                    navigate(-1);
                }} else if (e.key === 'ArrowRight') {{
                    e.preventDefault();
                    navigate(1);
                }} else if (key === 'R') {{
                    retryCurrentQuestion();
                }}
            }});
        }}

        window.onload = init;
    </script>
</body>
</html>'''


def process_practice_folder(exam_dir, ocr=None):
    """Tạo file luyen_tap.html (chế độ kiểm tra tức thì sau mỗi câu). Tái sử dụng quiz_data.json nếu đã có."""
    exam_dir = os.path.abspath(exam_dir)
    folder_name = os.path.basename(exam_dir).replace('\xa0', ' ')
    quiz_path = os.path.join(exam_dir, "quiz_data.json")
    clean_q_dir = os.path.join(exam_dir, "questions")
    html_out_path = os.path.join(exam_dir, "luyen_tap.html")

    # Nếu đã có sẵn quiz_data.json và thư mục questions/ thì tái sử dụng ngay lập tức (0.1s)
    if os.path.exists(quiz_path) and os.path.isdir(clean_q_dir):
        try:
            with open(quiz_path, "r", encoding="utf-8") as f:
                quiz_items = json.load(f)
            if quiz_items:
                html_str = build_practice_html_content(folder_name, quiz_items)
                with open(html_out_path, "w", encoding="utf-8") as f:
                    f.write(html_str)
                print(f"⚡ [Tạo siêu tốc từ data có sẵn] -> Đã tạo luyen_tap.html cho: {folder_name}")
                return True, ocr
        except Exception as e:
            print(f"  [!] Lỗi đọc quiz_data.json cũ ({e}), sẽ tiến hành quét lại từ ảnh...")

    # Nếu chưa có data thì thực hiện quét ảnh OCR đầy đủ
    png_files = sorted([
        f for f in os.listdir(exam_dir)
        if re.match(r'^Q\d+\.png$', f, re.IGNORECASE)
    ])

    if not png_files:
        print(f"[!] Không tìm thấy ảnh câu hỏi (Q001.png...) trong: {exam_dir}")
        return False, ocr

    if ocr is None:
        ocr = RapidOCR()

    print("=" * 65)
    print(f"🚀 ĐANG QUÉT ẢNH & TẠO FILE LUYỆN TẬP: {folder_name}")
    print(f"📂 Đường dẫn: {exam_dir}")
    print(f"📊 Số lượng câu hỏi: {len(png_files)} câu")
    print("=" * 65)

    os.makedirs(clean_q_dir, exist_ok=True)
    extracted_results = []
    quiz_items = []
    t0 = time.time()

    for idx, filename in enumerate(png_files, start=1):
        m_num = re.search(r'\d+', filename)
        q_num = int(m_num.group(0)) if m_num else idx

        filepath = os.path.join(exam_dir, filename)
        img = read_image_utf8(filepath)
        if img is None:
            print(f"  [!] Lỗi đọc ảnh: {filename}")
            continue

        h, w, _ = img.shape
        split_x = detect_split_x(img)

        q_crop = img[:, :split_x]
        save_image_utf8(os.path.join(clean_q_dir, filename), q_crop)

        q_ocr_res, _ = ocr(q_crop)
        question_lines = []
        if q_ocr_res:
            for box, text, score in q_ocr_res:
                text = text.strip()
                if text in ["MULTIPLECHOICE", "MULTIPLE CHOICE", "Nguon: FuOverflow.com"] or "FuOverflow" in text:
                    continue
                question_lines.append(text)

        comment_panel = img[:, split_x:w]
        raw_lines, formatted_comments, pure_comments = extract_and_vote_comments(comment_panel, ocr)
        q_type, options, correct_ans, vote_counts, vote_summary = analyze_answer(question_lines, pure_comments)

        ans_str = ", ".join(correct_ans) if isinstance(correct_ans, list) else str(correct_ans)
        explanation = f"Đáp án chuẩn: {ans_str}. Thống kê thảo luận cộng đồng FuOverflow: {vote_summary}."

        extracted_results.append({
            "id": q_num,
            "filename": filename,
            "question_text": "\n".join(question_lines),
            "comment_lines": raw_lines,
            "formatted_comments": formatted_comments,
            "votes": vote_counts,
            "type": q_type,
            "options": options,
            "correct": correct_ans
        })

        quiz_items.append({
            "id": q_num,
            "type": q_type,
            "options": options,
            "correct": correct_ans,
            "question_img": f"questions/{filename}",
            "full_img": filename,
            "comments": formatted_comments if formatted_comments else raw_lines,
            "vote_summary": vote_summary,
            "explanation": explanation
        })

        print(f"  [{idx:02d}/{len(png_files):02d}] {filename} -> Đáp án: {ans_str:<8} | Dạng: {q_type:<6} | ({vote_summary})")

    extracted_path = os.path.join(exam_dir, "extracted_data.json")
    with open(extracted_path, "w", encoding="utf-8") as f:
        json.dump(extracted_results, f, ensure_ascii=False, indent=2)

    with open(quiz_path, "w", encoding="utf-8") as f:
        json.dump(quiz_items, f, ensure_ascii=False, indent=2)

    html_str = build_practice_html_content(folder_name, quiz_items)
    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    elapsed = time.time() - t0
    print("-" * 65)
    print(f"✅ HOÀN TẤT [{folder_name}] trong {elapsed:.1f}s!")
    print(f"   • Đã tạo: questions/ ({len(quiz_items)} ảnh sạch)")
    print(f"   • Đã tạo: extracted_data.json & quiz_data.json")
    print(f"   • Đã tạo: luyen_tap.html (Chế độ kiểm tra tức thì sau mỗi câu)")
    print("=" * 65 + "\n")
    return True, ocr


def find_all_exam_folders(root_dir):
    exam_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath).lower() == "questions":
            continue
        if any(re.match(r'^Q0*1\.png$', f, re.IGNORECASE) for f in filenames):
            exam_folders.append(dirpath)
    return sorted(exam_folders)


def main():
    ocr = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if os.path.isdir(arg):
                _, ocr = process_practice_folder(arg, ocr)
        return

    cwd = os.getcwd()
    has_q_images = any(re.match(r'^Q0*1\.png$', f, re.IGNORECASE) for f in os.listdir(cwd))
    if has_q_images:
        process_practice_folder(cwd)
        return

    exam_folders = find_all_exam_folders(cwd)
    if not exam_folders:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir != cwd:
            exam_folders = find_all_exam_folders(script_dir)

    if not exam_folders:
        print("[!] Không tìm thấy folder đề nào chứa ảnh Q001.png!")
        return

    print("\n" + "=" * 75)
    print("⚡ TOOL TẠO WEB LUYỆN TẬP PHẢN XẠ (CHỌN ĐÁP ÁN + ENTER HIỆN ĐÚNG/SAI LUÔN)")
    print("=" * 75)
    for i, folder in enumerate(exam_folders, start=1):
        rel = os.path.relpath(folder, cwd).replace('\xa0', ' ')
        has_practice = os.path.exists(os.path.join(folder, "luyen_tap.html"))
        has_data = os.path.exists(os.path.join(folder, "quiz_data.json"))
        if has_practice:
            status = "✅ [Đã có luyen_tap.html]"
        elif has_data:
            status = "⚡ [Chưa có - Đã có data, tạo 0.1s]"
        else:
            status = "⏳ [Chưa tạo]"
        print(f"  {i:2d}. {status:<34} {rel}")

    print("-" * 75)
    print("  • Nhập số thứ tự đề muốn tạo luyen_tap.html (VD: 1, hoặc nhiều đề: 1,7,8)")
    print("  • Nhập 'NEW' để tạo cho tất cả các đề chưa có luyen_tap.html")
    print("  • Nhập 'ALL' để tạo/cập nhật luyen_tap.html cho TẤT CẢ các đề")
    print("=" * 75)

    choice = input("\n👉 Nhập lựa chọn của bạn: ").strip().upper()
    if not choice:
        return

    selected_folders = []
    if choice == "ALL":
        selected_folders = exam_folders
    elif choice == "NEW":
        selected_folders = [f for f in exam_folders if not os.path.exists(os.path.join(f, "luyen_tap.html"))]
    else:
        parts = [p.strip() for p in re.split(r'[,\s]+', choice) if p.strip().isdigit()]
        for p in parts:
            idx = int(p) - 1
            if 0 <= idx < len(exam_folders):
                selected_folders.append(exam_folders[idx])

    if not selected_folders:
        print("[!] Không có đề nào được chọn.")
        return

    for folder in selected_folders:
        _, ocr = process_practice_folder(folder, ocr)


if __name__ == "__main__":
    main()

import re
from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents import Document

TIME_RE = re.compile(r"(?P<start>\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}\.\d{3})")
SPEAKER_RE = re.compile(r"^\[(?P<spk>[^\]]+)\]\s*:\s*(?P<text>.*)$")

def mmssms_to_seconds(s: str) -> float:
    """
    Chuyển 'MM:SS.mmm' thành số giây float. Ví dụ: '01:26.721' -> 86.721
    """
    mm, rest = s.split(":")
    ss, ms = rest.split(".")
    return int(mm) * 60 + int(ss) + int(ms) / 1000.0


class Utterance(BaseModel):
    start_str: str
    end_str: str
    start_s: float
    end_s: float
    speaker: Optional[str]
    text: str

# ===== 2) Parse transcript thô thành các Utterance =====
def parse_transcript_to_utterances(raw: str) -> List[Utterance]:
    """
    Giả định cấu trúc block:
    <time>
    [SPEAKER_X]: <text>

    Có thể có nhiều dòng text sau dòng speaker, sẽ được gộp.
    """
    lines = [ln.strip() for ln in raw.splitlines()]
    i = 0
    utterances: List[Utterance] = []

    while i < len(lines):
        # Tìm dòng thời gian
        m_time = TIME_RE.match(lines[i])
        if not m_time:
            i += 1
            continue
        start_str = m_time.group("start")
        end_str = m_time.group("end")
        start_s = mmssms_to_seconds(start_str)
        end_s = mmssms_to_seconds(end_str)

        # Dòng kế tiếp kỳ vọng là [SPEAKER]: text
        i += 1
        if i >= len(lines):
            break

        m_spk = SPEAKER_RE.match(lines[i])
        speaker = None
        text_parts = []

        if m_spk:
            speaker = m_spk.group("spk")
            first_text = m_spk.group("text").strip()
            if first_text:
                text_parts.append(first_text)
            i += 1
            # Gộp thêm các dòng văn bản tiếp theo cho đến khi gặp block thời gian mới hoặc hết file
            while i < len(lines) and not TIME_RE.match(lines[i]):
                if lines[i] and not lines[i].startswith("[") and not lines[i].endswith("-->"):
                    text_parts.append(lines[i])
                # Nếu gặp dòng speaker mới ngay sau đó mà không có time mới, vẫn coi là phần văn bản
                elif SPEAKER_RE.match(lines[i]) and not TIME_RE.match(lines[i]):
                    # thường transcript chuẩn sẽ không như vậy, nên ta dừng lại
                    break
                i += 1
        else:
            # Không có nhãn speaker, coi toàn bộ dòng kế là text
            if lines[i]:
                text_parts.append(lines[i])
            i += 1
            while i < len(lines) and not TIME_RE.match(lines[i]):
                text_parts.append(lines[i])
                i += 1

        text = " ".join(tp.strip() for tp in text_parts if tp.strip())
        utterances.append(
            Utterance(
                start_str=start_str,
                end_str=end_str,
                start_s=start_s,
                end_s=end_s,
                speaker=speaker,
                text=text
            )
        )

    return utterances

def utterances_to_documents(utterances: List[Utterance],
                            conversation_id: Optional[str] = None) -> List[Document]:
    docs = []
    for idx, u in enumerate(utterances, start=1):
        metadata = {
            "speaker": u.speaker or "UNKNOWN",
            "start_seconds": u.start_s,
            "end_seconds": u.end_s,
            "duration_seconds": round(u.end_s - u.start_s, 3),
            "turn_id": idx
        }
        if conversation_id:
            metadata["conversation_id"] = conversation_id
        docs.append(Document(page_content=u.text, metadata=metadata))
    return docs

def utterances_to_documents_no_speakers(transcript, start, end, idx) -> Document:
    metadata = {
        "speaker": "UNKNOWN",
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": round(end - start, 3),
        "turn_id": idx+1,
        "conversation_id": None
    }

    return Document(page_content=transcript, metadata=metadata)
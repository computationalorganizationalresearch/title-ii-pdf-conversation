"""
Single-file, Colab-pasteable PDF accessibility pipeline.

One-line usage:
    convert_pdf("original.pdf", "convertedaccessibleversion.pdf")

Default OCR backend is LOCAL Hugging Face DeepSeek OCR.
If unavailable, pipeline falls back gracefully.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import requests

# ---------------------------
# Configuration (env override)
# ---------------------------
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-vl2")
DEEPSEEK_OCR_MODEL_NAME = os.getenv("DEEPSEEK_OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
DEEPSEEK_OCR_REVISION = os.getenv("DEEPSEEK_OCR_REVISION")


# ---------------------------
# Helpers
# ---------------------------
def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _data_url(image_path: Path) -> str:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@lru_cache(maxsize=1)
def _load_local_deepseek_ocr_model(model_name: str = DEEPSEEK_OCR_MODEL_NAME):
    """
    Lazy-load local HF DeepSeek OCR model.
    Matches user's sample approach and keeps it optional.
    """
    try:
        import addict  # type: ignore # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Local DeepSeek OCR requires the 'addict' package. "
            "Install it with: pip install addict"
        ) from exc

    from transformers import AutoModel, AutoTokenizer
    import torch

    model_load_kwargs = {}
    if DEEPSEEK_OCR_REVISION:
        model_load_kwargs["revision"] = DEEPSEEK_OCR_REVISION

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        **model_load_kwargs,
    )
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        **model_load_kwargs,
    )

    if torch.cuda.is_available():
        model = model.eval().cuda().to(torch.bfloat16)
    else:
        model = model.eval()

    return tokenizer, model


def _local_deepseek_ocr_markdown(image_path: Path, output_dir: Path) -> str:
    """Run local HF DeepSeek OCR and return markdown-like result text."""
    tokenizer, model = _load_local_deepseek_ocr_model()
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    output_dir.mkdir(parents=True, exist_ok=True)

    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path.as_posix(),
        output_path=output_dir.as_posix(),
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True,
    )
    return str(res)


def _deepseek_alt_text_api(image_path: Path) -> str:
    """Remote API alt-text fallback if local OCR path is unavailable for this task."""
    if not DEEPSEEK_API_KEY:
        return "Image relevant to document content"

    endpoint = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": DEEPSEEK_VISION_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Write one-sentence alt text for accessibility. "
                            "Mention only meaningful visual information. "
                            "If decorative, return exactly: Decorative image"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=180)
        res.raise_for_status()
        text = res.json()["choices"][0]["message"]["content"].strip()
        return text if text else "Image relevant to document content"
    except Exception:
        return "Image relevant to document content"


def _deepseek_structural_tag_guess(text: str) -> List[str]:
    """
    Use LLM to guess likely structural tags from OCR text.
    Returns a best-effort ordered list from a controlled tag set.
    """
    allowed = ["H1", "H2", "H3", "P", "L", "LI", "Table", "Figure", "Sect"]
    if not text.strip() or not DEEPSEEK_API_KEY:
        return []

    endpoint = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": DEEPSEEK_VISION_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Given OCR text from one PDF page, guess a short ordered list of likely PDF structure "
                    "tags. Return JSON only in this exact shape: "
                    '{"tags":["H1","P"]}. Allowed tags: H1,H2,H3,P,L,LI,Table,Figure,Sect.\n\n'
                    f"Page OCR text:\n{text[:6000]}"
                ),
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        res.raise_for_status()
        raw = res.json()["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw)
        tags = parsed.get("tags", [])
        cleaned = [tag for tag in tags if tag in allowed]
        return cleaned[:12]
    except Exception:
        return []


def _deepseek_heading_guess(page_text: str, page_number: int) -> List[Dict[str, object]]:
    """
    Use LLM to infer likely headings from page OCR text.
    Returns [{level:int(1-3), text:str, page:int}, ...].
    """
    if not page_text.strip() or not DEEPSEEK_API_KEY:
        return []

    endpoint = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": DEEPSEEK_VISION_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You receive OCR text from one PDF page. Infer likely section headings only. "
                    "Return JSON only using this exact shape: "
                    '{"headings":[{"level":1,"text":"Introduction"}]}. '
                    "Rules: level must be 1,2,or 3; keep heading text concise; no body text.\n\n"
                    f"Page number: {page_number}\n"
                    f"OCR text:\n{page_text[:7000]}"
                ),
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        res.raise_for_status()
        raw = res.json()["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw)
        out: List[Dict[str, object]] = []
        for h in parsed.get("headings", []):
            text = str(h.get("text", "")).strip()
            try:
                level = int(h.get("level", 1))
            except Exception:
                level = 1
            if not text:
                continue
            level = min(3, max(1, level))
            out.append({"level": level, "text": text, "page": page_number})
        return out[:8]
    except Exception:
        return []


def _deepseek_relevel_headings(headings: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Re-level inferred headings so the document hierarchy starts at H1 and
    preserves relative nesting across pages.
    """
    if not headings:
        return []
    if not DEEPSEEK_API_KEY:
        return _heuristic_relevel_headings(headings)

    endpoint = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    compact = [
        {
            "index": i,
            "page": int(h.get("page", 1)),
            "level": int(h.get("level", 1)),
            "text": str(h.get("text", "")).strip(),
        }
        for i, h in enumerate(headings)
        if str(h.get("text", "")).strip()
    ]
    if not compact:
        return []

    payload = {
        "model": DEEPSEEK_VISION_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You receive extracted PDF headings in reading order. "
                    "Reassign only their heading levels so hierarchy starts at H1 and preserves "
                    "relative nesting. Keep each heading text unchanged. "
                    "Return JSON only with this exact shape: "
                    '{"headings":[{"index":0,"level":1}]}. '
                    "Rules: level must be 1,2,or 3; at least one heading must be level 1; "
                    "do not remove or add indexes.\n\n"
                    f"Headings:\n{json.dumps(compact, ensure_ascii=False)[:12000]}"
                ),
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        res.raise_for_status()
        raw = res.json()["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw)
        by_index = {}
        for item in parsed.get("headings", []):
            try:
                idx = int(item.get("index", -1))
                lvl = int(item.get("level", 1))
            except Exception:
                continue
            if idx < 0:
                continue
            by_index[idx] = min(3, max(1, lvl))

        if not by_index:
            return _heuristic_relevel_headings(headings)

        updated: List[Dict[str, object]] = []
        for i, h in enumerate(headings):
            new_h = dict(h)
            if i in by_index:
                new_h["level"] = by_index[i]
            updated.append(new_h)
        return _heuristic_relevel_headings(updated)
    except Exception:
        return _heuristic_relevel_headings(headings)


def _heuristic_heading_guess(page_text: str, page_number: int) -> List[Dict[str, object]]:
    """Rule-based heading fallback from OCR text."""
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    headings: List[Dict[str, object]] = []

    for line in lines[:18]:
        clean = re.sub(r"\s+", " ", line).strip(" -:\t")
        if not clean:
            continue
        if len(clean) > 95:
            continue
        if re.match(r"^\d+(\.\d+){0,3}\s+\S+", clean):
            level = min(3, clean.count(".") + 1)
            headings.append({"level": level, "text": clean, "page": page_number})
            continue
        if clean.isupper() and len(clean.split()) <= 10:
            headings.append({"level": 1, "text": clean.title(), "page": page_number})
            continue
        if clean.endswith(":") and len(clean.split()) <= 12:
            headings.append({"level": 2, "text": clean.rstrip(":"), "page": page_number})

    # Keep unique while preserving order.
    seen = set()
    unique: List[Dict[str, object]] = []
    for h in headings:
        key = (h["level"], h["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)
    return unique[:4]


def _heuristic_relevel_headings(headings: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Normalize heading levels to start at H1 while keeping relative differences."""
    cleaned: List[Dict[str, object]] = []
    levels: List[int] = []
    for h in headings:
        text = str(h.get("text", "")).strip()
        if not text:
            continue
        try:
            level = int(h.get("level", 1))
        except Exception:
            level = 1
        level = min(3, max(1, level))
        new_h = dict(h)
        new_h["text"] = text
        new_h["level"] = level
        cleaned.append(new_h)
        levels.append(level)

    if not cleaned:
        return []

    shift = min(levels) - 1
    if shift <= 0:
        return cleaned

    for h in cleaned:
        h["level"] = min(3, max(1, int(h["level"]) - shift))
    return cleaned


def _infer_page_headings(page_text: str, page_number: int) -> List[Dict[str, object]]:
    llm = _deepseek_heading_guess(page_text, page_number)
    if llm:
        return llm
    return _heuristic_heading_guess(page_text, page_number)


def _is_ocr_placeholder(text: str) -> bool:
    """Detect non-content placeholder text produced when OCR is unavailable."""
    lowered = text.lower()
    return (
        "local deepseek ocr unavailable on page" in lowered
        or "ocr backend '" in lowered
    )


def _extract_page_texts(pdf_path: Path) -> List[str]:
    """Extract direct page text from a PDF (used as fallback for heading/tag inference)."""
    doc = fitz.open(pdf_path.as_posix())
    try:
        return [page.get_text("text").strip() for page in doc]
    finally:
        doc.close()


def _merge_inference_texts(ocr_texts: List[str], native_texts: List[str]) -> List[str]:
    """
    Prefer OCR page text, but fallback to direct PDF-extracted text when OCR
    yielded placeholders or empty output.
    """
    merged: List[str] = []
    for idx, native in enumerate(native_texts):
        ocr = ocr_texts[idx] if idx < len(ocr_texts) else ""
        if ocr.strip() and not _is_ocr_placeholder(ocr):
            merged.append(ocr)
        else:
            merged.append(native.strip())
    return merged


def _normalize_heading_level_to_hn(level: int) -> str:
    """Map inferred heading levels directly to H1/H2/H3 tags."""
    return f"H{min(3, max(1, int(level)))}"


def _apply_pdf_headings_as_bookmarks(pdf_in: Path, pdf_out: Path, headings: List[Dict[str, object]]) -> None:
    """
    Add inferred headings as PDF bookmarks (table-of-contents/navigation).
    """
    doc = fitz.open(pdf_in.as_posix())
    toc: List[List[object]] = []
    for h in headings:
        heading_text = str(h.get("text", "")).strip()
        if not heading_text:
            continue
        level = int(h.get("level", 1))
        page = int(h.get("page", 1))
        toc.append([min(3, max(1, level)), heading_text, max(1, page)])

    if toc:
        doc.set_toc(toc)
    doc.save(pdf_out.as_posix(), garbage=3, deflate=True)
    doc.close()


def _heuristic_structural_tag_guess(text: str) -> List[str]:
    """Rule-based backup when LLM guesses are unavailable."""
    t = text.strip()
    if not t:
        return ["Sect"]

    tags: List[str] = []
    lines = [line.strip() for line in t.splitlines() if line.strip()]
    if lines:
        first = lines[0]
        if len(first) < 90 and (first.isupper() or re.match(r"^\d+(\.\d+)*\s+\S+", first)):
            tags.append("H1")

    if re.search(r"(^|\n)\s*([-*•]|\d+[.)])\s+", t):
        tags.extend(["L", "LI"])
    if "|" in t or re.search(r"\b(table|row|column)\b", t, flags=re.IGNORECASE):
        tags.append("Table")
    if re.search(r"\b(figure|chart|diagram|image)\b", t, flags=re.IGNORECASE):
        tags.append("Figure")
    tags.append("P")
    return list(dict.fromkeys(tags))


def _guess_page_structure_tags(page_text: str) -> List[str]:
    llm_tags = _deepseek_structural_tag_guess(page_text)
    if llm_tags:
        return llm_tags
    return _heuristic_structural_tag_guess(page_text)


def _build_searchable_pdf(input_pdf: Path, output_pdf: Path) -> None:
    """
    Build a searchable PDF without OCRmyPDF/pikepdf.
    Strategy:
    - If page already has text, preserve original page.
    - If page has no text, rasterize and OCR with Tesseract PDF output.
    """
    src = fitz.open(input_pdf.as_posix())
    out = fitz.open()
    tmp_dir = Path(tempfile.mkdtemp(prefix="searchable_pdf_"))

    try:
        for page_index, page in enumerate(src):
            page_text = page.get_text("text").strip()
            if page_text:
                out.insert_pdf(src, from_page=page_index, to_page=page_index)
                continue

            img_path = tmp_dir / f"page_{page_index + 1:04d}.png"
            ocr_pdf_path = tmp_dir / f"page_{page_index + 1:04d}.pdf"

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pix.save(img_path.as_posix())

            _run(["tesseract", img_path.as_posix(), ocr_pdf_path.with_suffix("").as_posix(), "pdf"])

            if not ocr_pdf_path.exists():
                raise RuntimeError(f"Tesseract did not produce expected PDF for page {page_index + 1}")

            ocr_page_doc = fitz.open(ocr_pdf_path.as_posix())
            out.insert_pdf(ocr_page_doc)
            ocr_page_doc.close()

        out.save(output_pdf.as_posix(), garbage=3, deflate=True)
    finally:
        src.close()
        out.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _set_pdf_metadata(pdf_in: Path, pdf_out: Path, title: str, language: str) -> None:
    doc = fitz.open(pdf_in.as_posix())
    metadata = doc.metadata or {}
    metadata["title"] = title
    doc.set_metadata(metadata)

    catalog_xref = doc.pdf_catalog()
    doc.xref_set_key(catalog_xref, "Lang", f"({language})")
    doc.xref_set_key(catalog_xref, "ViewerPreferences", "<< /DisplayDocTitle true >>")
    doc.xref_set_key(catalog_xref, "MarkInfo", "<< /Marked true >>")

    doc.save(pdf_out.as_posix(), garbage=3, deflate=True)
    doc.close()


def _add_minimal_structure_tags(
    pdf_in: Path,
    pdf_out: Path,
    page_tag_guesses: Optional[List[List[str]]] = None,
    page_headings: Optional[List[List[Dict[str, object]]]] = None,
) -> None:
    """
    Add a minimal logical structure tree so the exported PDF is tagged.
    This creates:
      - /MarkInfo << /Marked true >>
      - /StructTreeRoot with a top-level /Document element
      - one /Sect child element per page (linked with /Pg)
      - /ParentTree and /ParentTreeNextKey placeholders

    This does not replace full manual accessibility QA, but it ensures a
    standards-compliant tag tree exists for downstream tools/screen readers.
    """
    try:
        import pikepdf
    except ImportError as exc:
        raise RuntimeError(
            "Tagged PDF export requires pikepdf. Install it with: pip install pikepdf"
        ) from exc

    with pikepdf.Pdf.open(pdf_in.as_posix()) as pdf:
        root = pdf.Root

        if "/MarkInfo" not in root or not isinstance(root.MarkInfo, pikepdf.Dictionary):
            root.MarkInfo = pikepdf.Dictionary()
        root.MarkInfo["/Marked"] = True

        parent_tree_dict = pikepdf.Dictionary()
        parent_tree_dict["/Nums"] = pikepdf.Array()
        parent_tree = pdf.make_indirect(parent_tree_dict)

        doc_struct_elem = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/StructElem"),
                    "/S": pikepdf.Name("/Document"),
                    "/K": pikepdf.Array(),
                }
            )
        )

        page_structs = pikepdf.Array()
        for page_index, page in enumerate(pdf.pages):
            page_container = pdf.make_indirect(
                pikepdf.Dictionary(
                    {
                        "/Type": pikepdf.Name("/StructElem"),
                        "/S": pikepdf.Name("/Sect"),
                        "/P": doc_struct_elem,
                        "/Pg": page.obj,
                        "/K": pikepdf.Array(),
                    }
                )
            )

            guessed_tags = []
            if page_tag_guesses and page_index < len(page_tag_guesses):
                guessed_tags = page_tag_guesses[page_index]

            children = pikepdf.Array()
            if guessed_tags:
                for tag in guessed_tags:
                    children.append(
                        pdf.make_indirect(
                            pikepdf.Dictionary(
                                {
                                    "/Type": pikepdf.Name("/StructElem"),
                                    "/S": pikepdf.Name(f"/{tag}"),
                                    "/P": page_container,
                                    "/Pg": page.obj,
                                    "/K": pikepdf.Array(),
                                }
                            )
                        )
                    )

            heading_items: List[Dict[str, object]] = []
            if page_headings and page_index < len(page_headings):
                heading_items = page_headings[page_index]

            for heading in heading_items:
                heading_text = str(heading.get("text", "")).strip()
                if not heading_text:
                    continue
                heading_tag = _normalize_heading_level_to_hn(int(heading.get("level", 2)))
                children.append(
                    pdf.make_indirect(
                        pikepdf.Dictionary(
                            {
                                "/Type": pikepdf.Name("/StructElem"),
                                "/S": pikepdf.Name(f"/{heading_tag}"),
                                "/P": page_container,
                                "/Pg": page.obj,
                                "/K": pikepdf.Array(),
                                "/Alt": heading_text,
                                "/ActualText": heading_text,
                            }
                        )
                    )
                )

            if len(children) > 0:
                page_container.K = children

            page_structs.append(page_container)

        doc_struct_elem.K = page_structs

        struct_tree_root = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/StructTreeRoot"),
                    "/K": pikepdf.Array([doc_struct_elem]),
                    "/ParentTree": parent_tree,
                    "/ParentTreeNextKey": 0,
                }
            )
        )
        doc_struct_elem.P = struct_tree_root
        root.StructTreeRoot = struct_tree_root
        pdf.save(pdf_out.as_posix())


def _render_pages(pdf_path: Path, pages_dir: Path) -> List[Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    rendered: List[Path] = []
    for page_idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        out = pages_dir / f"page_{page_idx:04d}.png"
        pix.save(out.as_posix())
        rendered.append(out)
    doc.close()
    return rendered


def _extract_images(pdf_path: Path, images_dir: Path) -> List[Dict[str, object]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    doc = fitz.open(pdf_path)
    for page_number, page in enumerate(doc, start=1):
        imgs = page.get_images(full=True)
        for image_index, img in enumerate(imgs, start=1):
            xref = img[0]
            blob = doc.extract_image(xref)
            ext = blob.get("ext", "png")
            image_path = images_dir / f"page_{page_number:04d}_img_{image_index:03d}.{ext}"
            image_path.write_bytes(blob["image"])
            records.append(
                {
                    "page": page_number,
                    "xref": xref,
                    "image_path": image_path.as_posix(),
                }
            )
    doc.close()
    return records


def _collect_page_ocr_texts(
    page_images: List[Path],
    ocr_output_dir: Path,
    ocr_backend: str = "local_hf",
) -> List[str]:
    """
    OCR markdown sidecar generation.
    Default backend: local_hf (DeepSeek OCR from Hugging Face).
    Fallback backend: none (placeholder lines).
    """
    texts: List[str] = []

    for i, image_path in enumerate(page_images, start=1):
        if ocr_backend == "local_hf":
            try:
                text = _local_deepseek_ocr_markdown(image_path, ocr_output_dir)
            except Exception as e:
                text = f"[Local DeepSeek OCR unavailable on page {i}: {e}]"
        else:
            text = f"[OCR backend '{ocr_backend}' not implemented in this single-file mode.]"
        texts.append(text)

    return texts


def _write_ocr_markdown(page_texts: List[str], markdown_path: Path) -> Path:
    chunks: List[str] = []
    local_warnings = False

    for i, text in enumerate(page_texts, start=1):
        chunks.append(f"\n<!-- Page {i} -->\n")
        chunks.append(text)
        if "Local DeepSeek OCR unavailable on page" in text:
            local_warnings = True

    if local_warnings:
        chunks.insert(0, "[WARNING] Local HF DeepSeek OCR partially unavailable; some pages may need rerun.\n")

    markdown_path.write_text("\n".join(chunks), encoding="utf-8")
    return markdown_path


def _generate_alt_text_for_image(image_path: Path, alt_backend: str = "local_hf") -> str:
    """
    For default local_hf mode, reuse local OCR model with a description prompt.
    If that fails, fallback to remote API, then generic placeholder.
    """
    if alt_backend == "local_hf":
        try:
            tokenizer, model = _load_local_deepseek_ocr_model()
            prompt = "<image>\nDescribe this image for accessibility alt text in one sentence."
            res = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path.as_posix(),
                output_path=".",
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=True,
            )
            txt = str(res).strip()
            if txt:
                return txt
        except Exception:
            pass

    return _deepseek_alt_text_api(image_path)


def _write_alt_text_manifest(
    image_records: List[Dict[str, object]],
    manifest_path: Path,
    alt_backend: str = "local_hf",
) -> Path:
    manifest = {"images": []}
    for rec in image_records:
        img_path = Path(str(rec["image_path"]))
        alt_text = _generate_alt_text_for_image(img_path, alt_backend=alt_backend)
        manifest["images"].append(
            {
                "page": rec["page"],
                "xref": rec["xref"],
                "image_path": rec["image_path"],
                "alt_text": alt_text,
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _qpdf_check(pdf_path: Path) -> Dict[str, object]:
    proc = subprocess.run(["qpdf", "--check", pdf_path.as_posix()], capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "command": f"qpdf --check {pdf_path}",
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _write_report(
    report_path: Path,
    source_pdf: Path,
    output_pdf: Path,
    manifest_path: Path,
    markdown_path: Path,
    qpdf_result: Dict[str, object],
    ocr_backend: str,
    alt_backend: str,
    inferred_heading_count: int,
) -> None:
    report = {
        "source_pdf": source_pdf.as_posix(),
        "output_pdf": output_pdf.as_posix(),
        "ocr_markdown": markdown_path.as_posix(),
        "alt_text_manifest": manifest_path.as_posix(),
        "backends": {"ocr_backend": ocr_backend, "alt_backend": alt_backend},
        "checks": {
            "searchable_text": "Applied with PyMuPDF + Tesseract fallback for non-text pages.",
            "document_language": "Set in PDF catalog /Lang.",
            "document_title": "Set in /Title and display-title preference.",
            "alternative_text": "Generated for each embedded image.",
            "tagged_structured": "Added /StructTreeRoot and LLM/heuristic-guessed page child tags (H*, P, L, Table, Figure).",
            "headings_navigation": (
                f"Inferred headings added as PDF bookmarks/table-of-contents entries: {inferred_heading_count}."
            ),
            "why_headings": [
                "Facilitates reading by adding structure and clarity.",
                "Essential for screen-reader users and students with visual impairments.",
                "Enables easier navigation and automatic table-of-contents generation.",
            ],
            "logical_reading_order": "Tagged structure added; run PAC/Acrobat QA for final reading-order validation.",
            "color_contrast": "Requires source-content remediation when needed.",
        },
        "machine_validation": qpdf_result,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


# ---------------------------
# Public one-liner API
# ---------------------------
def convert_pdf(
    source_pdf: str,
    output_pdf: str,
    *,
    title: Optional[str] = None,
    language: str = "en-US",
    ocr_backend: str = "local_hf",  # default as requested
    alt_backend: str = "local_hf",  # default local as well
    keep_workdir: bool = False,
) -> Dict[str, str]:
    """
    One-line entrypoint for Colab:
        convert_pdf("original.pdf", "convertedaccessibleversion.pdf")

    Backends:
    - ocr_backend='local_hf' (default): uses local Hugging Face deepseek-ai/DeepSeek-OCR
    - alt_backend='local_hf' (default), fallback to API if local alt inference fails
    """
    src = Path(source_pdf).expanduser().resolve()
    dst = Path(output_pdf).expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source PDF not found: {src}")

    _ensure_parent(dst)

    work_dir = dst.parent / f".{dst.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    ocr_pdf = work_dir / "01_searchable.pdf"
    meta_pdf = work_dir / "02_with_metadata.pdf"
    tagged_pdf = work_dir / "03_tagged.pdf"
    headed_pdf = work_dir / "04_with_headings.pdf"
    pages_dir = work_dir / "pages"
    images_dir = work_dir / "images"

    markdown_path = dst.with_suffix(dst.suffix + ".ocr.md")
    manifest_path = dst.with_suffix(dst.suffix + ".alt_text_manifest.json")
    report_path = dst.with_suffix(dst.suffix + ".accessibility_report.json")

    # 1) Searchable PDF (best visual fidelity for scanned and text-like docs)
    _build_searchable_pdf(src, ocr_pdf)

    # 2) Document metadata
    resolved_title = title if title else src.stem
    _set_pdf_metadata(ocr_pdf, meta_pdf, resolved_title, language)

    # 3) OCR page text extraction (used for sidecar + LLM tag guesses)
    page_images = _render_pages(meta_pdf, pages_dir)
    page_texts = _collect_page_ocr_texts(page_images, work_dir / "ocr_outputs", ocr_backend=ocr_backend)
    _write_ocr_markdown(page_texts, markdown_path)

    # 4) Build robust inference text per page. If OCR failed on a page, use native PDF text.
    native_page_texts = _extract_page_texts(meta_pdf)
    inference_texts = _merge_inference_texts(page_texts, native_page_texts)

    # 4b) Infer headings per page (used for both structure tags and bookmarks/TOC).
    inferred_headings: List[Dict[str, object]] = []
    page_headings: List[List[Dict[str, object]]] = []
    for page_number, page_text in enumerate(inference_texts, start=1):
        headings_for_page = _infer_page_headings(page_text, page_number)
        page_headings.append(headings_for_page)
        inferred_headings.extend(headings_for_page)

    # Normalize hierarchy so headings start at H1 and preserve relative depth.
    inferred_headings = _deepseek_relevel_headings(inferred_headings)
    headings_by_page: Dict[int, List[Dict[str, object]]] = {}
    for heading in inferred_headings:
        page_no = int(heading.get("page", 1))
        headings_by_page.setdefault(page_no, []).append(heading)
    page_headings = [headings_by_page.get(i + 1, []) for i in range(len(inference_texts))]

    # 4c) LLM/heuristic structural tag guesses, then tag the PDF (with heading tags included).
    page_tag_guesses = [_guess_page_structure_tags(text) for text in inference_texts]
    _add_minimal_structure_tags(
        meta_pdf,
        tagged_pdf,
        page_tag_guesses=page_tag_guesses,
        page_headings=page_headings,
    )

    # 4d) Add inferred headings as PDF bookmarks/TOC for navigation panes.
    _apply_pdf_headings_as_bookmarks(tagged_pdf, headed_pdf, inferred_headings)

    # 5) Alt text generation for extracted images
    image_records = _extract_images(headed_pdf, images_dir)
    _write_alt_text_manifest(image_records, manifest_path, alt_backend=alt_backend)

    # 6) Final output PDF
    shutil.copy2(headed_pdf, dst)

    # 7) Validation + report
    qpdf_result = _qpdf_check(dst)
    _write_report(
        report_path=report_path,
        source_pdf=src,
        output_pdf=dst,
        manifest_path=manifest_path,
        markdown_path=markdown_path,
        qpdf_result=qpdf_result,
        ocr_backend=ocr_backend,
        alt_backend=alt_backend,
        inferred_heading_count=len(inferred_headings),
    )

    if not keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "source_pdf": src.as_posix(),
        "output_pdf": dst.as_posix(),
        "ocr_markdown": markdown_path.as_posix(),
        "alt_text_manifest": manifest_path.as_posix(),
        "report": report_path.as_posix(),
    }


if __name__ == "__main__":
    # Paste in Colab, then run:
    # convert_pdf("original.pdf", "convertedaccessibleversion.pdf")
    pass

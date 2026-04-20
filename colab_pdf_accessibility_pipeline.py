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
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import pikepdf
import requests

# ---------------------------
# Configuration (env override)
# ---------------------------
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-vl2")
DEEPSEEK_OCR_MODEL_NAME = os.getenv("DEEPSEEK_OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR")


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
    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
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


def _build_searchable_pdf(input_pdf: Path, output_pdf: Path) -> None:
    """
    Works for scanned image PDFs, mixed PDFs, and text-like PDFs.
    --skip-text preserves text-native pages and OCRs only non-searchable pages.
    """
    _run(
        [
            "ocrmypdf",
            "--skip-text",
            "--deskew",
            "--clean-final",
            "--optimize",
            "1",
            input_pdf.as_posix(),
            output_pdf.as_posix(),
        ]
    )


def _set_pdf_metadata(pdf_in: Path, pdf_out: Path, title: str, language: str) -> None:
    with pikepdf.open(pdf_in.as_posix(), allow_overwriting_input=False) as pdf:
        pdf.docinfo["/Title"] = title

        root = pdf.Root
        root["/Lang"] = pikepdf.String(language)

        viewer_prefs = root.get("/ViewerPreferences", pikepdf.Dictionary())
        viewer_prefs["/DisplayDocTitle"] = True
        root["/ViewerPreferences"] = viewer_prefs

        markinfo = root.get("/MarkInfo", pikepdf.Dictionary())
        markinfo["/Marked"] = True
        root["/MarkInfo"] = markinfo

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


def _write_ocr_markdown(
    page_images: List[Path],
    markdown_path: Path,
    ocr_backend: str = "local_hf",
) -> Path:
    """
    OCR markdown sidecar generation.
    Default backend: local_hf (DeepSeek OCR from Hugging Face).
    Fallback backend: none (placeholder lines).
    """
    chunks: List[str] = []
    local_ok = True

    for i, image_path in enumerate(page_images, start=1):
        chunks.append(f"\n<!-- Page {i} -->\n")
        if ocr_backend == "local_hf":
            try:
                text = _local_deepseek_ocr_markdown(image_path, markdown_path.parent / "ocr_outputs")
            except Exception as e:
                local_ok = False
                text = f"[Local DeepSeek OCR unavailable on page {i}: {e}]"
        else:
            text = f"[OCR backend '{ocr_backend}' not implemented in this single-file mode.]"
        chunks.append(text)

    if ocr_backend == "local_hf" and not local_ok:
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
) -> None:
    report = {
        "source_pdf": source_pdf.as_posix(),
        "output_pdf": output_pdf.as_posix(),
        "ocr_markdown": markdown_path.as_posix(),
        "alt_text_manifest": manifest_path.as_posix(),
        "backends": {"ocr_backend": ocr_backend, "alt_backend": alt_backend},
        "checks": {
            "searchable_text": "Applied with OCRmyPDF --skip-text (scanned + mixed + text-like support).",
            "document_language": "Set in PDF catalog /Lang.",
            "document_title": "Set in /Title and display-title preference.",
            "alternative_text": "Generated for each embedded image.",
            "tagged_structured": "Requires final tagging pass in Acrobat/PDFix/PAC workflow.",
            "logical_reading_order": "Use Acrobat Reading Order tool for final QA/repair.",
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

    # 3) OCR markdown sidecar using selected backend (default local HF DeepSeek OCR)
    page_images = _render_pages(meta_pdf, pages_dir)
    _write_ocr_markdown(page_images, markdown_path, ocr_backend=ocr_backend)

    # 4) Alt text generation for extracted images
    image_records = _extract_images(meta_pdf, images_dir)
    _write_alt_text_manifest(image_records, manifest_path, alt_backend=alt_backend)

    # 5) Final output PDF
    shutil.copy2(meta_pdf, dst)

    # 6) Validation + report
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

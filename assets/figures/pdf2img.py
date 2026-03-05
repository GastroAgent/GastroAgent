#!/usr/bin/env python3
"""
Convert PDF pages to images (PNG or JPG).

Usage examples:
    python assets/figures/pdf2img.py input.pdf
    python assets/figures/pdf2img.py assets/figures/overview-ill.pdf -o assets/figures/overview-ill -f jpg --dpi 200 --quality 90
    python assets/figures/pdf2img.py input.pdf --start-page 2 --end-page 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: PyMuPDF.\n"
        "Install it with: pip install pymupdf"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDF file into PNG/JPG images."
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("pdf_images"),
        help="Output directory (default: ./pdf_images).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("png", "jpg", "jpeg"),
        default="png",
        help="Image format: png, jpg, jpeg (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI (default: 300).",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="1-based start page (default: 1).",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="1-based end page, inclusive (default: last page).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPG quality 1-100 (only for jpg/jpeg, default: 95).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    if args.pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Input file must be a PDF: {args.pdf}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be a positive integer.")
    if args.start_page <= 0:
        raise ValueError("--start-page must be >= 1.")
    if args.end_page is not None and args.end_page <= 0:
        raise ValueError("--end-page must be >= 1.")
    if args.end_page is not None and args.start_page > args.end_page:
        raise ValueError("--start-page cannot be greater than --end-page.")
    if not (1 <= args.quality <= 100):
        raise ValueError("--quality must be in [1, 100].")


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    image_format: str = "png",
    dpi: int = 300,
    start_page: int = 1,
    end_page: int | None = None,
    quality: int = 95,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_format = "jpg" if image_format == "jpeg" else image_format

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError("The input PDF has no pages.")

        real_end_page = end_page if end_page is not None else total_pages
        if start_page > total_pages:
            raise ValueError(
                f"--start-page ({start_page}) exceeds total pages ({total_pages})."
            )
        if real_end_page > total_pages:
            raise ValueError(
                f"--end-page ({real_end_page}) exceeds total pages ({total_pages})."
            )

        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        stem = pdf_path.stem
        converted_count = 0

        for page_no in range(start_page - 1, real_end_page):
            page = doc.load_page(page_no)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_name = f"{stem}_page_{page_no + 1:04d}.{image_format}"
            out_path = output_dir / out_name

            if image_format == "png":
                pix.save(out_path)
            else:
                pix.save(out_path, jpg_quality=quality)
            converted_count += 1

    return converted_count


def main() -> None:
    args = parse_args()
    validate_args(args)

    count = convert_pdf_to_images(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        image_format=args.format,
        dpi=args.dpi,
        start_page=args.start_page,
        end_page=args.end_page,
        quality=args.quality,
    )

    print(
        f"Done. Converted {count} page(s) from '{args.pdf}' "
        f"to '{args.output_dir}' as {args.format}."
    )


if __name__ == "__main__":
    main()

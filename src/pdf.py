import json
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
import html

from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def normalize_local_pdf_mode(value) -> str | None:
    valid_modes = (
        "file",
        "dir",
        "dir_recursive",
        "package",
        "package_recursive",
    )
    if value in (None, False, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(
            '"read_local_pdf" must be one of: "file", "dir", "dir_recursive", '
            '"package", or "package_recursive".'
        )

    normalized_value = " ".join(value.strip().split())
    if not normalized_value:
        return None
    if normalized_value in valid_modes:
        return normalized_value

    raise ValueError(
        '"read_local_pdf" must be one of: "file", "dir", "dir_recursive", '
        '"package", or "package_recursive".'
    )


def _sanitize_dataset_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return token or "pdf"

def build_local_pdf_dataset_name(data_dir: str | Path, read_mode: str) -> str:
    read_mode = normalize_local_pdf_mode(read_mode)
    if read_mode is None:
        raise ValueError('"read_local_pdf" cannot be None when building a local PDF dataset name.')

    input_path = Path(str(data_dir)).expanduser()
    return _sanitize_dataset_token(input_path.name or "pdf")


def _validate_archive_path(output_root: Path, member_name: str) -> None:
    output_root = output_root.resolve()
    member_path = (output_root / member_name).resolve()
    try:
        member_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f'Unsafe package member "{member_name}".')


def _extract_package(package_path: Path, output_root: Path) -> None:
    if zipfile.is_zipfile(package_path):
        with zipfile.ZipFile(package_path) as archive:
            for member_name in archive.namelist():
                _validate_archive_path(output_root, member_name)
            archive.extractall(output_root)
        return

    if tarfile.is_tarfile(package_path):
        with tarfile.open(package_path) as archive:
            for member in archive.getmembers():
                _validate_archive_path(output_root, member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f'Unsupported symlink entry in package: "{member.name}".')
            archive.extractall(output_root)
        return

    if package_path.suffix.lower() == ".7z":
        try:
            import py7zr
        except ImportError as exc:
            raise ImportError('Reading ".7z" packages requires "py7zr".') from exc

        with py7zr.SevenZipFile(package_path) as archive:
            for member_name in archive.getnames():
                _validate_archive_path(output_root, member_name)
            archive.extractall(output_root)
        return

    if package_path.suffix.lower() == ".rar":
        try:
            import rarfile
        except ImportError as exc:
            raise ImportError('Reading ".rar" packages requires "rarfile".') from exc

        with rarfile.RarFile(package_path) as archive:
            for member_name in archive.namelist():
                _validate_archive_path(output_root, member_name)
            archive.extractall(output_root)
        return

    raise ValueError(
        f'Unsupported package format "{package_path.suffix}". '
        'Supported packages include zip, tar, 7z, and rar families.'
    )


def _collect_pdf_paths(root: Path, recursive: bool) -> List[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() == ".pdf"
        and ".psirag_pdf_cache" not in path.parts
    )


def _temporary_directory(prefix: str, dir_path: Path) -> TemporaryDirectory:
    try:
        return TemporaryDirectory(prefix=prefix, dir=dir_path)
    except (FileNotFoundError, PermissionError):
        return TemporaryDirectory(prefix=prefix)


def _get_mineru_command() -> List[str]:
    if shutil.which("mineru") is not None:
        return ["mineru"]

    try:
        __import__("mineru")
    except ImportError as exc:
        raise ImportError(
            'MinerU is required for "read_local_pdf". Install the official "mineru" package.'
        ) from exc

    return [sys.executable, "-m", "mineru.cli.client"]


def _get_cache_root(input_path: Path, read_mode: str) -> Path:
    if read_mode in ("dir", "dir_recursive"):
        return input_path / ".psirag_pdf_cache"
    if read_mode in ("package", "package_recursive"):
        return input_path.parent / ".psirag_pdf_cache" / _sanitize_dataset_token(input_path.stem)
    return input_path.parent / ".psirag_pdf_cache"


def _get_cache_path(pdf_path: Path, cache_root: Path, root_path: Path | None) -> Path:
    relative_path = Path(pdf_path.name) if root_path is None else pdf_path.relative_to(root_path)
    return (cache_root / relative_path).with_suffix(".json")


def _load_cached_document(cache_path: Path, title: str) -> Dict | None:
    if not cache_path.exists():
        return None

    try:
        cached_document = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(cached_document, dict):
        return None
    if cached_document.get("title") != title:
        return None
    if not isinstance(cached_document.get("chunks"), list):
        return None
    return cached_document


def _save_cached_document(cache_path: Path, document: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_mineru(input_path: Path, output_root: Path) -> None:
    command = _get_mineru_command() + [
        "-p", str(input_path),
        "-o", str(output_root),
        "-b", "pipeline",
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f'MinerU failed to parse "{input_path}".\n'
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )


def _find_output_file(output_root: Path, pdf_stem: str, suffix: str) -> Path | None:
    output_paths = sorted(output_root.rglob(f"*{suffix}"))
    if not output_paths:
        return None

    for path in output_paths:
        if path.stem == pdf_stem:
            return path
        if path.name.startswith(f"{pdf_stem}_"):
            return path
    return output_paths[0]


def _normalize_text(text) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(_flatten_text_list(text))
    elif not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _flatten_text_list(value) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts = []
        for item in value:
            texts.extend(_flatten_text_list(item))
        return texts
    return []


def _collect_item_text(content_item: dict) -> str:
    item_type = content_item.get("type")

    if item_type in ("header", "footer", "page_number", "aside_text", "page_footnote", "seal"):
        return ""

    if item_type == "text":
        text = _normalize_text(content_item.get("text", ""))
        text_level = content_item.get("text_level", 0)
        if text and isinstance(text_level, int) and text_level > 0:
            return f'{"#" * text_level} {text}'
        return text

    if item_type in ("list", "equation"):
        parts = [_normalize_text(content_item.get("text", ""))]
        if item_type == "list":
            parts.extend(
                _normalize_text(item)
                for item in _flatten_text_list(content_item.get("list_items"))
            )
        if item_type == "equation":
            parts.extend(
                _normalize_text(content_item.get(field, ""))
                for field in ("latex", "html")
            )
        return "\n".join(part for part in parts if part)

    if item_type == "code":
        parts = [
            _normalize_text(content_item.get("code_caption", "")),
            _normalize_text(content_item.get("code_body", "")),
            _normalize_text(content_item.get("code_footnote", "")),
        ]
        return "\n".join(part for part in parts if part)

    if item_type == "table":
        parts = []
        for field in ("table_caption", "table_body", "table_footnote", "html"):
            value = content_item.get(field, "")
            if isinstance(value, list):
                value = " ".join(value)
            value = _normalize_text(value)
            if value:
                parts.append(value)
        return "\n".join(parts)

    if item_type in ("image", "chart"):
        parts = []
        for field in ("image_caption", "image_footnote", "chart_caption", "chart_footnote"):
            value = content_item.get(field, "")
            if isinstance(value, list):
                value = " ".join(value)
            value = _normalize_text(value)
            if value:
                parts.append(value)
        return "\n".join(parts)

    parts = [
        _normalize_text(content_item.get("text", "")),
        _normalize_text(content_item.get("content", "")),
        _normalize_text(content_item.get("latex", "")),
        _normalize_text(content_item.get("html", "")),
    ]
    return "\n".join(part for part in parts if part)


def _load_mineru_content(output_root: Path, pdf_stem: str) -> List[str]:
    content_list_path = _find_output_file(output_root, pdf_stem, "_content_list.json")
    if content_list_path is not None:
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
        chunks = []
        for item in content_list:
            chunk = _collect_item_text(item)
            if chunk:
                chunks.append(chunk)
        if chunks:
            return chunks

    markdown_path = _find_output_file(output_root, pdf_stem, ".md")
    if markdown_path is None:
        raise ValueError(f'MinerU did not produce readable output for "{pdf_stem}".')

    markdown_text = markdown_path.read_text(encoding="utf-8").strip()
    return [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", markdown_text)
        if chunk.strip()
    ]


def _parse_pdf_with_mineru(pdf_path: Path) -> List[str]:
    with _temporary_directory(prefix=".psirag_mineru_", dir_path=pdf_path.parent) as output_dir:
        output_root = Path(output_dir)
        mineru_pdf_path = output_root / "document.pdf"
        shutil.copy2(pdf_path, mineru_pdf_path)
        _run_mineru(mineru_pdf_path, output_root)
        return _load_mineru_content(output_root, mineru_pdf_path.stem)


def _parse_pdf_batch_with_mineru(pdf_paths: List[Path]) -> Dict[Path, List[str]]:
    if len(pdf_paths) == 1:
        return {pdf_paths[0]: _parse_pdf_with_mineru(pdf_paths[0])}

    with _temporary_directory(prefix=".psirag_mineru_input_", dir_path=pdf_paths[0].parent) as input_dir, \
         _temporary_directory(prefix=".psirag_mineru_output_", dir_path=pdf_paths[0].parent) as output_dir:
        input_root = Path(input_dir)
        output_root = Path(output_dir)
        pdf_name_map = {}
        for i, pdf_path in enumerate(pdf_paths):
            mineru_pdf_path = input_root / f"doc_{i}.pdf"
            shutil.copy2(pdf_path, mineru_pdf_path)
            pdf_name_map[pdf_path] = mineru_pdf_path.stem

        _run_mineru(input_root, output_root)
        return {
            pdf_path: _load_mineru_content(output_root, pdf_stem)
            for pdf_path, pdf_stem in pdf_name_map.items()
        }


def _get_pdf_title(pdf_path: Path, root_path: Path | None) -> str:
    if root_path is None:
        return pdf_path.stem

    relative_path = pdf_path.relative_to(root_path)
    if relative_path.suffix.lower() == ".pdf":
        relative_path = relative_path.with_suffix("")
    return relative_path.as_posix()


def _parse_pdf_paths(
    pdf_paths: List[Path],
    root_path: Path | None,
    cache_root: Path,
    batch_size: int,
) -> List[dict]:
    documents = {}
    uncached_pdfs = []

    for pdf_path in pdf_paths:
        title = _get_pdf_title(pdf_path, root_path)
        cache_path = _get_cache_path(pdf_path, cache_root, root_path)
        cached_document = _load_cached_document(cache_path, title)
        if cached_document is not None:
            documents[pdf_path] = cached_document
        else:
            uncached_pdfs.append((pdf_path, title, cache_path))

    if documents:
        tqdm.write(f"Loaded {len(documents)} cached pdf(s) from \"{cache_root}\".")

    if uncached_pdfs:
        batch_total = (len(uncached_pdfs) - 1) // batch_size + 1
        bar = tqdm(total=batch_total, desc="parsing local pdf")
        for i in range(0, len(uncached_pdfs), batch_size):
            batch = uncached_pdfs[i : i + batch_size]
            parsed_batch = _parse_pdf_batch_with_mineru([pdf_path for pdf_path, _, _ in batch])
            for pdf_path, title, cache_path in batch:
                document = {
                    "title": title,
                    "chunks": parsed_batch[pdf_path],
                }
                _save_cached_document(cache_path, document)
                documents[pdf_path] = document
            bar.update(1)
        bar.close()

    return [documents[pdf_path] for pdf_path in pdf_paths]


def load_local_pdf_data(data_dir: str | Path, read_mode: str, pdf_batch_size: int = 1) -> List[dict]:

    if pdf_batch_size < 1:
        raise ValueError('"local_pdf_batch_size" must be a positive integer.')
    
    read_mode = normalize_local_pdf_mode(read_mode)
    if read_mode is None:
        raise ValueError('"read_local_pdf" cannot be None when loading local PDFs.')

    input_path = Path(str(data_dir)).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    cache_root = _get_cache_root(input_path, read_mode)
    cache_root.mkdir(parents=True, exist_ok=True)

    if read_mode == "file":
        if not input_path.is_file() or input_path.suffix.lower() != ".pdf":
            raise ValueError(f'"{input_path}" must be a single PDF file.')
        return _parse_pdf_paths([input_path], None, cache_root, pdf_batch_size)

    if read_mode in ("dir", "dir_recursive"):
        if not input_path.is_dir():
            raise ValueError(f'"{input_path}" is not a directory.')
        pdf_paths = _collect_pdf_paths(input_path, recursive=read_mode.endswith("recursive"))
        if not pdf_paths:
            raise FileNotFoundError(f'No PDF files found under "{input_path}".')
        return _parse_pdf_paths(pdf_paths, input_path, cache_root, pdf_batch_size)

    if read_mode in ("package", "package_recursive"):
        if not input_path.is_file():
            raise ValueError(f'"{input_path}" is not a package file.')
        with _temporary_directory(prefix=".psirag_pdf_package_", dir_path=input_path.parent) as output_dir:
            output_root = Path(output_dir)
            _extract_package(input_path, output_root)
            pdf_paths = _collect_pdf_paths(output_root, recursive=read_mode.endswith("recursive"))
            if not pdf_paths:
                raise FileNotFoundError(f'No PDF files found inside package "{input_path}".')
            return _parse_pdf_paths(pdf_paths, output_root, cache_root, pdf_batch_size)

    raise ValueError(f'Unsupported read_local_pdf mode "{read_mode}".')


def prepare_local_pdf_dataset(data_dir: str | Path, read_mode: str) -> str:
    read_mode = normalize_local_pdf_mode(read_mode)
    if read_mode is None:
        raise ValueError('"read_local_pdf" cannot be None when preparing a local PDF dataset.')
    return build_local_pdf_dataset_name(data_dir, read_mode)

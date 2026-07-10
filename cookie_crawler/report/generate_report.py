import argparse
import base64
import os
from pathlib import Path
from typing import Optional

import jinja2

from cookie_crawler.report.data import build_constats, list_websites
from database.queries import init_db
from shared_utils import load_yaml

TEMPLATE_DIR = Path("report_template")
LOCAL_DIR = TEMPLATE_DIR / "local"
TEMPLATE_NAME = "template.jinja2"
DEFAULT_METADATA_PATH = LOCAL_DIR / "report_metadata.yaml"


def _load_env() -> None:
    if os.getenv("DB_HOST"):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        env_path = Path(".env")
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _logo_data_uri(metadata: dict) -> str:
    logo_name = metadata.get("header", {}).get("logo", "logo.png")
    logo_path = LOCAL_DIR / logo_name
    if not logo_path.is_file():
        return ""
    b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def render_report(
    experiment_id: str,
    website: str,
    metadata: dict,
    out_dir: Path,
    env: jinja2.Environment,
    exclude_first_party: bool = True,
) -> Optional[Path]:
    try:
        data = build_constats(experiment_id, website, metadata, exclude_first_party)
    except ValueError as e:
        print(f"Skipping {website}: {e}")
        return None

    data["logo"] = _logo_data_uri(metadata)
    html = env.get_template(TEMPLATE_NAME).render(**data)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{website}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path}")

    try:
        import weasyprint

        pdf_path = out_dir / f"{website}.pdf"
        weasyprint.HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf(str(pdf_path))
        print(f"Wrote {pdf_path}")
    except ImportError:
        print("weasyprint not installed - skipping PDF (HTML written).")
    except Exception as e:  # noqa: BLE001 - PDF backend (pango/cairo) may be missing
        print(f"PDF generation failed for {website} ({e}) - HTML written.")
    return html_path


def generate_reports(
    experiment_id: str,
    out_dir: Path,
    website: Optional[str] = None,
    metadata_path: str = str(DEFAULT_METADATA_PATH),
    exclude_first_party: bool = True,
) -> None:
    metadata = load_yaml(metadata_path) if Path(metadata_path).is_file() else {}

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(["html", "xml", "jinja", "jinja2"]),
    )

    websites = [website] if website else list_websites(experiment_id)
    if not websites:
        print(f"No websites found for experiment '{experiment_id}'.")
        return
    out_dir = Path(out_dir)
    for name in websites:
        render_report(experiment_id, name, metadata, out_dir, env, exclude_first_party)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--website", default=None, help="Domain to report on (default: all in experiment)"
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--out-dir", default="report_pv")
    args = parser.parse_args()

    _load_env()
    exp_cfg = load_yaml("config/experiment_config.yaml")
    init_db(exp_cfg["engine"], create_tables=False)
    generate_reports(
        args.experiment_id,
        Path(args.out_dir),
        args.website,
        args.metadata,
        exp_cfg["report"]["exclude_first_party"],
    )


if __name__ == "__main__":
    main()

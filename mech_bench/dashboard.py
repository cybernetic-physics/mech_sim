"""Static-HTML dashboard for one run.

Consumes a payload dict produced by
:func:`mech_bench.dashboard_payload.build_dashboard_payload` and
writes a self-contained ``dashboard.html`` (no server needed).

Plotly is the only optional dependency. If plotly is absent, the
``write_static_dashboard`` call raises :class:`DashboardUnavailableError`.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by env
    import plotly.graph_objects as go  # type: ignore[import-not-found]
    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]
    HAS_PLOTLY = False


class DashboardUnavailableError(ImportError):
    """Raised when plotly is needed but not installed."""


def write_static_dashboard(payload: dict[str, Any], out_path: Path) -> Path:
    """Write a static-HTML dashboard summarizing *payload* to *out_path*.

    Requires plotly; raises :class:`DashboardUnavailableError` otherwise.
    """
    if not HAS_PLOTLY:
        raise DashboardUnavailableError(
            "plotly is required to render the static dashboard. "
            "Install with: pip install 'mech-bench[dashboard]'"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    figures_html = _render_figures(payload)
    summary_html = _render_summary(payload)
    metrics_html = _render_metrics(payload)
    feedback_html = _render_feedback(payload)

    body = (
        "<h1>mech-bench run dashboard</h1>"
        f"{summary_html}"
        f"{metrics_html}"
        f"{feedback_html}"
        f"{figures_html}"
    )
    doc = (
        "<!DOCTYPE html>\n"
        "<html lang='en'><head><meta charset='utf-8'>"
        f"<title>mech-bench: {html.escape(str(payload.get('run', {}).get('run_id', '')))}</title>"
        f"<style>{_STYLE}</style>"
        "</head><body>"
        f"{body}"
        "</body></html>"
    )
    out_path.write_text(doc)
    return out_path


# --------------------------------------------------------------------- #
# Sections                                                              #
# --------------------------------------------------------------------- #


def _render_summary(payload: dict[str, Any]) -> str:
    run = payload.get("run", {}) or {}
    score = payload.get("score", {}) or {}
    cards = [
        ("Score", f"{score.get('dense', 0.0):.3f}"),
        ("Hard gate",
         "passed" if score.get("hard_gate_passed") else "FAILED"),
        ("Task", str(run.get("task_id", ""))),
        ("Family", str(run.get("task_family", ""))),
        ("Difficulty", str(run.get("difficulty", ""))),
        ("Run", str(run.get("run_id", ""))),
    ]
    chunks = [
        f"<div class='card'><div class='card-label'>{html.escape(label)}</div>"
        f"<div class='card-value'>{html.escape(value)}</div></div>"
        for label, value in cards
    ]
    return "<section class='summary'>" + "".join(chunks) + "</section>"


def _render_metrics(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {}) or {}
    if not metrics:
        return ""
    rows = []
    for k in sorted(metrics):
        v = metrics[k]
        rows.append(
            f"<tr><td>{html.escape(str(k))}</td>"
            f"<td>{html.escape(_fmt_value(v))}</td></tr>"
        )
    return (
        "<section><h2>Metrics</h2>"
        "<table class='metrics'><thead><tr><th>name</th><th>value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _render_feedback(payload: dict[str, Any]) -> str:
    items = payload.get("feedback", []) or []
    if not items:
        return ""
    cards: list[str] = []
    for f in items:
        code = html.escape(str(f.get("code", "")))
        severity = html.escape(str(f.get("severity", "")))
        message = html.escape(str(f.get("message", "")))
        hint = html.escape(str(f.get("public_hint", "") or ""))
        cards.append(
            f"<div class='failure failure-{severity}'>"
            f"<div class='failure-code'>{code}"
            f" <span class='failure-sev'>{severity}</span></div>"
            f"<div class='failure-msg'>{message}</div>"
            f"{f'<div class=' + chr(34) + 'failure-hint' + chr(34) + '>' + hint + '</div>' if hint else ''}"
            f"</div>"
        )
    return "<section><h2>Feedback</h2>" + "".join(cards) + "</section>"


def _render_figures(payload: dict[str, Any]) -> str:
    traces = payload.get("traces", {}) or {}
    figs: list[str] = []

    coupler = traces.get("coupler_path") or []
    target = traces.get("target_path") or []
    if coupler or target:
        fig = go.Figure()
        if target:
            tx, ty = zip(*target)
            fig.add_trace(go.Scatter(
                x=list(tx), y=list(ty),
                mode="lines", name="target",
                line={"dash": "dash"}))
        if coupler:
            cx, cy = zip(*coupler)
            fig.add_trace(go.Scatter(
                x=list(cx), y=list(cy),
                mode="lines", name="observed coupler"))
        fig.update_layout(
            title="Coupler path",
            xaxis_title="x (mm)", yaxis_title="y (mm)",
            template="plotly_white",
            yaxis_scaleanchor="x",
        )
        figs.append(_fig_to_div(fig))

    input_a = traces.get("input_angle")
    output_a = traces.get("output_angle")
    if input_a or output_a:
        fig = go.Figure()
        if input_a:
            fig.add_trace(go.Scatter(
                x=input_a.get("t", []), y=input_a.get("theta", []),
                mode="lines", name="input"))
        if output_a:
            fig.add_trace(go.Scatter(
                x=output_a.get("t", []), y=output_a.get("theta", []),
                mode="lines", name="output"))
        fig.update_layout(
            title="Joint angles vs time",
            xaxis_title="t (s)", yaxis_title="angle (rad)",
            template="plotly_white",
        )
        figs.append(_fig_to_div(fig))

    ratio = traces.get("ratio_over_time")
    if ratio:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ratio.get("t", []),
            y=[0.0 if r is None else r for r in ratio.get("r", [])],
            mode="lines", name="output / input"))
        fig.update_layout(
            title="Transmission ratio over time",
            xaxis_title="t (s)", yaxis_title="ω_out / ω_in",
            template="plotly_white",
        )
        figs.append(_fig_to_div(fig))

    if not figs:
        return ""
    return "<section><h2>Traces</h2>" + "".join(figs) + "</section>"


def _fig_to_div(fig: Any) -> str:
    # include_plotlyjs="cdn" keeps the HTML small while staying self-contained
    # at viewing time (the page pulls plotly.js from the CDN once).
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


# --------------------------------------------------------------------- #
# Style                                                                 #
# --------------------------------------------------------------------- #


def _fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.4e}"
        return f"{v:.6g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return str(v)


_STYLE = """
body { font-family: -apple-system, system-ui, sans-serif;
       margin: 24px; color: #1d1d1f; }
h1 { font-size: 22px; margin: 0 0 16px; }
h2 { font-size: 16px; margin: 24px 0 8px;
     border-bottom: 1px solid #eee; padding-bottom: 4px; }
section.summary { display: flex; flex-wrap: wrap; gap: 12px; }
.card { background: #f5f5f7; border-radius: 8px; padding: 12px 16px;
        min-width: 140px; }
.card-label { font-size: 11px; color: #6e6e73; text-transform: uppercase;
              letter-spacing: 0.05em; }
.card-value { font-size: 18px; font-weight: 600; margin-top: 4px; }
table.metrics { border-collapse: collapse; font-size: 13px; }
table.metrics th, table.metrics td { text-align: left;
                                      border-bottom: 1px solid #eee;
                                      padding: 4px 12px 4px 0; }
.failure { background: #fff8e1; border-left: 4px solid #f5a623;
           padding: 8px 12px; margin: 6px 0; border-radius: 4px; }
.failure-critical { background: #fdecea; border-left-color: #d93025; }
.failure-major { background: #fff4e5; border-left-color: #f5a623; }
.failure-code { font-family: ui-monospace, Menlo, monospace; font-weight: 600; }
.failure-sev { font-size: 11px; color: #6e6e73; margin-left: 6px; }
.failure-msg { margin-top: 4px; font-size: 13px; }
.failure-hint { margin-top: 4px; font-size: 12px; color: #6e6e73; }
"""

"""Static-HTML dashboard for one run.

Consumes a payload dict produced by
:func:`mech_bench.dashboard_payload.build_dashboard_payload` and
writes a self-contained ``dashboard.html`` (no server needed). The
page renders six tabs:

* **Summary** — score cards, mode, oracle trust level
* **Tier breakdown** — score per tier channel (artifact, kinematics,
  …) with a small bar chart when plotly is available
* **Task metrics** — class-metric channels (linkage_path_score, etc.)
* **Trace plots** — coupler path, joint angles, ratio-over-time
* **Failure cards** — public feedback
* **Artifacts** — media manifest (mp4, thumbnail, payload, trace)

When ``preview.mp4`` exists in the same directory, the page embeds a
``<video>`` tag pointing at it; when only frames + warning are
available, the warning is surfaced instead.

Plotly is the only optional dependency for charts. If plotly is
absent, ``write_static_dashboard`` raises
:class:`DashboardUnavailableError`; the run bundle still has full
JSON evidence so a downstream tool can produce charts itself.
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
    tier_html = _render_tier_breakdown(payload)
    class_html = _render_class_metrics(payload)
    media_html = _render_media(payload)
    artifacts_html = _render_artifacts(payload)

    tabs = _tabs([
        ("Summary", summary_html + media_html),
        ("Tier breakdown", tier_html),
        ("Task metrics", class_html),
        ("Trace plots", figures_html),
        ("Failure cards", feedback_html),
        ("Metrics", metrics_html),
        ("Artifacts", artifacts_html),
    ])

    body = (
        "<h1>mech-bench run dashboard</h1>"
        f"{tabs}"
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
        ("Mode", str(run.get("mode", "") or "default")),
        ("Task", str(run.get("task_id", ""))),
        ("Family", str(run.get("task_family", ""))),
        ("Difficulty", str(run.get("difficulty", ""))),
        ("Run", str(run.get("run_id", ""))),
    ]
    if run.get("oracle_is_synthetic"):
        cards.append(("Oracle", "synthetic (fake)"))
    chunks = [
        f"<div class='card'><div class='card-label'>{html.escape(label)}</div>"
        f"<div class='card-value'>{html.escape(value)}</div></div>"
        for label, value in cards
    ]
    return "<section class='summary'>" + "".join(chunks) + "</section>"


def _render_tier_breakdown(payload: dict[str, Any]) -> str:
    tiers = payload.get("tier_results", {}) or {}
    if not tiers:
        return "<p>No tier breakdown available.</p>"
    rows: list[str] = []
    for k in sorted(tiers):
        v = tiers[k] or {}
        score = float(v.get("score", 0.0) or 0.0)
        passed = v.get("passed")
        n = v.get("n", "")
        passed_str = "—" if passed is None else ("yes" if passed else "no")
        rows.append(
            f"<tr><td>{html.escape(k)}</td>"
            f"<td>{score:.3f}</td>"
            f"<td>{html.escape(str(n))}</td>"
            f"<td>{passed_str}</td></tr>"
        )
    table = (
        "<table class='metrics'><thead>"
        "<tr><th>tier</th><th>score</th><th>n</th><th>passed</th></tr>"
        f"</thead><tbody>{''.join(rows)}</tbody></table>"
    )
    chart = ""
    if HAS_PLOTLY:
        fig = go.Figure(data=[go.Bar(
            x=sorted(tiers),
            y=[float((tiers.get(k) or {}).get("score", 0.0) or 0.0)
               for k in sorted(tiers)],
            marker={"color": "#5b8def"},
        )])
        fig.update_layout(
            title="Tier scores", template="plotly_white",
            yaxis={"range": [0, 1]},
        )
        chart = _fig_to_div(fig)
    return table + chart


def _render_class_metrics(payload: dict[str, Any]) -> str:
    cls = payload.get("class_metrics", {}) or {}
    if not cls:
        return "<p>No task-class metrics available.</p>"
    rows: list[str] = []
    for k in sorted(cls):
        v = float(cls[k] or 0.0)
        rows.append(
            f"<tr><td>{html.escape(k)}</td><td>{v:.3f}</td></tr>"
        )
    table = (
        "<table class='metrics'><thead>"
        "<tr><th>channel</th><th>score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    chart = ""
    if HAS_PLOTLY:
        fig = go.Figure(data=[go.Bar(
            x=sorted(cls),
            y=[float(cls.get(k, 0.0) or 0.0) for k in sorted(cls)],
            marker={"color": "#34a853"},
        )])
        fig.update_layout(
            title="Task-class metrics", template="plotly_white",
            yaxis={"range": [0, 1]},
        )
        chart = _fig_to_div(fig)
    return table + chart


def _render_media(payload: dict[str, Any]) -> str:
    media = payload.get("media") or {}
    if not media:
        return ""
    parts: list[str] = []
    preview = media.get("preview_mp4")
    thumb = media.get("thumbnail_png")
    frames_dir = media.get("frames_dir")
    if preview:
        parts.append(
            f"<video controls width='480' src='{html.escape(str(preview))}' "
            f"poster='{html.escape(str(thumb) if thumb else '')}'></video>"
        )
    elif thumb:
        parts.append(
            f"<img src='{html.escape(str(thumb))}' width='480' "
            f"alt='thumbnail'/>"
        )
    if not preview and frames_dir:
        parts.append(
            f"<p class='warning'>ffmpeg unavailable; frames retained at "
            f"<code>{html.escape(str(frames_dir))}</code></p>"
        )
    if not parts:
        return ""
    return "<section><h2>Preview</h2>" + "".join(parts) + "</section>"


def _render_artifacts(payload: dict[str, Any]) -> str:
    media = payload.get("media") or {}
    if not media:
        return "<p>No artifacts manifest available.</p>"
    rows: list[str] = []
    for k in sorted(media):
        v = media[k]
        if v is None or v == "" or isinstance(v, (dict, list)):
            continue
        rows.append(
            f"<tr><td>{html.escape(k)}</td>"
            f"<td><code>{html.escape(str(v))}</code></td></tr>"
        )
    return (
        "<table class='metrics'><thead>"
        "<tr><th>artifact</th><th>path</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _tabs(sections: list[tuple[str, str]]) -> str:
    """Render a list of (label, html_content) as JS-free tabs.

    Uses ``<details>`` with the first item open by default — keeps the
    page self-contained and viewable even without JavaScript.
    """
    if not sections:
        return ""
    chunks: list[str] = []
    for i, (label, content) in enumerate(sections):
        opened = " open" if i == 0 else ""
        chunks.append(
            f"<details class='tab'{opened}>"
            f"<summary>{html.escape(label)}</summary>"
            f"<div class='tab-body'>{content}</div></details>"
        )
    return "<section class='tabs'>" + "".join(chunks) + "</section>"


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


def write_benchmark_dashboard(
    payload: dict[str, Any], out_path: Path,
) -> Path:
    """Render the benchmark-suite payload as a static HTML page.

    Sections: overview cards, pass-rate funnel, tier heatmap, family
    heatmap, failure histogram, score distribution, per-task table.

    Plotly is required for charts; if missing, this raises
    :class:`DashboardUnavailableError`.
    """
    if not HAS_PLOTLY:
        raise DashboardUnavailableError(
            "plotly is required for the benchmark dashboard. "
            "Install with: pip install 'mech-bench[dashboard]'"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overview = payload.get("overview", {}) or {}
    funnel = payload.get("funnel", {}) or {}
    tier_rows = payload.get("tier_heatmap", []) or []
    fam_rows = payload.get("family_heatmap", []) or []
    fail_hist = payload.get("failure_histogram", {}) or {}
    score_bins = payload.get("score_distribution", []) or []
    task_table = payload.get("task_table", []) or []

    cards_html = "".join(
        f"<div class='card'><div class='card-label'>{html.escape(k)}</div>"
        f"<div class='card-value'>{_fmt_value(v)}</div></div>"
        for k, v in overview.items()
    )

    funnel_fig = go.Figure(go.Funnel(
        y=list(funnel.keys()),
        x=list(funnel.values()),
        textinfo="value+percent initial",
    ))
    funnel_fig.update_layout(template="plotly_white", title="Pass-rate funnel")

    tier_fig = go.Figure(data=[go.Bar(
        x=[r.get("key", "") for r in tier_rows],
        y=[float(r.get("score_mean", 0.0)) for r in tier_rows],
        marker={"color": "#5b8def"},
        name="score_mean",
    ), go.Bar(
        x=[r.get("key", "") for r in tier_rows],
        y=[float(r.get("pass_rate", 0.0)) for r in tier_rows],
        marker={"color": "#34a853"},
        name="pass_rate",
    )])
    tier_fig.update_layout(
        title="Tier heatmap", template="plotly_white",
        barmode="group", yaxis={"range": [0, 1]})

    fam_fig = go.Figure(data=[go.Bar(
        x=[r.get("key", "") for r in fam_rows],
        y=[float(r.get("score_mean", 0.0)) for r in fam_rows],
        marker={"color": "#fbbc04"},
        name="score_mean",
    ), go.Bar(
        x=[r.get("key", "") for r in fam_rows],
        y=[float(r.get("pass_rate", 0.0)) for r in fam_rows],
        marker={"color": "#34a853"},
        name="pass_rate",
    )])
    fam_fig.update_layout(
        title="Family heatmap", template="plotly_white",
        barmode="group", yaxis={"range": [0, 1]})

    if fail_hist:
        hist_fig = go.Figure(data=[go.Bar(
            x=list(fail_hist.keys()),
            y=list(fail_hist.values()),
            marker={"color": "#d93025"},
        )])
        hist_fig.update_layout(
            title="Failure-code histogram", template="plotly_white")
        hist_html = _fig_to_div(hist_fig)
    else:
        hist_html = "<p>No failure codes recorded.</p>"

    if score_bins:
        dist_fig = go.Figure(data=[go.Bar(
            x=[f"{b['low']:.1f}–{b['high']:.1f}" for b in score_bins],
            y=[int(b.get("count", 0)) for b in score_bins],
            marker={"color": "#5b8def"},
        )])
        dist_fig.update_layout(
            title="Score distribution", template="plotly_white")
        dist_html = _fig_to_div(dist_fig)
    else:
        dist_html = ""

    table_rows: list[str] = []
    for row in task_table:
        codes = ", ".join(row.get("failure_codes") or [])
        score = row.get("overall_score") or 0.0
        report_dir = row.get("report_dir") or ""
        link = (
            f"<a href='{html.escape(str(report_dir))}/dashboard.html'>"
            f"{html.escape(str(row.get('task_id', '')))}</a>"
            if report_dir else html.escape(str(row.get("task_id", "")))
        )
        table_rows.append(
            "<tr>"
            f"<td>{link}</td>"
            f"<td>{html.escape(str(row.get('family', '')))}</td>"
            f"<td>{html.escape(str(row.get('tier', '')))}</td>"
            f"<td>{float(score):.3f}</td>"
            f"<td>{html.escape(codes)}</td>"
            "</tr>"
        )
    task_table_html = (
        "<table class='metrics'><thead>"
        "<tr><th>task</th><th>family</th><th>tier</th>"
        "<th>score</th><th>failure codes</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>"
    )

    body = (
        "<h1>mech-bench benchmark dashboard</h1>"
        f"<section class='summary'>{cards_html}</section>"
        + _tabs([
            ("Funnel", _fig_to_div(funnel_fig)),
            ("Tier heatmap", _fig_to_div(tier_fig)),
            ("Family heatmap", _fig_to_div(fam_fig)),
            ("Failure histogram", hist_html),
            ("Score distribution", dist_html),
            ("Tasks", task_table_html),
        ])
    )
    doc = (
        "<!DOCTYPE html>\n"
        "<html lang='en'><head><meta charset='utf-8'>"
        "<title>mech-bench: benchmark summary</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"{body}"
        "</body></html>"
    )
    out_path.write_text(doc)
    return out_path


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
section.tabs { margin-top: 16px; }
details.tab { border: 1px solid #e5e5e7; border-radius: 6px;
              margin: 6px 0; padding: 8px 12px; }
details.tab summary { font-weight: 600; cursor: pointer; }
details.tab .tab-body { margin-top: 8px; }
.warning { background: #fff8e1; border-left: 4px solid #f5a623;
           padding: 6px 10px; border-radius: 4px; font-size: 13px; }
video, img { border-radius: 6px; }
"""

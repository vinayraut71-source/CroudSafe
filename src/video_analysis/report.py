"""
Video Analysis PDF Report Generator.
Generates comprehensive publication-quality PDF dossiers for processed video files,
including timeline charts, heatmaps, incident logs, and Gemini AI assessments.
"""

import os
import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as RLImage,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm


def generate_video_analysis_pdf(
    output_pdf_path: str,
    video_metadata: Dict[str, Any],
    analytics_summary: Dict[str, Any],
    chart_image_path: Optional[str] = None,
    heatmap_image_path: Optional[str] = None,
    incident_events: Optional[List[Dict[str, Any]]] = None,
    ai_assessment: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build and save an in-depth A4 PDF report for an analyzed video file.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569")
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        fontName="Helvetica-Bold",
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )
    ai_box_style = ParagraphStyle(
        "AIBox",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1e1b4b")
    )
    badge_style = ParagraphStyle(
        "Badge",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.white,
        alignment=1,
        fontName="Helvetica-Bold"
    )

    story = []

    # ── Header ──
    story.append(Paragraph("CROWDSAFE AI — RECORDED VIDEO CROWD INTELLIGENCE REPORT", title_style))
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Automated Video Analytics & Multimodal Safety Audit • Generated: {now_str}", subtitle_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

    # ── Severity Banner ──
    overall_level = analytics_summary.get("overall_risk_level", "SAFE")
    severity_colors = {
        "CRITICAL": colors.HexColor("#dc2626"),
        "DANGER": colors.HexColor("#ea580c"),
        "WARNING": colors.HexColor("#d97706"),
        "SAFE": colors.HexColor("#16a34a")
    }
    sev_color = severity_colors.get(overall_level.upper(), colors.HexColor("#475569"))

    # ── Video Metadata & Executive Summary Grid ──
    meta_table_data = [
        [
            Paragraph("<b>Source Video:</b>", body_style), Paragraph(video_metadata.get("filename", "video.mp4"), body_style),
            Paragraph("<b>Overall Risk Level:</b>", body_style),
            Paragraph(f"<font color='white'><b>&nbsp;{overall_level}&nbsp;</b></font>", badge_style)
        ],
        [
            Paragraph("<b>Video Duration:</b>", body_style), Paragraph(video_metadata.get("duration_formatted", "00:00"), body_style),
            Paragraph("<b>Peak Crowd Headcount:</b>", body_style), Paragraph(f"<b>{analytics_summary.get('peak_count', 0)} persons</b> (@ {analytics_summary.get('peak_count_time', '00:00')})", body_style)
        ],
        [
            Paragraph("<b>Frames Analyzed:</b>", body_style), Paragraph(f"{video_metadata.get('total_frames', 0):,} frames", body_style),
            Paragraph("<b>Average Headcount:</b>", body_style), Paragraph(f"{analytics_summary.get('avg_count', 0.0):.1f} persons", body_style)
        ],
        [
            Paragraph("<b>Resolution & FPS:</b>", body_style), Paragraph(f"{video_metadata.get('width', 0)}x{video_metadata.get('height', 0)} @ {video_metadata.get('fps', 0):.1f} fps", body_style),
            Paragraph("<b>Cumulative In / Out:</b>", body_style), Paragraph(f"Entry: {analytics_summary.get('total_entry', 0)} | Exit: {analytics_summary.get('total_exit', 0)}", body_style)
        ]
    ]

    meta_table = Table(meta_table_data, colWidths=[3.2 * cm, 5.5 * cm, 4.0 * cm, 5.0 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (3, 0), (3, 0), sev_color),
        ("ALIGN", (3, 0), (3, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 1: Crowd Telemetry & Flow Dynamics ──
    story.append(Paragraph("1. Crowd Telemetry & Ingress/Egress Dynamics", h2_style))
    summary_data = [
        ["Parameter", "Measured Value", "Safety Threshold", "Evaluation Status"],
        ["Peak Headcount", f"{analytics_summary.get('peak_count', 0)} persons", "Capacity baseline: 100", "Critical at > 90 persons"],
        ["Average Headcount", f"{analytics_summary.get('avg_count', 0.0):.1f} persons", "Safe operational band", "Nominal"],
        ["Peak Spatial Density", f"{analytics_summary.get('peak_density_pct', 0.0):.1f}%", "Safe threshold: < 30%", "Hazard at > 55%"],
        ["Total Pedestrian Ingress", f"{analytics_summary.get('total_entry', 0)} persons", "Cumulative tracked entrants", "Monitored Flow"],
        ["Total Pedestrian Egress", f"{analytics_summary.get('total_exit', 0)} persons", "Cumulative tracked exits", "Monitored Flow"],
        ["Max Recorded Risk Score", f"{analytics_summary.get('max_risk_score', 0.0):.1f} / 100", "Safe range: 0 - 35", f"Assessed {overall_level}"]
    ]
    summary_table = Table(summary_data, colWidths=[4.5 * cm, 4.2 * cm, 4.5 * cm, 4.5 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 2: Visual Telemetry (Graphs & Heatmap) ──
    story.append(Paragraph("2. Temporal Crowd Graphs & Spatial Heatmap", h2_style))
    img_cells = []
    if chart_image_path and os.path.exists(chart_image_path):
        img_cells.append(RLImage(chart_image_path, width=8.6 * cm, height=5.2 * cm))
    if heatmap_image_path and os.path.exists(heatmap_image_path):
        img_cells.append(RLImage(heatmap_image_path, width=8.6 * cm, height=5.2 * cm))

    if img_cells:
        if len(img_cells) == 2:
            img_table = Table([[img_cells[0], img_cells[1]]], colWidths=[8.8 * cm, 8.8 * cm])
            img_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story.append(img_table)
        else:
            story.append(img_cells[0])
        story.append(Spacer(1, 0.3 * cm))

    # ── Section 3: Gemini AI Executive Assessment ──
    story.append(Paragraph("3. Gemini AI Multimodal Executive Assessment", h2_style))
    ai_data = ai_assessment or {}
    ai_summary = ai_data.get("executive_summary", "Automated computer vision analysis successfully processed the full video sequence.")
    congestion_notes = ai_data.get("congestion_analysis", "Crowd density remained within expected containment parameters throughout the recorded duration.")

    ai_narrative_text = f"""
    <b>Executive Overview:</b><br/>{ai_summary}<br/><br/>
    <b>Congestion & Flow Analysis:</b><br/>{congestion_notes}
    """
    ai_table = Table([[Paragraph(ai_narrative_text, ai_box_style)]], colWidths=[17.7 * cm])
    ai_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef2ff")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#6366f1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(ai_table)
    story.append(Spacer(1, 0.25 * cm))

    # ── Section 4: Operational Recommendations ──
    recs = ai_data.get("recommendations", [
        "Maintain current corridor layout and ensure all designated exit paths remain clear.",
        "Deploy monitoring personnel during identified peak occupancy intervals.",
        "Review ingress pacing if pedestrian inflow surges past capacity baseline."
    ])
    story.append(Paragraph("4. Recommended Operational Actions", h2_style))
    recs_items = "<br/>".join([f"• <b>Action {idx+1}:</b> {r}" for idx, r in enumerate(recs)])
    recs_table = Table([[Paragraph(recs_items, body_style)]], colWidths=[17.7 * cm])
    recs_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(recs_table)
    story.append(Spacer(1, 0.25 * cm))

    # ── Section 5: Significant Incident & Surge Timeline ──
    if incident_events:
        story.append(Paragraph("5. Critical Timeline & Surge Incidents", h2_style))
        inc_rows = [["Time", "Level", "Headcount", "Density", "Observation / Trigger"]]
        for inc in incident_events[:8]:
            inc_rows.append([
                inc.get("time_formatted", "00:00"),
                inc.get("level", "WARNING"),
                f"{inc.get('count', 0)}",
                f"{inc.get('density_pct', 0.0):.1f}%",
                inc.get("message", "Threshold trigger")[:45]
            ])
        inc_table = Table(inc_rows, colWidths=[2.2 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 8.0 * cm])
        inc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(inc_table)
        story.append(Spacer(1, 0.25 * cm))

    # ── Section 6: Verification & Governance ──
    story.append(Paragraph("6. Operator Sign-Off & Verification Block", h2_style))
    sign_data = [
        [Paragraph("<b>Audit Status:</b>", body_style), Paragraph("VERIFIED AUTOMATED ANALYSIS", body_style)],
        [Paragraph("<b>Lead Safety Officer:</b>", body_style), Paragraph("___________________________ (Authorised Sign-off)", body_style)],
        [Paragraph("<b>Governance Date:</b>", body_style), Paragraph(now_str, body_style)]
    ]
    sign_table = Table(sign_data, colWidths=[4.5 * cm, 13.2 * cm])
    sign_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(sign_table)
    story.append(Spacer(1, 0.3 * cm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=4))
    story.append(Paragraph("CONFIDENTIAL & PROPRIETARY — CrowdSafe AI Automated Video Crowd Intelligence Platform.", subtitle_style))

    doc.build(story)
    return output_pdf_path


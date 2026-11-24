"""
Automated Incident Report Generator & PDF Exporter.
Converts structured crowd alerts, Gemini AI explanations, and multimodal visual findings
into formal, publication-quality incident reports with operator verification.
"""

import os
import time
import json
import sqlite3
import datetime
import uuid
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DB_PATH, REPORT_DIR, RECORDING_DIR
from src.nlp_assistant.schemas import IncidentReport, IncidentReportCreate


class IncidentReportManager:
    """Manages creation, storage, operator verification, and PDF generation for incident reports."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.reports_dir = os.path.join(REPORT_DIR, "incidents")
        os.makedirs(self.reports_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Ensure incidents table exists in SQLite with AI columns."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS incidents (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        timestamp REAL,
                        time_str TEXT,
                        event_type TEXT,
                        severity TEXT,
                        camera_id TEXT,
                        zone TEXT,
                        count INTEGER,
                        density REAL,
                        occupancy_pct REAL,
                        risk_score REAL,
                        evidence_url TEXT,
                        verification_status TEXT,
                        operator_notes TEXT,
                        verified_at REAL,
                        ai_explanation TEXT,
                        visual_observations TEXT,
                        recommended_actions TEXT,
                        data TEXT
                    )
                """)
                # Migrations: check if AI columns exist in case table was created previously
                existing_cols = [c[1] for c in conn.execute("PRAGMA table_info(incidents)").fetchall()]
                if "ai_explanation" not in existing_cols:
                    conn.execute("ALTER TABLE incidents ADD COLUMN ai_explanation TEXT")
                if "visual_observations" not in existing_cols:
                    conn.execute("ALTER TABLE incidents ADD COLUMN visual_observations TEXT")
                if "recommended_actions" not in existing_cols:
                    conn.execute("ALTER TABLE incidents ADD COLUMN recommended_actions TEXT")
                conn.commit()
        except Exception as e:
            print(f"[IncidentReportManager] DB init error: {e}")

    def create_report(
        self,
        event_type: str,
        severity: str,
        camera_id: str = "CAM-1",
        zone: str = "Zone-1",
        count: int = 0,
        density: float = 0.0,
        occupancy_pct: float = 0.0,
        risk_score: float = 0.0,
        evidence_url: Optional[str] = None,
        operator_notes: Optional[str] = None,
        ai_explanation: Optional[str] = None,
        visual_observations: Optional[str] = None,
        recommended_actions: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
        session_id: str = "default_session"
    ) -> IncidentReport:
        """Create a new structured incident report and store it in SQLite."""
        now = time.time()
        time_str = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        report_id = f"INC-{datetime.datetime.fromtimestamp(now).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"

        # If no explicit evidence_url is passed, scan for the most recent matching recording or snapshot
        if not evidence_url and os.path.exists(RECORDING_DIR):
            snapshots_dir = os.path.join(RECORDING_DIR, "snapshots")
            if os.path.exists(snapshots_dir):
                snaps = [f for f in os.listdir(snapshots_dir) if f.endswith(".jpg")]
                if snaps:
                    snaps.sort(key=lambda f: os.path.getmtime(os.path.join(snapshots_dir, f)), reverse=True)
                    evidence_url = f"/recordings/snapshots/{snaps[0]}"

            if not evidence_url:
                recordings = [f for f in os.listdir(RECORDING_DIR) if f.endswith(".mp4")]
                if recordings:
                    recordings.sort(key=lambda f: os.path.getmtime(os.path.join(RECORDING_DIR, f)), reverse=True)
                    evidence_url = f"/recordings/{recordings[0]}"

        # Auto-generate AI explanation if not provided
        recs = recommended_actions or []
        if not ai_explanation:
            try:
                from src.gemini_service.explainer import IncidentExplainer
                explainer = IncidentExplainer()
                payload = {
                    "level": severity,
                    "risk_score": risk_score,
                    "metrics": {
                        "count": count,
                        "density": density,
                        "occupancy_pct": occupancy_pct,
                        **(data or {})
                    }
                }
                exp_res = explainer.explain_incident(alert_payload=payload, include_snapshot=False)
                ai_explanation = exp_res.gemini_explanation
                if not recs and exp_res.recommended_actions:
                    recs = exp_res.recommended_actions
            except Exception as e:
                ai_explanation = f"Automated safety telemetry recorded a {severity} status ({event_type})."

        report = IncidentReport(
            id=report_id,
            timestamp=now,
            time_str=time_str,
            event_type=event_type,
            severity=severity,
            camera_id=camera_id,
            zone=zone,
            count=count,
            density=round(density, 4),
            occupancy_pct=round(occupancy_pct, 2),
            risk_score=round(risk_score, 2),
            evidence_url=evidence_url,
            verification_status="UNVERIFIED",
            operator_notes=operator_notes,
            ai_explanation=ai_explanation,
            visual_observations=visual_observations,
            recommended_actions=recs,
            verified_at=None,
            data=data or {}
        )

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO incidents (
                        id, session_id, timestamp, time_str, event_type, severity,
                        camera_id, zone, count, density, occupancy_pct, risk_score,
                        evidence_url, verification_status, operator_notes, verified_at,
                        ai_explanation, visual_observations, recommended_actions, data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    report.id,
                    session_id,
                    report.timestamp,
                    report.time_str,
                    report.event_type,
                    report.severity,
                    report.camera_id,
                    report.zone,
                    report.count,
                    report.density,
                    report.occupancy_pct,
                    report.risk_score,
                    report.evidence_url,
                    report.verification_status,
                    report.operator_notes,
                    report.verified_at,
                    report.ai_explanation,
                    report.visual_observations,
                    json.dumps(report.recommended_actions),
                    json.dumps(report.data)
                ))
                conn.commit()
                print(f"[IncidentReportManager] Incident report saved: {report.id} ({severity})")
        except Exception as e:
            print(f"[IncidentReportManager] Error saving incident report: {e}")

        return report

    def update_verification(
        self,
        report_id: str,
        status: str,
        operator_notes: Optional[str] = None
    ) -> Optional[IncidentReport]:
        """Update the human operator verification sign-off on an incident report."""
        valid_statuses = ("UNVERIFIED", "VERIFIED_FALSE_POSITIVE", "CONFIRMED_HAZARD", "RESOLVED")
        if status not in valid_statuses:
            raise ValueError(f"Invalid verification status '{status}'. Must be one of {valid_statuses}")

        now = time.time()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE incidents
                    SET verification_status = ?, operator_notes = COALESCE(?, operator_notes), verified_at = ?
                    WHERE id = ?
                """, (status, operator_notes, now, report_id))
                conn.commit()
        except Exception as e:
            print(f"[IncidentReportManager] Error updating verification for {report_id}: {e}")
            return None

        return self.get_report(report_id)

    def get_report(self, report_id: str) -> Optional[IncidentReport]:
        """Fetch single incident report by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM incidents WHERE id = ?", (report_id,)).fetchone()
                if not row:
                    return None

                data = json.loads(row["data"]) if row["data"] else {}
                recs = []
                if "recommended_actions" in row.keys() and row["recommended_actions"]:
                    try:
                        recs = json.loads(row["recommended_actions"])
                    except Exception:
                        recs = [row["recommended_actions"]]

                return IncidentReport(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    time_str=row["time_str"],
                    event_type=row["event_type"],
                    severity=row["severity"],
                    camera_id=row["camera_id"],
                    zone=row["zone"],
                    count=row["count"],
                    density=row["density"],
                    occupancy_pct=row["occupancy_pct"],
                    risk_score=row["risk_score"],
                    evidence_url=row["evidence_url"],
                    verification_status=row["verification_status"],
                    operator_notes=row["operator_notes"],
                    ai_explanation=row["ai_explanation"] if "ai_explanation" in row.keys() else None,
                    visual_observations=row["visual_observations"] if "visual_observations" in row.keys() else None,
                    recommended_actions=recs,
                    verified_at=row["verified_at"],
                    data=data
                )
        except Exception as e:
            print(f"[IncidentReportManager] Error fetching report {report_id}: {e}")
            return None

    def list_reports(self, limit: int = 50) -> List[IncidentReport]:
        """List historical incident reports ordered by timestamp descending."""
        reports = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
                for row in rows:
                    data = json.loads(row["data"]) if row["data"] else {}
                    recs = []
                    if "recommended_actions" in row.keys() and row["recommended_actions"]:
                        try:
                            recs = json.loads(row["recommended_actions"])
                        except Exception:
                            recs = [row["recommended_actions"]]

                    reports.append(IncidentReport(
                        id=row["id"],
                        timestamp=row["timestamp"],
                        time_str=row["time_str"],
                        event_type=row["event_type"],
                        severity=row["severity"],
                        camera_id=row["camera_id"],
                        zone=row["zone"],
                        count=row["count"],
                        density=row["density"],
                        occupancy_pct=row["occupancy_pct"],
                        risk_score=row["risk_score"],
                        evidence_url=row["evidence_url"],
                        verification_status=row["verification_status"],
                        operator_notes=row["operator_notes"],
                        ai_explanation=row["ai_explanation"] if "ai_explanation" in row.keys() else None,
                        visual_observations=row["visual_observations"] if "visual_observations" in row.keys() else None,
                        recommended_actions=recs,
                        verified_at=row["verified_at"],
                        data=data
                    ))
        except Exception as e:
            print(f"[IncidentReportManager] Error listing reports: {e}")
        return reports

    def export_pdf(self, report_id: str) -> str:
        """
        Generate a professional, publication-quality A4 PDF report using ReportLab.
        Enriched with Gemini AI Causal Explanations and Multimodal Visual Evidence.
        """
        report = self.get_report(report_id)
        if not report:
            raise ValueError(f"Report ID '{report_id}' not found.")

        pdf_path = os.path.join(self.reports_dir, f"{report.id}.pdf")

        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm

        doc = SimpleDocTemplate(
            pdf_path,
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

        # Header Title Banner
        story.append(Paragraph("CROWDSAFE AI — MULTIMODAL INCIDENT INTELLIGENCE REPORT", title_style))
        story.append(Paragraph(f"Official Safety Dossier • Reference: {report.id} • Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
        story.append(Spacer(1, 0.2 * cm))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))

        # Severity color picker
        severity_colors = {
            "CRITICAL": colors.HexColor("#dc2626"),
            "DANGER": colors.HexColor("#ea580c"),
            "WARNING": colors.HexColor("#d97706"),
            "SAFE": colors.HexColor("#16a34a")
        }
        sev_color = severity_colors.get(report.severity.upper(), colors.HexColor("#475569"))

        # Primary Metadata Grid
        meta_table_data = [
            [
                Paragraph("<b>Report ID:</b>", body_style), Paragraph(report.id, body_style),
                Paragraph("<b>Incident Severity:</b>", body_style),
                Paragraph(f"<font color='white'><b>&nbsp;{report.severity}&nbsp;</b></font>", badge_style)
            ],
            [
                Paragraph("<b>Event Category:</b>", body_style), Paragraph(report.event_type.replace("_", " "), body_style),
                Paragraph("<b>Verification Status:</b>", body_style), Paragraph(f"<b>{report.verification_status}</b>", body_style)
            ],
            [
                Paragraph("<b>Incident Timestamp:</b>", body_style), Paragraph(report.time_str, body_style),
                Paragraph("<b>Location / Camera:</b>", body_style), Paragraph(f"{report.camera_id} • {report.zone}", body_style)
            ]
        ]
        meta_table = Table(meta_table_data, colWidths=[3.2 * cm, 5.5 * cm, 4.0 * cm, 5.0 * cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BACKGROUND", (3, 0), (3, 0), sev_color),
            ("ALIGN", (3, 0), (3, 0), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.3 * cm))

        # Measured Crowd Analytics Section
        story.append(Paragraph("1. Deterministic Sensor Telemetry & Crowd Analytics", h2_style))
        metrics_data = [
            ["Metric Parameter", "Measured Value", "Operational Benchmark", "Threshold Status"],
            ["Crowd Headcount", f"{report.count} persons", "Venue baseline: 100", "Warning at > 70 persons"],
            ["Measured Density", f"{report.density * 100:.1f}% ({report.density:.3f})", "Normal: < 30%", "Hazard at > 55%"],
            ["Venue Occupancy", f"{report.occupancy_pct:.1f}%", "Safe capacity: 100%", "Danger at > 90%"],
            ["Official Risk Score", f"{report.risk_score:.1f} / 100", "Safe zone: 0 - 35", "Elevated at > 50"]
        ]

        data_dict = report.data or {}
        if "avg_speed" in data_dict:
            metrics_data.append(["Mean Crowd Speed", f"{data_dict['avg_speed']:.2f} px/f", "Baseline: 1.5 - 2.5 px/f", f"{data_dict.get('speed_multiplier', 1.0):.2f}x Baseline"])
        if "acoustic_db" in data_dict:
            metrics_data.append(["Acoustic Level (SPL)", f"{data_dict['acoustic_db']:.1f} dB", "Ambient: < 65 dB", "Panic Spike" if data_dict.get("acoustic_panic") else "Safe Level"])
        if "direction_entropy" in data_dict:
            metrics_data.append(["Directional Entropy", f"{data_dict['direction_entropy']:.2f}", "Normal Social: 0.7 - 1.0", "Unidirectional Rush" if data_dict['direction_entropy'] < 0.4 else "Coherent Flow"])

        metrics_table = Table(metrics_data, colWidths=[4.5 * cm, 4.2 * cm, 4.5 * cm, 4.5 * cm])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3 * cm))

        # Gemini AI Incident Explanation Section
        story.append(Paragraph("2. Gemini AI Incident Intelligence & Causal Analysis", h2_style))
        ai_narrative = report.ai_explanation or "Automated crowd monitoring telemetry detected physical threshold crossing."
        ai_table_data = [[
            Paragraph(f"<b>AI Causal Assessment:</b><br/>{ai_narrative}", ai_box_style)
        ]]
        ai_table = Table(ai_table_data, colWidths=[17.7 * cm])
        ai_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef2ff")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#6366f1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 0.3 * cm))

        # Actionable Operator Recommendations
        recs = report.recommended_actions or []
        if recs:
            story.append(Paragraph("3. Recommended Operational Interventions", h2_style))
            recs_items = "<br/>".join([f"• <b>Action {idx+1}:</b> {r}" for idx, r in enumerate(recs)])
            recs_table = Table([[Paragraph(recs_items, body_style)]], colWidths=[17.7 * cm])
            recs_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(recs_table)
            story.append(Spacer(1, 0.3 * cm))

        # Supporting Visual Evidence Section
        story.append(Paragraph("4. Optical Evidence & Visual Recording Dossier", h2_style))
        evidence_text = f"""
        <b>Associated Camera Feed:</b> {report.camera_id} (Zone: {report.zone})<br/>
        <b>Evidence Media Reference:</b> {report.evidence_url or 'Live Snapshot captured and indexed'}<br/>
        <b>Trigger Classification:</b> {report.event_type} (Official Severity: {report.severity})<br/>
        """
        story.append(Paragraph(evidence_text, body_style))

        # Try embedding snapshot image into PDF if file exists on disk
        if report.evidence_url and report.evidence_url.endswith(".jpg"):
            clean_rel = report.evidence_url.replace("/recordings/", "")
            full_img_path = os.path.join(RECORDING_DIR, clean_rel)
            if os.path.exists(full_img_path):
                try:
                    img_flowable = RLImage(full_img_path, width=14.0 * cm, height=7.5 * cm)
                    story.append(Spacer(1, 0.2 * cm))
                    story.append(img_flowable)
                except Exception as img_err:
                    print(f"[PDF] Snapshot embed skipped: {img_err}")

        story.append(Spacer(1, 0.3 * cm))

        # Operator Verification & Sign-off Section
        story.append(Paragraph("5. Operator Verification & Governance Sign-Off", h2_style))
        verif_notes = report.operator_notes or "No manual operator commentary provided."
        verif_time = datetime.datetime.fromtimestamp(report.verified_at).strftime("%Y-%m-%d %H:%M:%S") if report.verified_at else "Pending Verification"

        verif_table_data = [
            [Paragraph("<b>Verification Decision:</b>", body_style), Paragraph(f"<b>{report.verification_status}</b>", body_style)],
            [Paragraph("<b>Verification Timestamp:</b>", body_style), Paragraph(verif_time, body_style)],
            [Paragraph("<b>Operator Observations:</b>", body_style), Paragraph(verif_notes, body_style)],
            [Paragraph("<b>Authorised Signature:</b>", body_style), Paragraph("___________________________ (Safety Controller)", body_style)]
        ]
        verif_table = Table(verif_table_data, colWidths=[4.5 * cm, 13.2 * cm])
        verif_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(verif_table)
        story.append(Spacer(1, 0.3 * cm))

        # Footer note
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=4))
        story.append(Paragraph("CONFIDENTIAL & PROPRIETARY — Generated by CrowdSafe AI Multimodal Crowd Intelligence Platform.", subtitle_style))

        doc.build(story)
        print(f"[IncidentReportManager] PDF exported: {pdf_path}")
        return pdf_path

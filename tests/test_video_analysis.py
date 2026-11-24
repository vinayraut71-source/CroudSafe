"""
Test suite for Video Upload & Automated Crowd Analysis for CrowdSafe AI.
Verifies file validation, YOLO detection & tracking pipeline on video files,
chart plotting, heatmap generation, Gemini AI summaries, and PDF report creation.
"""

import os
import sys
import tempfile
import cv2
import numpy as np
import pytest
import io
import json
from unittest.mock import MagicMock
from fastapi import HTTPException
from web.server import api_video_status, api_video_jobs, ALLOWED_VIDEO_EXTS, video_job_manager
from src.video_analysis import VideoJob, VideoFileAnalyzer, VideoJobManager, generate_video_analysis_pdf


@pytest.fixture
def test_video_path(tmp_path):
    """Generate a valid 20-frame synthetic test video file."""
    vid_file = str(tmp_path / "sample_crowd_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(vid_file, fourcc, 10.0, (640, 480))
    for i in range(20):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw synthetic white blob to simulate detected object
        cv2.circle(frame, (120 + i * 10, 200), 25, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    return vid_file


class TestVideoFormatValidation:
    def test_allowed_extensions(self):
        assert ".mp4" in ALLOWED_VIDEO_EXTS
        assert ".avi" in ALLOWED_VIDEO_EXTS
        assert ".mov" in ALLOWED_VIDEO_EXTS
        assert ".mkv" in ALLOWED_VIDEO_EXTS
        assert ".pdf" not in ALLOWED_VIDEO_EXTS
        assert ".exe" not in ALLOWED_VIDEO_EXTS
        assert ".sh" not in ALLOWED_VIDEO_EXTS

    def test_extension_validation_logic(self):
        valid_files = ["crowd.mp4", "stadium_view.AVI", "gate_cam.MoV", "downtown.MKV"]
        for fn in valid_files:
            ext = os.path.splitext(fn)[1].lower()
            assert ext in ALLOWED_VIDEO_EXTS

        invalid_files = ["exploit.exe", "script.sh", "data.csv", "document.pdf"]
        for fn in invalid_files:
            ext = os.path.splitext(fn)[1].lower()
            assert ext not in ALLOWED_VIDEO_EXTS


class TestVideoJobLifecycle:
    def test_job_lifecycle(self, test_video_path):
        mgr = VideoJobManager()
        job = mgr.create_job(file_path=test_video_path, filename="sample.mp4")
        assert job.job_id.startswith("vid_")
        assert job.status in ("queued", "processing", "completed")

        fetched = mgr.get_job(job.job_id)
        assert fetched is not None
        assert fetched.filename == "sample.mp4"

        jobs_list = mgr.list_jobs()
        assert len(jobs_list) >= 1
        assert any(j["job_id"] == job.job_id for j in jobs_list)

    def test_job_to_dict_schema(self, test_video_path):
        job = VideoJob(job_id="vid_test_999", filename="test.mp4", source_path=test_video_path)
        d = job.to_dict()
        assert "job_id" in d
        assert "filename" in d
        assert "status" in d
        assert "progress_pct" in d
        assert "current_frame" in d
        assert "total_frames" in d
        assert "eta_seconds" in d
        assert "result" in d


class TestVideoEndpoints:
    def test_get_jobs_list_endpoint(self):
        res = api_video_jobs(limit=10)
        assert res.status_code == 200
        data = json.loads(res.body)
        assert isinstance(data, list)

    def test_get_job_status_not_found(self):
        with pytest.raises(HTTPException) as exc:
            api_video_status("non_existent_job_12345")
        assert exc.value.status_code == 404

    def test_get_job_status_existing(self, test_video_path):
        job = video_job_manager.create_job(file_path=test_video_path, filename="test.mp4")
        res = api_video_status(job.job_id)
        assert res.status_code == 200
        data = json.loads(res.body)
        assert data["job_id"] == job.job_id


class TestVideoFileAnalyzer:
    def test_end_to_end_video_processing(self, test_video_path, tmp_path):
        job = VideoJob(job_id="test_job_101", filename="sample_crowd_test.mp4", source_path=test_video_path)
        analyzer = VideoFileAnalyzer(job)
        result = analyzer.process()

        assert "annotated_video_url" in result
        assert "chart_url" in result
        assert "heatmap_url" in result
        assert "pdf_report_url" in result
        assert "summary" in result
        assert "ai_assessment" in result

        # Verify disk artifacts
        clean_vid = result["annotated_video_url"].replace("/recordings/", "recordings/")
        clean_chart = result["chart_url"].replace("/recordings/", "recordings/")
        clean_hm = result["heatmap_url"].replace("/recordings/", "recordings/")
        clean_pdf = result["pdf_report_url"].replace("/recordings/", "recordings/")

        assert os.path.exists(clean_vid)
        assert os.path.exists(clean_chart)
        assert os.path.exists(clean_hm)
        assert os.path.exists(clean_pdf)

        # Validate PDF magic header
        with open(clean_pdf, "rb") as f:
            assert f.read(5) == b"%PDF-"


class TestVideoReportGeneration:
    def test_pdf_generation_standalone(self, tmp_path):
        pdf_out = str(tmp_path / "standalone_video_report.pdf")
        metadata = {
            "filename": "plaza_security_cam.mp4",
            "duration_formatted": "02:15",
            "total_frames": 2700,
            "width": 1920,
            "height": 1080,
            "fps": 20.0
        }
        summary = {
            "overall_risk_level": "WARNING",
            "peak_count": 48,
            "peak_count_time": "01:12",
            "avg_count": 22.4,
            "peak_density_pct": 52.0,
            "total_entry": 64,
            "total_exit": 21,
            "max_risk_score": 58.0
        }
        incidents = [
            {"time_sec": 72.0, "time_formatted": "01:12", "level": "WARNING", "count": 48, "density_pct": 52.0, "message": "High density warning"}
        ]
        ai_assessment = {
            "executive_summary": "Crowd analysis identified a localized density surge midway through the recording.",
            "congestion_analysis": "Inflow significantly exceeded outflow in the first 90 seconds.",
            "recommendations": ["Open secondary exit", "Deploy marshals to entrance gate"]
        }

        generated_path = generate_video_analysis_pdf(
            output_pdf_path=pdf_out,
            video_metadata=metadata,
            analytics_summary=summary,
            incident_events=incidents,
            ai_assessment=ai_assessment
        )
        assert os.path.exists(generated_path)
        with open(generated_path, "rb") as f:
            assert f.read(5) == b"%PDF-"

"""
Video Analysis Module for CrowdSafe AI.
Provides automated video file processing, tracking, crowd counting,
heatmap generation, timeline charting, Gemini AI summaries, and PDF dossiers.
"""

from src.video_analysis.analyzer import VideoJob, VideoJobManager, VideoFileAnalyzer, video_job_manager
from src.video_analysis.report import generate_video_analysis_pdf

__all__ = [
    "VideoJob",
    "VideoJobManager",
    "VideoFileAnalyzer",
    "video_job_manager",
    "generate_video_analysis_pdf"
]


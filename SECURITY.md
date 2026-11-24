# Security policy

## Scope

This repository is an experimental crowd-monitoring application. It can process camera imagery, create local logs and recordings, and optionally send alert data to Twilio, SMTP, or a webhook.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the repository owner through the contact information on the [vincenzo-afk GitHub profile](https://github.com/vincenzo-afk). Do not publish exploit details, credentials, private URLs, or camera footage in a public issue.

Include the affected commit or branch, operating system, Python version, exact command, relevant logs with secrets removed, and a minimal reproduction when safe to provide. This repository does not promise a response time or a supported-version service level.

## Security expectations for deployments

Keep `.env`, model files, SQLite databases, logs, recordings, and generated reports outside public web roots and under appropriate filesystem permissions. The FastAPI server currently provides no authentication, authorization, TLS, or rate limiting, so do not expose it directly to the public internet. Use private networking and an access-controlled reverse proxy or gateway for any remote access.

The limited bounding-box blur in `src/detector.py` is not a complete anonymization or privacy guarantee. Validate privacy behavior against the actual camera views and retention requirements before processing real footage.

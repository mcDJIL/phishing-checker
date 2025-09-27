---
title: Phishing Detector
emoji: 🐨
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🚀 Phishing URL Detection API

This Space hosts a **FastAPI backend** for detecting phishing URLs.

### 🔹 Usage
Send a POST request to `/predict` with JSON body:

```json
{
  "url": "http://example.com/login?user=123"
}

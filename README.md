

<p align="center">
  <h1 align="center">📬 InboxPilot</h1>
  <p align="center"><strong>AI-Native Email Operations Environment</strong></p>
  <p align="center">
    An OpenEnv-compatible RL environment where AI agents learn to triage, classify, prioritize, and resolve professional email workflows — the way real operations teams do.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OpenEnv-Compatible-0969da?style=for-the-badge" alt="OpenEnv Compatible" />
  <img src="https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ed?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready" />
  <img src="https://img.shields.io/badge/HF_Spaces-Deployable-ffd21e?style=for-the-badge&logo=huggingface&logoColor=black" alt="HF Spaces" />
</p>

---

## Why InboxPilot?

Every organization runs on email. Support queues, investor updates, legal notices, HR complaints, security alerts — **professionals spend 28% of their workweek managing email** (McKinsey). Yet most AI benchmarks test agents on games and toy tasks.

**InboxPilot bridges that gap.** It gives AI agents a realistic operational inbox and asks them to do what a skilled operations professional would:

| Real-world skill | What the agent must do |
|---|---|
| **Triage** | Classify emails by type and urgency |
| **Prioritize** | Distinguish critical from routine |
| **Respond** | Draft safe, empathetic, contextual replies |
| **Route** | Escalate to the correct internal team |
| **Protect** | Detect phishing, avoid unsafe actions |
| **Follow through** | Schedule follow-ups for accountability |

> **This is not a chatbot.** This is a structured decision environment with deterministic grading.

---

## ⚡ Quick Start

```bash
git clone <repo-url> && cd inboxpilot-openenv
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 7860

# Run tests (42 tests, <1s)
pytest tests/ -v

# Run baseline inference
export OPENAI_API_KEY=sk-...
python inference.py
```

---

## 🏗️ Architecture

```
inboxpilot-openenv/
├── app/
│   ├── main.py              # FastAPI server (6 endpoints)
│   ├── env.py               # Core environment — reset / step / state
│   ├── models.py            # Typed Pydantic models
│   ├── rewards.py           # Shaped per-action reward engine
│   ├── graders.py           # Deterministic answer-key graders
│   ├── tasks.py             # JSON task loader
│   └── utils.py             # Text normalization utilities
├── data/tasks/              # 3 task definitions with answer keys
├── tests/                   # 42 tests (env, graders, API)
├── inference.py             # Baseline LLM agent (OpenAI SDK)
├── openenv.yaml             # OpenEnv metadata
├── Dockerfile               # HF Spaces–ready container
└── requirements.txt
```

---

## 🎯 Tasks & Difficulty Progression

InboxPilot includes **3 tasks** with deterministic, reproducible grading — no LLM judges.

### Task 1 · Support Inbox Triage `EASY`
> **6 emails** — refund request, duplicate charge, meeting reschedule, invoice request, login issue, phishing spam.  
> Agent must classify, prioritize, and route each correctly.

### Task 2 · Customer Resolution Workflow `MEDIUM`
> **1 complex case** — an angry customer charged twice with no response for 3 days.  
> Agent must classify → set high priority → draft empathetic reply → send → escalate to billing → schedule follow-up.

### Task 3 · Executive Inbox Risk Management `HARD`
> **7 high-stakes emails** — investor request, security breach alert, VIP complaint, legal notice, HR harassment complaint, phishing disguised as IT, media inquiry about layoffs.  
> Agent must avoid dangerous replies, route to specialized teams (legal, security, HR, communications), and handle ambiguity safely.

---

## 🕹️ Environment API

InboxPilot implements the standard OpenEnv interface:

```python
from app.env import InboxPilotEnv
from app.models import Action

env = InboxPilotEnv()

# Start a task
result = env.reset(task_id="easy_support_triage")
print(result.observation)  # inbox, goal, instructions

# Take actions
result = env.step(Action(action_type="open_email", email_id="email_001"))
result = env.step(Action(action_type="classify_email", email_id="email_001", payload={"category": "refund"}))
result = env.step(Action(action_type="set_priority", email_id="email_001", payload={"priority": "high"}))
result = env.step(Action(action_type="escalate", email_id="email_001", payload={"team": "billing"}))

# Check reward
print(result.reward.score, result.reward.message)

# Finish
result = env.step(Action(action_type="finish"))
print(result.info["grade"]["score"])  # final score ∈ [0.0, 1.0]
```

### Action Space (11 actions)

| Action | Payload | Purpose |
|--------|---------|---------|
| `open_email` | — | Read email contents |
| `classify_email` | `{"category": "..."}` | Assign category |
| `set_priority` | `{"priority": "low\|medium\|high\|critical"}` | Set urgency |
| `draft_reply` | `{"reply_text": "..."}` | Compose response |
| `send_reply` | — | Send drafted reply |
| `escalate` | `{"team": "..."}` | Route to internal team |
| `mark_spam` | — | Flag as spam/phishing |
| `archive` | — | Archive email |
| `schedule_followup` | — | Set follow-up reminder |
| `request_more_info` | — | Request clarification |
| `finish` | — | Signal task completion |

### Observation Space

Each observation gives the agent everything it needs:

```
task_id, goal, instruction, inbox_summary[], current_email,
available_actions[], action_history[], pending_items[],
step_count, max_steps
```

---

## 📈 Reward Design

Rewards are **shaped across the full trajectory** — not just binary at the end.

| Signal | Reward | Rationale |
|--------|--------|-----------|
| Correct classification | **+0.15** | Core triage skill |
| Correct priority | **+0.10** | Urgency awareness |
| Correct escalation target | **+0.15** | Routing accuracy |
| Quality draft (keyword match) | **+0.10** | Communication skill |
| Correct spam detection | **+0.15** | Security awareness |
| Correct follow-up | **+0.10** | Accountability |
| Good finish (≥80% handled) | **+0.15** | Completeness |
| Repeated/looping action | **−0.10** | Penalize waste |
| Invalid email reference | **−0.20** | Penalize errors |
| Wrong spam on legit email | **−0.15** | Penalize dangerous false positives |
| Premature finish | **−0.15** | Penalize incompleteness |
| Unknown action type | **−0.10** | Penalize invalid input |

---

## 🧪 Grading

All graders are **fully deterministic** — no LLM judges, no stochastic evaluation.

**Scoring formula:** `0.85 × correctness + 0.15 × efficiency`

**Correctness** checks per email:
- Classification accuracy (exact match)
- Priority correctness (exact match)
- Correct action taken (escalate / reply / spam / archive)
- Escalation target match (exact match)
- Reply keyword coverage (controlled keyword set)
- Follow-up scheduled (when required)
- Unsafe action detection (replied to phishing → penalty)

**Efficiency** rewards agents that use fewer steps.

---

## 🌐 HTTP API

The FastAPI server exposes 6 endpoints compatible with HF Spaces and `openenv validate`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset environment (accepts `{}` or `{"task_id": "..."}`) |
| `POST` | `/step` | Execute an action |
| `GET` | `/state` | Current internal state |
| `GET` | `/tasks` | List available tasks |
| `GET` | `/health` | Health check → `{"status": "ok"}` |
| `GET` | `/` | Landing page |

```bash
# Reset (validator-compatible — returns 200 on empty body)
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'

# Step
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d '{"action_type": "open_email", "email_id": "email_001", "payload": {}}'

# Health
curl http://localhost:7860/health
```

---

## 🐳 Docker & Deployment

```bash
# Build and run locally
docker build -t inboxpilot .
docker run -p 7860:7860 inboxpilot
```

### Hugging Face Spaces

1. Create a new Space → select **Docker** SDK
2. Push this repository to the Space
3. The container exposes port `7860` (HF Spaces default)
4. Verify deployment:

```bash
curl -X POST https://your-space.hf.space/reset \
  -H "Content-Type: application/json" -d '{}'
# → HTTP 200 ✓
```

---

## 🤖 Baseline Inference

`inference.py` runs all 3 tasks via the OpenAI SDK with `temperature=0`:

```bash
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini        # optional
export API_BASE_URL=https://api.openai.com/v1  # optional
python inference.py
```

**Expected baseline scores** (`gpt-4o-mini`, `temperature=0`):

```
══════════════════════════════════════════════════════════
  BASELINE RESULTS SUMMARY
══════════════════════════════════════════════════════════
  easy_support_triage                      ~0.85
  medium_customer_resolution               ~0.80
  hard_exec_risk_management                ~0.70
  AVERAGE                                  ~0.78
══════════════════════════════════════════════════════════
```

Features: robust JSON parsing with nested payload support, safe fallback actions, per-task error isolation, max step cap.

---

## ✅ Validation

```bash
# OpenEnv validation
openenv validate

# Test suite (42 tests)
pytest tests/ -v

# Docker build check
docker build .
```

| Check | Status |
|-------|--------|
| `POST /reset {}` → HTTP 200 | ✅ |
| `Dockerfile` builds from root | ✅ |
| `openenv validate` passes | ✅ |
| `inference.py` runs without crash | ✅ |
| All graders deterministic | ✅ |
| Scores clamped to `[0.0, 1.0]` | ✅ |
| Runs on 2 vCPU / 8 GB RAM | ✅ |

---

## 🔮 Future Directions

- Multi-language inbox support
- SLA deadline modeling with time-pressure mechanics
- Multi-agent delegation and handoff
- Curriculum learning with progressive difficulty
- Richer email types: attachments, threads, calendar invites
- Custom rubric builder for enterprise-specific workflows

---

<p align="center">
  <strong>Built for the Meta × Scaler Hackathon</strong> 🚀
  <br/>
  <sub>Real-world AI evaluation for real-world workflows.</sub>
</p>

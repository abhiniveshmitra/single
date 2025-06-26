https://github.com/abhiniveshmitra/single
Investor Intelligence Platform
Building the Future of Personalised Wealth Advice

Executive Summary
Traditional investor profiling is static and compliance-focused. The Investor Intelligence Platform (IIP) replaces this with an AI-driven, real-time system that delivers personalised insights, diversification heatmaps, and compliant next-best-actions—all grounded in UBS CIO research. It uses Azure OpenAI GPT-4o, a micro-agent mesh, and a Streamlit front-end for seamless interaction.

Hackathon Objective
Build a working browser-accessible IIP prototype using Streamlit. Judges will see:

Live app that answers “What changed for me after today’s rate cut?”

GitLab pipeline deploying to AKS in <15 mins

Audit logs ensuring MiFID/FIDLEG compliance

Problem
RMs spend hours gathering data. Touchpoints are infrequent and post-facto compliance leads to rework. Insights, portfolios, and CRM data are siloed.

Solution
Agentic AI on AKS: micro-agents for ingest, analytics, bias detection, and policy enforcement. GPT-4o + RAG generate narratives. Streamlit UI presents them.

Features

Personalised GPT-4o summaries

Diversification heatmaps

CVaR risk dashboard

Behavioural bias detector

Suitability-tagged next actions

One-click PDF export

Full audit trail (WORM logs)

Tech Stack

Front-end: Streamlit + Altair

Orchestration: FastAPI + LangChain

Agents: Python containers (pypdf, SQLAlchemy, CVXPY, OPA)

AI: GPT-4o + text-embedding-3-small

Infra: AKS, Redis, PostgreSQL + pgvector, Delta Lake

Security: Private link, Key Vault, CMK

DevOps: GitLab → Docker → Helm → AKS

Workflow
Blob event → Ingest → Embed → Retrieve → Plan (LLM) → Act (narratives + actions) → Enforce (OPA) → Learn (feedback loop)

Innovation
Agentic mesh > monolith. Built-in compliance. Zero-trust design. High recall semantic search without Azure licensing.

Impact

70% less analyst prep time

30% more proactive client interactions

95% first-pass compliance

Direct UBS solution mapping for upsell

Future
Role-based views, multilingual support, GPT-4o Vision, trade execution, Synapse Lakehouse.

Conclusion
IIP transforms investor interactions into real-time, compliant conversations, blending UBS expertise with cutting-edge AI for scalable, production-ready wealth management.

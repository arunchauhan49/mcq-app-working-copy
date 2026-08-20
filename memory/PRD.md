# PRD — Quizr (AI-powered MCQ Study App)

## Original Problem Statement
Create an AI-powered MCQ app that analyzes uploaded notes/chapters (PDF, images, text) and generates MCQs. Practice one question at a time with immediate correct/incorrect feedback + explanation. Mixed MCQs across chapters. MCQ History (auto-saved, filter by chapter). My Progress dashboard. AI must primarily use uploaded notes as source.

## User Personas
- Student preparing for exams from personal notes
- Self-learner uploading study material and drilling MCQs

## Core Requirements (Static)
- Email/password auth (JWT)
- Upload notes as PDF / image / text (AI extraction via Gemini)
- Generate 5/10/15/20 MCQs per chapter (user chooses)
- Mixed MCQ session across selected chapters
- MCQ Practice: single question view + immediate feedback + explanation
- MCQ History: auto-saved, filter by chapter and by session type (chapter/mixed)
- Progress: accuracy per chapter + overall stats
- AI grounded strictly in uploaded notes (no outside knowledge by default)

## Architecture
- Backend: FastAPI + MongoDB (motor) + Emergent Universal LLM Key + Gemini 2.5 Flash
- Frontend: React 19 + Shadcn UI + Tailwind + Recharts
- Auth: JWT (HS256), bcrypt password hashing

## What's Been Implemented (2026-02)
- Full auth flow (register / login / me)
- Notes upload (PDF / image / .txt via Gemini extraction; direct text paste)
- MCQ generation from single note and mixed across chapters
- Single-question practice UI with feedback + explanation
- History with chapter + type filters
- Progress dashboard with per-chapter accuracy chart
- Neo-brutalist Cabinet/Outfit + IBM Plex Sans design system

## Backlog / Next Actions (P1/P2)
- Timed practice / exam mode
- Bookmark / star tricky MCQs
- Export MCQs to PDF for offline revision
- Shareable session links with friends
- Streaks & daily goals

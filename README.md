Scenario. You have joined the AI team at Northwind Lending, a fictional consumer-finance company. The Head of Data wants a working proof of concept by the end of the day, covering the main families of AI technique the company plans to use:

A goal-based agent that plans routes for a warehouse robot (classical AI search).
A supervised learning model that predicts whether a loan applicant will default.
An NLP classifier that routes customer-support tickets to the right team.
An unsupervised customer-segmentation model for marketing.
A responsible-AI review of the loan model (fairness, explainability, model card).
A containerised REST API that serves the approved loan model.
All data is generated synthetically by the lab with fixed random seeds, so every run is reproducible. Everything runs on a single CPU-only Ubuntu 24.04 server through the terminal. No GPU is needed. Numeric results shown below are approximate unless stated as exact; compare the direction and the ranges, not the last digit.

What you will build

~/ai-lab/
├── data/        synthetic loans, tickets, customers
├── src/         search agent, training, evaluation, fairness scripts
├── models/      trained pipelines (joblib) + metadata
├── reports/     JSON metrics, model card
├── service/     FastAPI app + Dockerfile
└── validate.sh  final automated checks

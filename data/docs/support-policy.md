# Orbitly Cloud Support Policy

Orbitly Cloud is a fictional managed Kubernetes and database platform. This policy
describes how customers get technical help.

## Support plans

| Plan | Channels | Hours |
|---|---|---|
| Basic | Community forum, email | Best effort |
| Pro | Email, web portal | Monday–Friday, 08:00–20:00 CET |
| Enterprise | Email, web portal, phone, dedicated Slack channel | 24/7 for P1, business hours otherwise |

## Ticket priorities and first response times

Priority is set by the customer when opening a ticket and may be adjusted by the
support engineer after triage.

- **P1 — Critical**: production is down or data loss is occurring. First response:
  Enterprise 15 minutes, Pro 1 hour. Basic plan customers cannot open P1 tickets.
- **P2 — High**: a production feature is degraded with no workaround. First response:
  Enterprise 1 hour, Pro 4 business hours.
- **P3 — Normal**: non-production issue or a workaround exists. First response:
  Enterprise 4 business hours, Pro 1 business day, Basic 3 business days.
- **P4 — Low**: questions, feature requests. First response: 2 business days for all plans.

## Escalation

If a P1 or P2 ticket is not progressing, Enterprise customers can escalate through the
phone line to the duty manager. Pro customers escalate by replying "ESCALATE" in the
ticket, which notifies the support team lead. Escalations are reviewed within 30 minutes
during support hours.

## What support does not cover

Support does not cover debugging of customer application code, custom Helm charts that
are not based on the official Orbitly templates, or third-party tools running inside
customer clusters.

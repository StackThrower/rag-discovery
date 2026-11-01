# Incident Response Runbook

Internal runbook for Orbitly on-call engineers.

## Severity levels

- **SEV1** — multiple customers affected, a region or core service is unavailable.
- **SEV2** — a single customer or a non-core service is significantly degraded.
- **SEV3** — minor degradation, no customer impact yet.

## SEV1 procedure

1. The on-call engineer acknowledges the page within 5 minutes and opens an incident
   channel named `inc-<date>-<short-name>`.
2. An incident commander (IC) is appointed; the IC does not debug, the IC coordinates.
3. The status page is updated within 15 minutes of detection and then every 30 minutes.
4. Enterprise customers in the affected region are notified by their account manager.
5. After resolution a blameless postmortem is published within 5 business days.

## Common error codes

- `ORB-503` — the regional control plane is failing over; deployments are queued and
  retried automatically. Do not restart the scheduler during a failover.
- `ORB-429` — the customer exceeded the API rate limit of 600 requests per minute.
- `ORB-507` — managed storage quota is exhausted; the customer must expand storage or
  delete data.

## Rolling back a release

Use `orbctl deploy rollback --to <release-id>` from the bastion host. Rollbacks of the
control plane require approval from the IC. Database schema migrations are never rolled
back automatically; write a forward fix instead.

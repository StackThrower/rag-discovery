# Service Level Agreement (SLA)

## Uptime commitment

Orbitly commits to a monthly uptime of **99.9%** for the Pro plan and **99.95%** for the
Enterprise plan. The Basic plan has no uptime commitment.

Uptime is measured per region as the percentage of minutes in a calendar month during
which the Kubernetes API and managed database endpoints respond successfully.

## Service credits

If monthly uptime falls below the commitment, the customer receives a credit on the
next invoice, calculated as a share of the monthly fee for the affected service:

| Monthly uptime | Credit |
|---|---|
| below the commitment but at least 99.0% | 10% |
| below 99.0% but at least 95.0% | 25% |
| below 95.0% | 50% |

Credits must be requested through the support portal within **30 days** after the end of
the affected month. Credits are the sole remedy for SLA breaches and cannot be paid out
as cash.

## Exclusions

Scheduled maintenance is excluded from uptime calculations. Maintenance windows are on
Sundays between 02:00 and 04:00 CET and are announced at least 72 hours in advance on
the status page. Outages caused by customer configuration, force majeure, or
suspension for non-payment are also excluded.

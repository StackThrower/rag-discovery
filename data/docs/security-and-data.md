# Security and Data Protection

## Encryption

All customer data is encrypted at rest with AES-256 using keys managed in a hardware
security module. Data in transit is protected with TLS 1.3; TLS 1.2 is accepted only for
legacy client libraries and will be disabled in 2027. Enterprise customers can bring
their own keys (BYOK).

## Data residency

Customers choose a region when creating a project: Frankfurt (eu-central-1),
Warsaw (eu-east-1) or Dublin (eu-west-1). Data never leaves the selected region,
except encrypted backups, which are replicated to a second region in the EU.

## Backups

Managed databases are backed up every 6 hours. Backups are retained for 30 days on
Pro and 90 days on Enterprise. Point-in-time recovery is available for the last 7 days
on Enterprise.

## Account deletion and data retention

When an account is closed, customer data is deleted from production systems within
**35 days**. Backups containing the data expire according to the normal backup
retention schedule. Billing records are kept for 10 years to meet tax obligations.

## Access control

Two-factor authentication is mandatory for all organization administrators.
Single sign-on via SAML 2.0 and SCIM provisioning are available on the Enterprise plan only.
Orbitly staff can access customer environments only through a time-limited,
audited break-glass procedure approved by the customer.

## Certifications and testing

Orbitly holds a SOC 2 Type II report and ISO/IEC 27001 certification. An independent
penetration test is performed every year; a summary is available under NDA.

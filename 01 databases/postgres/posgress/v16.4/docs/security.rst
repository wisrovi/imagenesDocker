Security
========

Security considerations and best practices.

Passwords
---------

- Use strong, unique passwords
- Store in .env file (not committed)
- Consider Docker secrets for production

Network Security
----------------

- Services communicate via internal network
- Exposed ports are documented
- Use firewalls to restrict access

Database Security
-----------------

- Limit user privileges
- Use SSL/TLS for connections (future enhancement)
- Regular security updates

Container Security
------------------

- Run as non-root user where possible
- Keep images updated
- Scan for vulnerabilities: ``./scripts/scan.sh``

Access Control
--------------

- pgAdmin requires authentication
- Grafana has admin user
- API access via PostgREST roles
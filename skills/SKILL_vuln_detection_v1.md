# Vulnerability Detection Skill — v1

## Objective

Systematically analyse source code to detect security vulnerabilities.
The analysis must produce a structured JSON report indicating whether the
code is vulnerable, the type of weakness (CWE), an explanation, a potential
exploit scenario, and a suggested patch.

## Analysis Procedure

Follow every step below **in order** before producing the final JSON output.

### Step 1 — Context Identification

- Identify the programming language and any frameworks or libraries used.
- Identify entry points: functions that accept external or user-controlled input.
- Note imported modules, system calls, and security-relevant APIs.

### Step 2 — Data-Flow Analysis

- Trace all untrusted data from its **source** (user input, network, file,
  environment variable, command-line argument) through transformations to
  **sinks** (SQL queries, shell commands, file writes, memory allocations,
  output rendering).
- Flag any path where sanitisation, validation, or encoding is absent or
  insufficient.

### Step 3 — Memory Safety (C/C++ specific)

- Check for buffer overflows (stack and heap).
- Check for use-after-free and double-free.
- Check for integer overflows in size or length calculations.
- Check for format-string vulnerabilities.
- Check for out-of-bounds reads and writes.
- Verify that dynamically allocated memory is properly freed on all code paths.

### Step 4 — Input Validation

- Verify that all external inputs are validated for type, length, range, and
  format **before** use.
- Check for missing NULL-pointer checks after allocations or function returns.
- Identify any assumptions about input size or structure that are not enforced.

### Step 5 — Injection and Command Composition

- Check all sinks that compose dynamic queries or commands (SQL, OS shell,
  LDAP, XPath, regular expressions).
- Verify use of parameterised queries, prepared statements, or proper
  escaping / quoting.

### Step 6 — Authentication and Access Control

- Verify that authentication state is checked before privileged operations.
- Look for missing authorisation checks, insecure session handling,
  hard-coded credentials, and privilege-escalation paths.

### Step 7 — Cryptographic Misuse

- Flag weak algorithms (MD5, SHA-1 for security purposes, DES, ECB mode).
- Flag insecure random-number generators used for security tokens.
- Flag hard-coded keys, IVs, or salts.

### Step 8 — Error Handling and Resource Management

- Check for sensitive data exposed in error messages or logs.
- Check for exceptions swallowed silently that may hide security failures.
- Verify proper resource cleanup (file handles, sockets, locks) on all
  exit paths, including error paths.

### Step 9 — Control-Flow and Logic Errors

- Check for race conditions (TOCTOU, shared state without synchronisation).
- Check for unreachable security checks or dead-code paths that bypass
  validation.
- Verify that loops and recursive calls have proper termination conditions.

### Step 10 — Verdict and Report

After completing steps 1–9, determine whether the code contains a security
vulnerability.

- If **vulnerable**: identify the root cause, map it to a CWE identifier,
  describe a concrete exploitation scenario, and propose a minimal fix.
- If **safe**: state that no vulnerabilities were detected.

## Expected Output Format

Respond **strictly** with a valid JSON object and **no additional text**.

```json
{
  "vulnerable": true or false,
  "cwe_id": "CWE-XXX",
  "vulnerable_code": "relevant code snippet",
  "explanation": "brief explanation of the vulnerability",
  "exploit": "how the vulnerability could be exploited",
  "patch": "suggested fix"
}
```

Only include fields relevant to the prediction.
If the code is not vulnerable, return:

```json
{
  "vulnerable": false
}
```

## Guidelines for Reducing Errors

### Reducing False Positives
- Do **not** flag code as vulnerable unless you can describe a concrete
  exploitation scenario.
- Standard defensive patterns (bounds checks before use, `sizeof`-based
  allocations, parameterised queries) are **not** vulnerabilities.
- Consider the scope: a variable that appears unsafe in isolation may be
  validated by its caller.

### Reducing False Negatives
- Pay special attention to edge cases: off-by-one errors, signed/unsigned
  comparisons, implicit type conversions.
- Do not assume that helper functions perform proper validation unless you
  can see their implementation.
- Check all code paths, including error-handling and early-return paths.

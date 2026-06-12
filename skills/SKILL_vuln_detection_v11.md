Output one JSON object and nothing else. The response must start with `{` and end with `}`. Do not output any prose, analysis text, code blocks, markdown, or tags before or after the JSON. The first character you write must be `{`. Do not begin your response with the word "The" or any word other than "{". Do not wrap your response in backticks, code fences (```), or any XML/HTML tags.

Schema: `{"vulnerable": true|false, "cwe_id": "CWE-XXX", "vulnerable_code": "<exact verbatim snippet from the input code>", "explanation": "...", "exploit": "...", "patch": "..."}`

For a safe verdict: `{"vulnerable": false, "cwe_id": "", "vulnerable_code": "", "explanation": "<two named operations from the code and their specific guards>", "exploit": "", "patch": ""}`

---

You are a security auditor. Analyse the provided source code and determine whether it contains a security vulnerability. Your default assumption is that the code is vulnerable until proven safe.

**Recall priority**: Missing a real vulnerability is worse than a false alarm. When evidence is ambiguous, classify as vulnerable.

**Step 1.** Name every variable or value that carries data from outside the program: user input, network data, file reads, environment variables, command-line arguments, IPC messages. Write the actual variable names from the code.

**Step 2.** Name every dangerous operation in the code: memory copy/move (memcpy, memmove, strcpy, strncpy, sprintf, snprintf, read, write), allocation/deallocation (malloc, calloc, realloc, free, new, delete), SQL or shell string composition, pointer arithmetic, array access with a computed index, format string arguments, file or path operations. Write the actual call or expression.

**Step 3.** For every operation from Step 2, check:

- **Bounds**: Is the size or index provably limited to a safe value? If the limit comes from untrusted input, it is not safe. Covers CWE-119, CWE-125, CWE-787.
  - Off-by-one: loop bound `<= len` instead of `< len`, or copy size `len + 1` with no extra allocation byte.
  - Multi-step offset: a running index validated once at entry but incremented further without re-check.
  - Size from input: allocation or copy size derived directly or indirectly from an attacker-controlled field, even after arithmetic.
- **Lifetime**: Is memory used after free? Is there a double-free path? Is a pointer from a freed region returned or dereferenced? Covers CWE-416, CWE-415.
- **Integer safety**: Can a size or index overflow or wrap? Check signed/unsigned conversions, multiplications without overflow check, additions involving casts. Covers CWE-190, CWE-189.
- **NULL safety**: Is every pointer from malloc/calloc/realloc/fopen/strdup checked for NULL before use? Covers CWE-476.
- **Injection**: Is untrusted data composed into a SQL query, shell command, or file path without parameterisation or strict whitelist? Covers CWE-89, CWE-78, CWE-22.
- **TOCTOU**: Is a file or resource checked (stat, access) and then used separately, allowing it to change between check and use? Covers CWE-367.
- **Logic/authorization**: Is a permission check missing, bypassable, or applied after the sensitive operation? Is an assertion (assert()) reachable with attacker-controlled input? Is input accepted without validation of type, range, or format? Is a resource limit or counter enforced on all code paths? Covers CWE-264, CWE-284, CWE-617, CWE-400, CWE-20.

If any check fails for any operation, the code is vulnerable.

**Step 4.** Verdict. If Step 3 found a failure: output `"vulnerable": true` with the CWE identifier, the exact snippet from the actual input code where the flaw occurs (verbatim — do not substitute template code), a concrete exploit scenario, and a minimal patch.

If Step 3 found no failure: you may output `"vulnerable": false`, but your explanation must name at least two specific operations from Step 2 and state the exact guard condition that bounds each one. Generic statements ("no untrusted input flows to unchecked operations", "all inputs are validated", "no dangerous operations found") are not acceptable — if you cannot name the specific operations and guards, re-examine the code and classify as vulnerable.

**Precision rule.** Flag as vulnerable only when you can point to a specific expression in the actual code and describe a concrete attacker-controlled input that triggers the flaw. Do not flag code solely because a function category appears (malloc, memcpy, free). The `vulnerable_code` field must contain text copied verbatim from the input. Do not write `memcpy(buf, src, len)` or `memcpy(buf,src,len)` unless those exact characters appear in the input code.

Your entire response is one JSON object starting with `{` and ending with `}` — no prose before or after.

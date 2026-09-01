# Runtime architecture

`POST /wio/query` resolves a supplied or unambiguous in-text location, deterministically normalizes time, creates a `RetrievalPlan`, retrieves independent sources concurrently, converts source records to CEOs, applies the temporal and semantic gates, persists evidence in the process index, builds a WIO, runs structured agents, and rejects the result when the reviewer finds an invalid evidence ID.

`POST /decision` runs the same path then invokes RADE v2. RADE uses member values only when present; otherwise it uses the source precipitation probability and amount as two explicit scenarios. If those are absent it returns `defer_decision`.

Weather data, decision mathematics, and response language remain separate. The current response synthesis is deterministic; Groq is an optional client utility and is not needed for liveness or readiness.

User context and feedback are user-ID scoped SQLite records. Context is retrieved only for the requesting `user_id`; no endpoint enumerates another user’s data.

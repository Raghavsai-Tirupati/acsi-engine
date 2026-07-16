# TypeScript Capture Snippet

The TypeScript wrapper is documentation-only in v1. It must be fail-open: wrap all ACSI capture
writes in `try/catch`, enqueue asynchronously, and drop events when the local queue is full. Never
raise capture errors into production request handling.

```ts
type AcsiTrace = Record<string, unknown>;

export function captureAcsiTrace(trace: AcsiTrace): void {
  try {
    void enqueueTrace(trace).catch(() => undefined);
  } catch {
    // Fail open: ACSI capture must never break the host application.
  }
}
```


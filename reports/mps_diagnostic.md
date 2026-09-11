# MPS diagnostic

Captured 2026-09-11 on the Apple M2 Max host. This diagnostic did not execute
backward propagation, construct an optimizer, or perform an optimizer update.
The only full-model execution was the existing inference-mode
`--forward-only` path.

## Disposition

MPS is healthy in a normal host-visible process. It is invisible only to the
restricted command sandbox used for the earlier checks. A controlled test used
the same project venv, Python executable, arm64 process, and PyTorch 2.8.0 in
both contexts:

| Context | MPS built | MPS available | Result |
| --- | ---: | ---: | --- |
| Restricted command sandbox | `True` | `False` | Creating an MPS tensor raises PyTorch's generic macOS-version availability error. |
| Host-visible process | `True` | `True` | A tensor operation completed on `mps:0`. |

This isolates the failure to the sandbox's Metal/GPU device visibility, not the
Python architecture, venv, PyTorch wheel, or project device-selection logic.
The generic sandbox-side exception mentions macOS 13.0+, but the host is macOS
26.5 and the same interpreter works outside the sandbox, so that message is not
an accurate OS diagnosis.

No portability fix is required. `src/utils/device.py` correctly selects MPS
when both build support and process availability are true, and otherwise makes
the CPU fallback explicit.

## Host MPS probe

The host-visible process reported:

- executable: project `.venv/bin/python`
- process architecture: `arm64`
- PyTorch: `2.8.0`
- test: `(arange(8).square() + 1).sum()` on `mps:0`
- result: `148.0`
- elapsed time, including first-use initialization: `0.366502 s`
- current allocated memory after the test: `512` bytes
- driver allocated memory after the test: `25,591,808` bytes (24.41 MiB)

## SILMA forward-only result

The repository's existing `src.training.train --forward-only` path ran in the
host-visible process with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Unsupported-operation
fallback was therefore disabled, and the run completed without such an error:

| Measurement | Result |
| --- | ---: |
| Selected device | `mps` |
| Loaded parameters | `162,666,852` |
| Batch mel shape | `[1, 100, 201]` |
| Batch mel length | `201` |
| Forward loss | `0.63564199` (finite) |
| End-to-end elapsed time | `12.458913 s` |
| Peak sampled MPS current allocation | `665,816,320` bytes (634.97 MiB) |
| Peak sampled MPS driver allocation | `1,181,499,392` bytes (1.100 GiB) |
| End current MPS allocation | `0` bytes |
| End driver allocation | `1,181,499,392` bytes (1.100 GiB) |
| Peak process RSS | `1,761,984,512` bytes (1.641 GiB) |

The elapsed time covers manifest checks, batch preparation, strict model load,
and the no-grad forward; it is not a forward-kernel benchmark. MPS allocator
values were sampled every 20 ms, so the reported peaks are observed samples,
not guaranteed instantaneous maxima. Driver allocation can remain cached after
live tensor allocation returns to zero.

The prior authorized CPU training-path check also produced a finite loss
(`0.568487286567688`). That establishes basic CPU/MPS sanity only. The values
are not a numerical-parity comparison because the CPU evidence used a training
step with a different row-selection limit, model mode, and random seed, while
this MPS evidence used the dry-run inference-mode path. No exact-parity claim is
made.

## Reproduction boundary

Run the MPS check from a normal local macOS process, not the restricted command
sandbox:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m src.training.train --forward-only
```

Expected evidence includes `device=mps`, `loaded_parameters=162666852`, and a
finite `forward_loss`. This command is no-grad and must not be replaced with
`--smoke-one-step` or `--train` when only forward validation is authorized.

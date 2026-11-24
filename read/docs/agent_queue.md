# Answering requests with agents instead of an API

PAVE Read normally sends each request to an OpenAI-compatible API (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`). It can instead put each request in a file queue, where coding agents answer it. Examples are Claude Code subagents and Codex subagents. Each request goes to one fresh agent, which answers it and stops. No agent sees more than one request, so no answer depends on an earlier one.

## Run the pipeline with the queue

```bash
export PAVE_VLM_BACKEND=bridge
export PAVE_VLM_BRIDGE_DIR=runs/queue_my_site        # the queue folder
export PAVE_BRIDGE_MODEL=claude-opus-5-5             # the model that answers, recorded with every answer
export PAVE_BRIDGE_PROVIDER=claude_code_subagent     # who answers, e.g. codex_subagent
python scripts/pipeline/run_read.py --imagery runs/imagery_my_site --output runs/read_my_site
```

The pipeline writes each request to the queue and waits for its answer. It also caches every answer under the recorded model and provider. A run that is stopped and started again resumes from the cache. An answer from one model or provider is never reused for another.

| Setting | Meaning | Default |
|---|---|---|
| `PAVE_VLM_BACKEND` | `bridge` for the queue, `api` for the API | `api` |
| `PAVE_VLM_BRIDGE_DIR` | the queue folder | `<runs root>/vlm_bridge` |
| `PAVE_BRIDGE_MODEL` | the model that answers | `claude-opus-5-5` |
| `PAVE_BRIDGE_PROVIDER` | who answers | `claude_code_subagent` |
| `PAVE_BRIDGE_TIMEOUT_S` | how long a request waits for its answer | 4 hours |
| `PAVE_BRIDGE_REUSE_IDENTICAL` | `0` sends a request to the agents even when the API model already answered the same image and prompt | `1` |
| `PAVE_PARALLEL_REQUESTS` | requests in flight at once | 1 |

## The queue tool

```bash
python scripts/pipeline/vlm_bridge.py status                     # pending, claimed, answered
python scripts/pipeline/vlm_bridge.py claim --max 1 --worker w1_001 --wait 900
python scripts/pipeline/vlm_bridge.py answer --id <id> --file <answer.json> --worker w1_001
python scripts/pipeline/vlm_bridge.py release --older-than-min 30   # requeue requests claimed by agents that stopped
```

Each request has two files: `pending/<id>.json` holds its metadata, and `prompts/<id>.txt` holds the system prompt, the prompt and the path of the image. `claim` moves a request to `claimed/`. `answer` checks that the answer is JSON and writes it to `answers/`, where the pipeline picks it up.

## Starting the agents

Start one agent per pending request. Give each a name of its own and this instruction:

> You are a one-shot vision worker. NAME = <name>, QUEUE = <queue folder>. Read the section "Worker instructions" in `read/docs/agent_queue.md` and follow it exactly (replace NAME and QUEUE with yours).

The first requests of a site are the traces of its arms and the blind readings of its approaches. Most later requests depend on earlier answers, so check `status` after each agent finishes and start another agent for each new pending request.

## Worker instructions

You were given a worker NAME (e.g. `w1_005`) and a QUEUE (the queue folder). Below, replace NAME and QUEUE with yours.

You answer exactly ONE vision request from a file queue, then you stop. The request is one image plus a prompt that asks for a JSON answer about road lanes, markings or signals.

Work from the repository's `read/` folder (always `cd` there first). Every queue command sets the queue folder:

```bash
PAVE_VLM_BRIDGE_DIR=QUEUE python scripts/pipeline/vlm_bridge.py ...
```

Steps:

1. Claim one request: `... vlm_bridge.py claim --max 1 --worker NAME --wait 900`. The output line gives the request id, its prompt_file and its image. If nothing comes back, reply "none" and stop.
2. Read the whole prompt_file (system prompt, prompt, image path, answer command).
3. Look at the image. Large images may be shown scaled down. When a detail matters (lane lines, arrows, labels, ruler ticks), crop that region with Python and PIL. Save crops only in `QUEUE/work/NAME/`, then look at the crop. When the prompt asks where thin lines lie (lane lines, stop bars, curbs), crop the region around them at native resolution, and check each line in the crop before you answer.
4. Coordinates: use exactly the unit and frame the prompt asks for. If the image has rulers, the ruler ticks are the coordinate system. The ticks may be offset from the image's pixel origin, and the image may be shown enlarged. When you crop, add the crop offset back and check against the nearest tick. Where the prompt asks for boxes as fractions (e.g. `bbox_xyxy` like `[0,0,1,1]`), give fractions of the named panel from 0 to 1, never pixels.
5. Write exactly the JSON the prompt asks for, with every required field, to a file in your work folder. Then record it: `... vlm_bridge.py answer --id <id> --file <file> --worker NAME`. If the command reports invalid JSON or a validation problem, fix the file and run the command again.
6. Stop. Do not claim another request.

Rules:

- Open only the paths the claim command prints (prompt_file, image). Never guess a path.
- Use only these instructions, the prompt and the image (and your crops of it). Do not open any other file: no run outputs, caches, other requests or answers, reference data, the code, or the web.
- Do not invent what you cannot see. Use the confidence, limitations and unresolved fields the prompt provides.
- Do not modify any file other than your own files in your work folder (and the effects of the answer command). Never read or print the `.env` file.

Reply with one line: the request id you answered (or "none"), and anything that went wrong.

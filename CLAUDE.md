# CLAUDE.md

## Coding Agent

- In your comment replies, avoid using #<numeral> style info, such as "#1", unless you're specifically referring to an issue or pull request, since this auto-formats as an issue or PR link. Instead, state "No. 1" or "number 1" etc.
- The default branch of this repo (AURORA) is `main` (verified via `git remote show origin`).
- Be cautious with `~` ("approximately") since if you use it too many times it starts causing strikeout formatting in markdown previews
- Include plots directly in your comment reply via `![image name](https://github.com/<user/org>/<repo>/blob/<shortened-commit-hash>/<filename>?raw=true)`. Truncate the commit hash to the first 7 characters only. For provenance, ensure you use the shortened (7-character) commit hash, not the branch name
- If you mention files in your comment reply, add direct hyperlinks based on the shortened (7-character) commit hash
- IMPORTANT: Never echo/grep/print environment secrets. These should never be exposed in your terminal history or other outputs

## Asta (Allen Institute for AI)

The Asta plugins (Theorizer, AutoDiscovery, and related tools) are installed via the Claude Code plugin system, configured in `.github/workflows/claude.yml`:

- Marketplace: `https://github.com/allenai/asta-plugins.git`
- Plugins: `asta-tools@asta-plugins` and `asta-flows@asta-plugins`

Authentication is provided via the `ASTA_TOKEN` GitHub Actions secret, exposed as the `ASTA_TOKEN` environment variable. Do not echo, grep, or otherwise surface this token. If Asta calls fail with authentication errors, the token has likely expired; report this in your comment reply so the operator can regenerate it locally (`asta auth login`, then `asta auth print-token --raw --refresh`) and update the repo secret. Asta beta credits are tied to the operator's account, so use the tools purposefully rather than for trivial lookups.

Use Asta tools when the task involves literature-grounded hypothesis generation (Theorizer) or automated discovery workflows (AutoDiscovery), such as proposal background research, identifying prior work, or generating candidate research directions.

## Edison Scientific

When waiting on an Edison task in GitHub Actions, NEVER run the polling script in the background (run_in_background, nohup, &) — the runner is destroyed the moment you post your final comment, killing background processes. Poll in the FOREGROUND: either run the blocking fetch script as a single Bash call with an explicit long timeout, or loop on short calls (sleep 240 + fetch --once, each under the Bash timeout). If you loop on short calls, you can take care of intermediate tasks while you wait. Do not post your final comment until results are fetched and committed, or ~45 minutes of wall-clock have elapsed — in which case commit the task-id file and state that a follow-up @claude comment is needed to fetch. If you need to upload files, use analysis query type. See the docs: https://edisonscientific.gitbook.io/edison-cookbook/edison-client. Here is the endpoint you should use: https://api.platform.edisonscientific.com. The API key is `EDISON_PLATFORM_API_KEY`. Don't expose this secret, e.g., by echoing or grepping it. Pass the API key in explicitly:

```
from edison_client import EdisonClient, JobNames
client = EdisonClient(api_key=EDISON_PLATFORM_API_KEY)
```

Whenever you retrieve results (either during the current agent session or during the next session), make sure to fetch and commit all artifacts associated with a trajectory.

If using Edison Analysis, refer to https://docs.edisonscientific.com/edison-client/file-management#upload for instructions on how to upload files. If able to use Context7, to better inform use of EdisonClient, see https://context7.com/future-house/edison-client-docs/llms.txt?tokens=10000

## LaTeX

Install MiKTeX instead of TeXLive to reduce download size and time. In the first installation of MiKTeX, download known required packages based on the LaTeX file itself, and install anything else ad-hoc as needed.

## University of Utah CHPC Access

The University of Utah Center for High Performance Computing (CHPC) credentials are provided to agent runs via GitHub Actions secrets, exposed as the `CHPC_USERNAME` and `CHPC_PASSWORD` environment variables.

<!-- TODO: This section was adapted from a BYU ORC workflow. Verify every
     detail below against the repo where CHPC access already works, and copy
     that repo's hpc/ helper scripts (agent_connect.py, the file-based 2FA
     wrapper, ssh_config.example, run_remote.sh, etc.) into this repo.
     None of those scripts exist here yet; the agent cannot connect without them. -->

- Do **not** echo, `cat`, `grep`, `printenv`, `env | grep`, write to a file, log, or otherwise surface the value of `$CHPC_PASSWORD` (or any variable derived from it) anywhere — not in shell output, not in commit messages, not in PR descriptions, not in steering replies, not in source files. Treat it like a secret key.
- Do **not** hard-code the password in scripts, tests, fixtures, or `.env` files.
- Pass it to tooling implicitly via the environment. The `CredentialProvider` in `hpc/agent_connect.py` reads `$CHPC_PASSWORD` and feeds it directly into the SSH password prompt without echoing it. Use that path; do **not** read the variable yourself just to pipe it elsewhere.
- **Never** invoke `agent_connect.py` (or the wrapper) with `log_file=sys.stderr`. `pexpect`'s `logfile_read` would echo the password we send back through the transcript and leak `$CHPC_PASSWORD` into shell output. If you absolutely need an SSH transcript for debugging, write to a file under `/tmp/` and shred it immediately.
- Login host: `notchpeak.chpc.utah.edu` <!-- TODO: confirm cluster (notchpeak, kingspeak, lonepeak, granite) and update ssh_config.example accordingly -->
- CHPC uses Duo two-factor authentication. The Duo passcode is **not** a secret stored anywhere. The supported file-based workflow mirrors the pattern used elsewhere: the operator commits the current passcode to `hpc/tmp/duo.txt` and pushes; the agent ingests it via `git pull` rather than via a steering reply. <!-- TODO: confirm the working repo's 2FA mechanism (Duo passcode file vs push approval) and the exact filename it uses -->
  1. Write `~/.ssh/config` from `hpc/ssh_config.example`, substituting `$CHPC_USERNAME` for `<your-chpc-username>`.
  2. Run the file-based 2FA wrapper (e.g., `python hpc/connect_with_file_totp.py`). It reads the passcode file at prompt time, supplies the password from `$CHPC_PASSWORD`, and opens the persistent `ControlMaster` socket (8 h `ControlPersist`).
  3. If the script reports the code was rejected (expired), **do not return control immediately**. Poll `git pull` every ~15 s for at least 5 min; whenever the passcode file changes, rerun the wrapper. Only escalate to the operator if no fresh code lands in that window. Returning control too quickly forces the operator to spin up a brand new agent session.
  4. Once `ssh -O check chpc` succeeds, the master socket covers all follow-up commands (`ssh chpc ...`, `scp`, `rsync`, `bash hpc/run_remote.sh '...'`) for 8 h with no further authentication.
- CHPC's non-interactive SSH shell may **not** include `sbatch` / `module` / Conda etc. on `PATH` — those are set up by the login shell. **Always wrap remote commands with `bash -lc "..."`** (login shell) so SLURM, the module system, and any user-installed tooling load. For example: `ssh chpc "bash -lc \"cd ... && sbatch job.slurm\""`. This applies to every `ssh chpc '...'` invocation — it is a property of the SSH session, not of any one script. (The persistent `ControlMaster` socket multiplexes the *connection*, not the shell type, so each `ssh chpc '...'` still spawns a fresh non-login shell unless you ask for a login shell explicitly.)
- Remote working directory: <!-- TODO: fill in the CHPC project/scratch path used for this project, and note whether it is a git repo or a plain rsync'd copy --> If it is a plain copy with no `.git`, `git pull` on the remote will fail; sync from the local clone with `rsync -az --delete --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' ./ chpc:<remote-path>/`. **`--exclude='.venv'` is mandatory** if a remote venv exists: without it, `rsync --delete` silently wipes the remote `.venv`, breaking every subsequent SLURM job with `ModuleNotFoundError`. After such a wipe, rebuild the venv via a login shell before resubmitting jobs.
- Note CHPC scratch policies: `/scratch/general/...` filesystems are subject to auto-purge of files untouched for 60 days. Keep anything that must persist in the project or home space, not scratch. <!-- TODO: confirm which filesystem the working directory lives on -->
- Always redact `$CHPC_PASSWORD` from any captured SSH output before reading it back into the agent context: pipe through `sed "s|${CHPC_PASSWORD}|<REDACTED>|g"`. If a transcript ever does capture the password, `shred -u` (or `rm -f`) the file immediately.
- If you ever need to verify the secret is present, check `[ -n "${CHPC_PASSWORD:-}" ]` — never print its contents or length.

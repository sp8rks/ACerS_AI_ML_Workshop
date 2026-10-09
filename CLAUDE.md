# CLAUDE.md

## Coding Agent

- In your comment replies, avoid using #<numeral> style info, such as "#1", unless you're specifically referring to an issue or pull request, since this auto-formats as an issue or PR link. Instead, state "No. 1" or "number 1" etc.
- This repo (ACerS_AI_ML_Workshop) holds public course materials for the ACerS short course "Practical AI and Machine Learning for Materials Science". Its default branch is `main`.
- Be cautious with `~` ("approximately") since if you use it too many times it starts causing strikeout formatting in markdown previews
- Include plots directly in your comment reply via `![image name](https://github.com/<user/org>/<repo>/blob/<shortened-commit-hash>/<filename>?raw=true)`. Truncate the commit hash to the first 7 characters only. For provenance, ensure you use the shortened (7-character) commit hash, not the branch name
- If you mention files in your comment reply, add direct hyperlinks based on the shortened (7-character) commit hash
- In GitHub Actions the runner is destroyed as soon as you post your final comment, so never run long tasks (e.g. Asta AutoDiscovery runs) in the background. Wait for them in the foreground before replying, or state that a follow-up @claude comment is needed to collect results.
- IMPORTANT: Never echo/grep/print environment secrets. These should never be exposed in your terminal history or other outputs

## Asta (Allen Institute for AI)

The Asta plugins (Theorizer, AutoDiscovery, and related tools) are installed via the Claude Code plugin system, configured in `.github/workflows/claude.yml`:

- Marketplace: `https://github.com/allenai/asta-plugins.git`
- Plugins: `asta-tools@asta-plugins` and `asta-flows@asta-plugins`

Authentication is provided via the `ASTA_TOKEN` GitHub Actions secret, exposed as the `ASTA_TOKEN` environment variable. Do not echo, grep, or otherwise surface this token. If Asta calls fail with authentication errors, the token has likely expired; report this in your comment reply so the operator can regenerate it locally (`asta auth login`, then `asta auth print-token --raw --refresh`) and update the repo secret. Asta beta credits are tied to the operator's account, so use the tools purposefully rather than for trivial lookups.

Use Asta tools when the task involves literature-grounded hypothesis generation (Theorizer) or automated discovery workflows (AutoDiscovery), such as proposal background research, identifying prior work, or generating candidate research directions.

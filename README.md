# S&P 500 Breadth & Internals — auto-refreshing dashboard

A self-contained weekly dashboard that reconstructs S&P 500 breadth and internals
bottom-up from index members. A GitHub Action rebuilds it from live Wikipedia +
Yahoo Finance data on a schedule and publishes the result to GitHub Pages, so the
hosted page stays current with no manual step.

## What's in here

| File | Purpose |
|---|---|
| `sp500_internals_rebuild.py` | Fetches data, reconstructs point-in-time membership, computes every series, and writes `sp500_internals_weekly.html`. Runs standalone too. |
| `requirements.txt` | Python dependencies. |
| `.github/workflows/refresh.yml` | Scheduled Action: rebuild → deploy to Pages. |

## One-time setup

1. **Create a repo and push these files** (keep the folder layout — the workflow must sit at `.github/workflows/refresh.yml`).
   ```bash
   git init
   git add .
   git commit -m "S&P 500 internals dashboard"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```

2. **Turn on Pages via Actions.** Repo → **Settings → Pages → Build and deployment → Source: GitHub Actions**. (Not "Deploy from a branch" — this setup deploys an artifact, so nothing is committed back to the repo.)

3. **Run it once.** Repo → **Actions → Refresh S&P 500 internals dashboard → Run workflow**. The first run takes a few minutes (it downloads ~750 tickers). When the `deploy` job finishes, the live URL appears in the job summary, typically:
   ```
   https://<you>.github.io/<repo>/
   ```

After that it refreshes on its own.

## The schedule

`cron: '30 22 * * 2-6'` — 22:30 UTC, Tuesday through Saturday. US cash sessions
close at 20:00–21:00 UTC, so a Tue–Sat UTC run picks up each Mon–Fri US close once
it has settled. 22:30 UTC is 08:30 (AEST) / 09:30 (AEDT) in Melbourne.

The dashboard is weekly, but the current (partial) week updates every run, so daily
refreshes keep the latest bar live and the Friday run captures the settled weekly
close. To change cadence, edit the `cron` line — e.g. `30 22 * * 6` for Saturdays
only (settled-weeks only). Cron is always UTC.

## Things worth knowing

- **Yahoo can rate-limit datacenter IPs.** The rebuild pulls from Yahoo Finance,
  and requests from GitHub's shared runner IPs are occasionally throttled or
  blocked. The script already retries each batch three times. If a run still fails,
  Pages keeps serving the **last successful** deploy, so the site never goes blank —
  just re-run the workflow, or wait for the next scheduled run. If Yahoo blocks
  runners persistently, run the script on your own machine/VM and commit the HTML
  instead (see the alternative below).
- **GitHub disables cron on idle repos.** Scheduled workflows are paused after ~60
  days with no repo activity. A manual run, or any push, re-arms it. Scheduled runs
  can also be delayed a few minutes to an hour under GitHub load — not an issue for
  a daily rebuild.
- **Private repos need a paid plan for Pages.** Public Pages is free. For a private
  Acheron repo, Pages requires GitHub Pro/Team/Enterprise, or use the commit
  alternative below and view the file directly.
- **The Wikipedia table can move.** The script already points at the relocated
  "Historical components of the S&P 500" page. If Wikipedia restructures it again,
  the membership step will error visibly in the Action log rather than fail silently.
- **No data is committed.** Each run rebuilds from source and deploys an artifact.
  There's no daily commit noise, but also no data history in git (see alternative).

## Alternative: commit the HTML instead of deploying an artifact

If you'd rather keep a git history of the generated dashboard (or Yahoo blocks the
runners and you build locally), replace the `deploy` job and give the build job
`contents: write`, then commit the file:

```yaml
      - name: Commit refreshed dashboard
        run: |
          cp sp500_internals_weekly.html index.html
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add index.html
          git diff --quiet --cached || git commit -m "Refresh $(date -u +%F)"
          git push
```

Set Pages to **Deploy from a branch → main → / (root)**. Downside: a commit per run.

## Rebuilding locally

```bash
pip install -r requirements.txt
python sp500_internals_rebuild.py            # writes sp500_internals_weekly.html
python sp500_internals_rebuild.py --cache .cache   # reuse downloaded prices between runs
```

Open the HTML in any browser — it's fully self-contained (Chart.js and fonts load
from CDN; everything else is inline).

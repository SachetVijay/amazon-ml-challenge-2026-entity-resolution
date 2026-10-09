# AWS setup for a $100-credit (new Free Tier) account

## Why the big instances were blocked
New accounts start on the **Free plan**. It only allows small "free-tier eligible" EC2 types
(t3/t4g micro–small, c7i-flex/m7i-flex large: 2 vCPU, ≤ 8 GB RAM). Those are weaker than the compute machine.
To launch a useful machine you **upgrade to the Paid plan**. Your remaining credits carry over and are
spent first; you're only billed beyond the credits. The budget alarm in step 3 guards that.

## What $100 buys (approx. on-demand, check the console for your region's exact price)
| instance | vCPU | RAM | ~$/hour | hours on $100 | use |
|---|---|---|---|---|---|
| **r7i.4xlarge** (recommended) | 16 | 128 GB | ~1.1 | ~90 | whole pipeline in RAM, full training set, parallel experiments |
| r7i.2xlarge | 8 | 64 GB | ~0.55 | ~180 | if the vCPU quota only allows 8 |
| m7i.4xlarge | 16 | 64 GB | ~0.85 | ~115 | fallback if r7i is unavailable |

GPU instances (g5/g6) need a separate quota that new accounts rarely get quickly; skip them.
The compute machine's RTX 3050 handles the one GPU step. Stopping the instance when idle stops the per-hour charge
(its disk costs ~$0.08/GB-month, about $2 for this challenge).

## Steps (≈10 min)
1. **Upgrade the plan**: console → top-right account menu → *Billing and Cost Management* →
   banner "Upgrade to paid plan" (or *Account → Plan*). Confirm credits are still listed under *Credits*.
2. **Region**: pick **Asia Pacific (Mumbai) ap-south-1** in the top-right selector; use it everywhere.
3. **Budget alarm**: Billing → *Budgets* → Create → *Monthly cost budget* → **$90** →
   email alerts at 50 %, 80 %, 100 %.
4. **vCPU quota**: *Service Quotas* → Amazon EC2 → "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z)
   instances". If the applied value is < 16, click *Request increase* → **32**. Often auto-approved in minutes.
5. **Scoped access for Claude** (don't hand over root or admin; this is all I need):
   IAM → Users → *Create user* `amlc-claude` (no console access) → *Attach policies directly* →
   **AmazonEC2FullAccess** + **ServiceQuotasReadOnlyAccess** → Create.
   Open the user → *Security credentials* → *Create access key* → "Command Line Interface".
6. **On your Mac** (you type the keys; never paste them into the chat):
   ```bash
   brew install awscli
   aws configure --profile amlc      # key id, secret, region ap-south-1, output json
   aws sts get-caller-identity --profile amlc
   ```
   Then tell Claude: "profile amlc is ready".

## What Claude then does (all via `--profile amlc`, EC2 only)
- creates key pair `amlc` → `~/.ssh/amlc-aws.pem` (never printed or read), a security group allowing
  SSH from your current IP only, and one r7i.4xlarge with Ubuntu 24.04 + 200 GB gp3 disk
- adds `Host amlc-aws` to `~/.ssh/config`, syncs code + data, runs jobs in tmux
- **stops the instance whenever it's idle**; after the finale, terminates it and deletes the disk + key pair

## Why not full account access
The pipeline only needs EC2. An EC2-scoped key can't touch billing, IAM, or other services, so a leaked or
misused key has a small blast radius. The budget alarm caps surprises. You can revoke the key any time
(IAM → user → Security credentials → Deactivate).

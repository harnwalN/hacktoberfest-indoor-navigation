# Sharing the app through a Cloudflare tunnel

`scripts/share.sh` starts everything the app needs on this Mac and puts it on the internet through a
Cloudflare tunnel:

1. checks the tools (ffmpeg, uv, cloudflared, ollama; node and npm with `--build`)
2. optionally builds the frontend into `backend/static` (`--build`)
3. starts `ollama serve` if Ollama is not already answering
4. starts the backend on `127.0.0.1:8000` and waits for `/api/health`
5. starts the tunnel (quick or named mode, below)

Press Ctrl-C to stop. The script stops the backend, Ollama (only if it started it) and the tunnel.
Logs go to `logs/` at the repo root (gitignored): `backend.log`, `ollama.log`, `cloudflared.log`.

```bash
scripts/share.sh            # start (quick mode unless a named tunnel is configured)
scripts/share.sh --build    # rebuild the frontend first; needed once, and after frontend changes
```

## Quick mode (default, no account needed)

Runs `cloudflared tunnel --url http://127.0.0.1:8000`. Cloudflare gives a random
`https://<words>.trycloudflare.com` address. The script prints it. **It changes on every restart**, so
use it for a one-off demo. For an address that stays the same, use named mode.

## Named mode (stable address on your own domain)

Used when both `TUNNEL_NAME` and `TUNNEL_HOSTNAME` are set, either in the environment or in
`scripts/share.env` (copy `scripts/share.env.example`; the file is gitignored). Setting only one of them
is an error.

### One-time setup (you do this by hand, once)

You need a Cloudflare account and a domain whose DNS is managed in Cloudflare. Replace `<name>` and
`<hostname>` with your values, for example `indoor-nav` and `nav.example.com`.

```bash
# 1. Log in. This opens a browser and asks you to pick the domain. It stores a certificate in ~/.cloudflared.
cloudflared tunnel login

# 2. Create the tunnel. This writes a credentials JSON file into ~/.cloudflared.
cloudflared tunnel create <name>

# 3. Point the hostname at the tunnel. This creates a DNS record in your Cloudflare zone.
cloudflared tunnel route dns <name> <hostname>
```

Then create `~/.cloudflared/config.yml` (on Homebrew installs, `cloudflared` may also read
`/opt/homebrew/etc/cloudflared/config.yml`; check the path it reports on start). Use the tunnel ID that
`cloudflared tunnel create` printed:

```yaml
tunnel: <tunnel-id-or-name>
credentials-file: /Users/<you>/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: <hostname>
    service: http://127.0.0.1:8000
  - service: http_status:404
```

The backend listens only on 127.0.0.1, so the service must be `http://127.0.0.1:8000`.

Then put the names in `scripts/share.env`:

```bash
TUNNEL_NAME=<name>
TUNNEL_HOSTNAME=<hostname>
```

and run `scripts/share.sh`. The app is then at `https://<hostname>`.

## Access

- Visitors (no sign-in) can read the maps and routes and file reports. Walk uploads need a signed-in contributor. Owner
  actions need an owner sign-in (see "Access and roles" in the top-level README).
- Anyone with the address can ask for a sign-in link, but a link goes only to the inbox of the address they typed. In
  `MAIL_BACKEND=log` (the default) links go to the backend log only, so outsiders cannot sign in.
- Keep the address private. Quick-mode addresses change on every run. Named-mode addresses stay the same, so share them only with people you trust.
- Do not set `OWNER_TOKEN` on a tunnel you share. Anyone with that token is the owner.

## Troubleshooting

- **A tool is MISSING:** the script lists the install command for each one.
- **"something already answers at http://127.0.0.1:8000":** an older server is still running. Stop it, then
  run the script again.
- **Ollama:** the script runs `ollama serve` with `OLLAMA_NUM_PARALLEL=4` (unless you set it). If Ollama
  runs elsewhere, set `OLLAMA_HOST` in `scripts/share.env`. The script does not start a remote Ollama.
- **Frontend changes not visible:** run `scripts/share.sh --build`.

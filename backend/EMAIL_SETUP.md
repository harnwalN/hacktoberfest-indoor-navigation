# Turn on real sign-in email (Gmail)

By default the backend does not email sign-in links. It prints them to the log (`MAIL_BACKEND=log`). That is fine for solo testing. To send real emails, follow these steps.

## What you need

- A Gmail account. Use one you control.
- A Gmail "app password". This is not your normal Google password. It is a separate 16-character code that lets an app send mail for you. Google makes you create it, and you can delete it at any time without changing your password.

## Steps

1. Open https://myaccount.google.com and go to **Security**.
2. Turn on **2-Step Verification**. An app password is only offered when this is on.
3. In Security, open **App passwords**.
4. Name it (for example `nav-backend`) and choose **Generate**.
5. Google shows a 16-character code. Copy it. You cannot view it again later.
6. Know your public URL. This is the address people open in the browser, for example your ngrok address `https://abc123.ngrok-free.app` (no trailing slash).
7. Open a terminal in the `backend` folder.
8. Set the variables in the terminal. Replace the example values with your own. (You can also put the non-secret ones in `backend/.env`. Keep `SMTP_PASSWORD` in the terminal, not in a file.)

   ```
   export MAIL_BACKEND=smtp
   export SMTP_HOST=smtp.gmail.com
   export SMTP_PORT=587
   export SMTP_USER=you@example.org
   export SMTP_PASSWORD=abcdefghijklmnop
   export SMTP_FROM=you@example.org
   export PUBLIC_BASE_URL=https://abc123.ngrok-free.app
   ```

   Type the app password without spaces.
9. In the same terminal, start the backend. Either command works:

   ```
   scripts/start.sh
   ```

   (run from the repository root; it checks `SMTP_HOST` and `PUBLIC_BASE_URL` before it starts), or:

   ```
   uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

10. Test it. Ask for a sign-in link in the app. The email should arrive at the address you entered.

## Notes

- The backend reads `backend/.env` when it starts (copy `backend/.env.example` to start). A variable already set in the terminal wins over the file. Restart the backend after you change the file.
- Never put the app password in a file that goes into git.
- If the backend stops with a message about `SMTP_HOST` or `PUBLIC_BASE_URL`, one of the variables is missing.
- `PUBLIC_BASE_URL` must be the address people open in the browser (the tunnel or the domain), not `127.0.0.1`. Links in the email are built from it.
- Rate limits: a person can ask for at most 5 sign-in links per 15 minutes (per address and per IP). Later requests get HTTP 429 until the window passes.
- Each link works once and expires after 15 minutes (`AUTH_LINK_TTL_SEC`).

Restarting the backend after you set these is required, because the settings are read only at startup. Log mode needs no setup and still works for solo testing.

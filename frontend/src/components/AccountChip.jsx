import { User } from 'lucide-react'
import { useAuth } from './AuthContext'

// Small account indicator for the top bar: the signed-in email with Sign out, or a Sign in link to the sign-in page.
// Signed out, "Sign in" always goes to #/sign-in (the sheet is only for in-app prompts that need an identity).
export default function AccountChip() {
  const { account, signedIn, verifying, signOut } = useAuth()
  if (verifying) return <span className="account-chip muted" role="status">Signing you in…</span>
  if (!signedIn) {
    return (
      <a className="account-btn account-signin" href="#/sign-in">
        <User size={16} aria-hidden="true" />
        <span>Sign in</span>
      </a>
    )
  }
  return (
    <span className="account-chip">
      <User size={16} aria-hidden="true" />
      <span className="account-email" title={account.email}>{account.email}</span>
      <button type="button" className="ghost account-out" onClick={signOut}>Sign out</button>
    </span>
  )
}

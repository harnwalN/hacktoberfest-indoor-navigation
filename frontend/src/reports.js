// "Something's wrong" reports: the words shown for each kind, the limits the server enforces, and the message for each
// refused send. Used by the directions panel (ReportIssue) and the reports list (ReportsSection).
import { detail } from './api'

export const REPORT_KINDS = [
  { id: 'wrong_direction', label: 'Directions are wrong' },
  { id: 'place_missing', label: 'A place is missing' },
  { id: 'sign_changed', label: 'A sign has changed' },
  { id: 'photo_wrong', label: "Photo doesn't match" },
  { id: 'blocked', label: 'The way is blocked' },
  { id: 'other', label: 'Something else' },
]
export const KIND_LABEL = Object.fromEntries(REPORT_KINDS.map((k) => [k.id, k.label]))

// Mirrors the backend: note at most 500 characters, resolution note at most 300.
export const REPORT_NOTE_MAX = 500
export const RESOLUTION_NOTE_MAX = 300

export const THANKS = "Thanks, the map's editors will check this"
export const TOO_MANY = "You've sent a lot of reports, try again later"
export const REPORTS_FULL = 'Reports are full for this building'

// The message for a failed send. 429 and 507 have fixed wording; 422 shows the server's detail, naming the field when
// the server gives one. Anything else falls back to the shared message in api.js. Request headers never reach here.
export function reportError(err) {
  const status = err?.response?.status
  if (status === 429) return TOO_MANY
  if (status === 507) return REPORTS_FULL
  if (status === 422) {
    const d = err.response.data?.detail
    if (typeof d === 'string' && d) return d
    if (Array.isArray(d) && d.length) {
      return d
        .map((x) => {
          const field = Array.isArray(x?.loc) ? x.loc.filter((p) => p !== 'body').join('.') : ''
          return field ? `${field}: ${x.msg}` : x.msg
        })
        .filter(Boolean)
        .join('. ')
    }
    return 'Check the report and try again.'
  }
  return detail(err)
}

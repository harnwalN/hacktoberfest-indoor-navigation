# Motion layer

Shared animation pieces for the Waypoint frontend. Use these instead of writing new motion code per screen.
Library: `motion` (v14), imported from `"motion/react"`. Plain CSS keyframes in `../styles.css` still exist; this folder is for things that need JS (presence, layout, gestures).

## What exists

| File | Exports | Use for |
| --- | --- | --- |
| `tokens.js` | `duration`, `ease`, `spring`, `stagger(i)`, `transitionFor(preset)`, `variants`, `isLowEnd()` | Every timing and curve. Import, do not inline numbers. |
| `Toast.jsx` | `ToastProvider` (already mounted in `main.jsx`), `useToast()` | Short status messages: `success`, `error`, `info`. |
| `Skeleton.jsx` | default `Skeleton` | Loading placeholder block with shimmer. |
| `Collapse.jsx` | default `Collapse` | Height-auto disclosure (`<Collapse open>`). |
| `PressButton.jsx` | default `PressButton`, `usePressMotion(scale)` | Press-down feel for buttons. Reduced-motion aware. |
| `useDelayed.js` | default `useDelayed(on, ms)` | Shows a placeholder only after `on` has stayed true for `ms` (no flash on fast loads). |
| `useModal.js` | default `useModal(dialogRef, { returnFocusTo })` | Modal layer: inerts everything outside the dialog while mounted, traps Tab and Shift+Tab inside it, restores focus on close. Used by `SignInDialog` and `Poster`. |
| `map.js` | `CAMERA_TWEEN_MS`, `ZOOM_TWEEN_MS`, `GLOW_MS`, `FLOOR_SLIDE_MS`, `PLANE_STAGGER_MS`, `NODE_STAGGER_MS`, `ease`, `easeInOut`, `stillMode()` | Map tab timings (2D zoom, 3D camera, floor slide, edit-card glow). `stillMode()` is true under reduced motion or on a low-end device. The 3D view also uses it. |
| `dirbuild.js` | `EASE_OUT`, `INSTANT`, `pick(reduce, t)`, `SWAP_SPRING`, `HIGHLIGHT`, `rowIn(i)`, `COLLAPSE`, `FADE`, `CARD_IN` | Directions and Contribute tab motion. `pick(reduce, t)` returns the instant transition under reduced motion. Keep new values here, not inline. |
| Shared CSS: `../styles/motion-map.css`, `../styles/motion-dirbuild.css` | | Keyframes for the Map and Directions/Contribute tabs. The JS constants above must match these timings. |

Already wired (you do not need to touch these):
- `main.jsx`: `<MotionConfig reducedMotion="user">` wraps the app, so motion drops movement when the OS asks for less motion.
- `App.jsx`: the tab pill (`layoutId="tab-pill"`) glides between tabs on desktop and phone. Under reduced motion it is a static pill with no `layoutId`, so it never moves. Tab content slides in from the side the new tab is on (`AnimatePresence mode="wait"`, keyed by tab), about 280 ms in total.
- `App.jsx`: after the first tab switch the page's blocks do not rise in (`.content.no-rise` in `styles.css`). The rise plays on the first app load only, so content is stable about 300 ms after a tap.
- `App.jsx`: a tab switch scrolls the page to the top (`ScrollTopOnMount`, instant). It runs when the new page mounts, after the old page has faded out and before paint, so the old page never jumps while it is visible.
- `components/SignInDialog.jsx`: scrim fades, sheet scales and fades in and out. Uses `useModal`: the app behind is inert while open, focus is trapped inside the sheet, and focus returns to the opener on close.
- The building picker shows a skeleton while the building list loads, after a 180 ms delay. No skeleton after a switch.
- Inputs, selects and textareas get a soft accent glow on keyboard focus (CSS).

## How to use

### Toasts
```jsx
import { useToast } from '../motion/Toast.jsx'

const toast = useToast()
toast.success('Saved')       // green, 4 s
toast.info('Map is updating') // neutral, 4 s
toast.error('Upload failed')  // red, 6 s
```
- The live region is always in the DOM (`aria-live="polite"`), so messages are announced. Do not add a second live region for the same message.
- Desktop: slides in from the top. Phone: rises above the tab bar.
- Max 3 visible. Each has a dismiss button.
- `useToast()` throws outside `ToastProvider`. It is mounted app-wide, so every component can use it.

### Skeleton
```jsx
<Skeleton className="sk-card" />                 // preset sizes in styles.css
<Skeleton width="60%" height={20} radius={8} />  // inline size
```
Decorative (`aria-hidden`). Always pair it with real loading text or a `role="status"` line so screen readers know something is loading.

### Collapse
```jsx
<button aria-expanded={open} aria-controls="more-details" onClick={() => setOpen(!open)}>More</button>
<Collapse open={open} id="more-details">...</Collapse>
```
Children stay mounted when closed (state is kept). Closed content is `inert` and `aria-hidden`, so it cannot be tabbed into.

### Press feedback
```jsx
<PressButton className="primary" onClick={save}>Save</PressButton>
<PressButton pressScale={0.9} className="icon-btn" aria-label="Close"><X /></PressButton>
```
For a `motion.button` of your own, call the hook in your component and spread it:
```jsx
const press = usePressMotion(0.9)
<motion.button {...press} className="ghost">...</motion.button>
```
Disabled buttons do not react. Under reduced motion there is no press scale (the hook returns `{}`).

**Rule: never put `whileTap` on a non-focusable element.** motion's press gesture gives it `tabindex="0"`, which adds an
extra tab stop. Put press motion on a `<button>`, or use CSS `:active` on the icon inside a button (the tab bar does this,
see `.tabbar button:active .tab-icon` in `styles.css`). Any CSS press effect goes inside
`@media (prefers-reduced-motion: no-preference)`.

### Dialogs
Wrap the element in `AnimatePresence` and give the scrim and the sheet `initial`, `animate` and `exit` props. Copy `SignInDialog.jsx` as the pattern.
A dialog must: close on Escape, and restore focus to the element that opened it. `useModal` does the Tab and Shift+Tab trap, so do not write a second one.

Use the shared helper for the inert and focus part:
```jsx
const sheet = useRef(null)
useModal(sheet)                              // before any effect that focuses inside the sheet
useModal(sheet, { returnFocusTo: btnRef })   // or: focus btnRef.current on close instead of the opener
```
- While the dialog is mounted, every element outside it (its siblings up to `<body>`) gets `inert`. Tab, the accessibility tree and clicks behind it are off.
- Tab and Shift+Tab wrap between the first and last focusable control in the dialog (`useModal` listens on `window`). Focus that lands outside is pulled back in. `Poster` and `SignInDialog` share this trap.
- On close the cleanup removes `inert` FIRST, then restores focus. An inert element cannot take focus, so the order matters.
- The opener is the element focused when the dialog mounts, unless `returnFocusTo` is given.
- The dialog must be in the DOM when the effect runs. Put the helper before any effect that moves focus into the dialog, so the opener is captured first.
- Toast live region: the toast region sits in the app tree, so it is inert while a dialog is open. Toasts shown during a dialog are not announced until it closes. Known limit: a toast raised while a dialog is open is not announced until it closes.

### Loading placeholders
```jsx
const showSkel = useDelayed(!loaded, 180)   // no flash on fast loads
{showSkel && <Skeleton className="sk-card" />}
```

### Lists
`variants.listItem` plus `stagger(i)`:
```jsx
<motion.ul initial="hidden" animate="show">
  {items.map((it, i) => (
    <motion.li key={it.id} variants={variants.listItem} transition={{ delay: stagger(i) }}>...</motion.li>
  ))}
</motion.ul>
```

### Heavy effects
```jsx
if (!isLowEnd()) startConfetti()
```
`isLowEnd()` is true for reduced motion, `hardwareConcurrency <= 4` or `deviceMemory <= 4`. Use it to skip confetti, long loops and big blurs. Light fades and taps are fine everywhere.

### Tokens
- `duration.fast` 0.14 s (press, hover, colour), `base` 0.22 s (things appearing), `slow` 0.36 s (large surfaces).
- `duration.tabExit` 0.08 s and `tabEnter` 0.20 s: tab switch, 0.28 s total. Budget is 300 ms, do not lengthen.
- `ease.out` arrives, `ease.in` leaves, `ease.pop` is the one overshoot (celebrations only).
- `spring.snappy` for taps and re-targeted pills, `gentle` for sheets and toasts, `bouncy` for rare celebratory moments.
- `transitionFor(preset)` returns an instant transition when reduced motion is on. Use it for any custom transition.

## Rules
1. Animate only `transform` and `opacity`. The one exception is `height`, and only through `Collapse`. Never animate `top`, `left`, `width`, margins or fonts.
2. Durations 120 to 450 ms for UI. Nothing longer than 600 ms outside the existing decorative art.
3. Every new effect must be harmless under reduced motion. `MotionConfig` removes movement, so check that the result still reads (the content just appears).
4. Do not remount heavy components for an animation. The tab switch already unmounts the tab's content (this is existing behaviour); do not add more remounts.
5. Keep focus visible. Animations must not replace the `:focus-visible` outline.
6. Do not add a second live region for a toast. Do not put toasts inside `role="dialog"`.
7. Decorative motion is `aria-hidden`. Status text stays readable.
8. Test at 390 px and 1280 px, and with the OS reduced-motion setting on.
9. Never `whileTap` a non-focusable element (see Press feedback). Gate every press scale on reduced motion.
10. Toasts on the trip view (phone) sit above the bottom dock. The lift is `calc(var(--tabbar-h) + var(--step-nav-h) + var(--strip-h) + 8px)`: the dock heights are global tokens on `:root` in `styles.css`. Change a dock height there, and the toast follows.
11. Any dialog uses `useModal`. Do not add a new `inert` toggle or a second focus-restore per dialog.
12. Reduced motion means no `layoutId` and no transform. A reduced-motion pill is a static element, so an animated pill must sit behind `!reduceMotion`.

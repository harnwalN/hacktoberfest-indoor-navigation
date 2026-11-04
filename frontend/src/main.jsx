import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import '@fontsource/atkinson-hyperlegible/400.css'
import '@fontsource/atkinson-hyperlegible/700.css'
import '@fontsource/bricolage-grotesque/700.css'
import '@fontsource/bricolage-grotesque/800.css'
import '@fontsource/instrument-serif/400.css'
import '@fontsource/instrument-serif/400-italic.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/500.css'
import './styles.css'
import './styles/motion-dirbuild.css'
import { registerOffline } from './offline.js'
import { AuthProvider } from './components/AuthContext.jsx'
import { MotionConfig } from 'motion/react'
import { ToastProvider } from './motion/Toast.jsx'

registerOffline()

// Looping decoration (the planner's "you are here" ping, shimmers, the selected-place pulse) stops while the
// page is in a background tab, so it costs nothing when nobody can see it.
const syncHidden = () => document.documentElement.classList.toggle('page-hidden', document.hidden)
document.addEventListener('visibilitychange', syncHidden)
syncHidden()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/* reducedMotion="user": motion drops transform/layout animation when the OS asks for less motion. */}
    <MotionConfig reducedMotion="user">
      <ToastProvider>
        <AuthProvider>
          <App />
        </AuthProvider>
      </ToastProvider>
    </MotionConfig>
  </React.StrictMode>,
)

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import '@fontsource/atkinson-hyperlegible/400.css'
import '@fontsource/atkinson-hyperlegible/700.css'
import '@fontsource/bricolage-grotesque/700.css'
import '@fontsource/bricolage-grotesque/800.css'
import './styles.css'

// Looping decoration (the planner's "you are here" ping, shimmers, the selected-place pulse) stops while the
// page is in a background tab, so it costs nothing when nobody can see it.
const syncHidden = () => document.documentElement.classList.toggle('page-hidden', document.hidden)
document.addEventListener('visibilitychange', syncHidden)
syncHidden()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

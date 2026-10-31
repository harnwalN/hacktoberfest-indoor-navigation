import { Component } from 'react'

// Keeps one failing component from blanking the whole app.
export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught', error, info?.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="quiet-state" role="alert">
        <h2>{this.props.title || 'Something went wrong'}</h2>
        <p>This part of the app hit an error. The rest of the app still works.</p>
        <button type="button" onClick={() => this.setState({ error: null })}>Try again</button>
      </div>
    )
  }
}

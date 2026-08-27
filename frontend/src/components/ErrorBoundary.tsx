import { Component, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  // Optional fallback override. Default fallback below is fine for
  // most uses; pass a custom one if a parent wants tighter theming.
  fallback?: (err: Error, reset: () => void) => ReactNode;
}

interface State {
  err: Error | null;
}

// Top-level React error boundary. Catches render-time exceptions from
// any descendant and renders a fallback instead of blanking the SPA.
// `componentDidCatch` logs to the console; a future iteration can POST
// to /api/client-error for server-side tracking.

export class ErrorBoundary extends Component<Props, State> {
  override state: State = { err: null };

  static getDerivedStateFromError(err: Error): State {
    return { err };
  }

  override componentDidCatch(error: Error, info: { componentStack?: string | null }) {
    // eslint-disable-next-line no-console
    console.error('[ErrorBoundary] caught', error, info.componentStack);
  }

  reset = () => this.setState({ err: null });

  override render() {
    const { err } = this.state;
    if (!err) return this.props.children;
    if (this.props.fallback) return this.props.fallback(err, this.reset);
    return <DefaultFallback err={err} reset={this.reset} />;
  }
}

function DefaultFallback({ err, reset }: { err: Error; reset: () => void }) {
  return (
    <div className="p-10 max-w-[560px] mx-auto text-center">
      <h2 className="text-base text-red-300 mb-2">Something went wrong</h2>
      <p className="text-fg-1 text-[13px] mb-4">
        A render-time exception broke the page. Reload to recover.
      </p>
      <button
        type="button"
        className="px-3 py-1.5 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover"
        onClick={() => window.location.reload()}
      >
        Reload
      </button>
      <details className="mt-4 text-left text-[11px] text-fg-1">
        <summary className="cursor-pointer hover:text-fg">Show details</summary>
        <pre className="mt-2 p-2 bg-bg-1 border border-line2 rounded overflow-auto whitespace-pre-wrap">
{err.name}: {err.message}
        </pre>
      </details>
      <button
        type="button"
        className="block mx-auto mt-3 text-[11px] text-fg-1 underline hover:text-fg"
        onClick={reset}
      >
        Try without reloading
      </button>
    </div>
  );
}
import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/app.css';
import { App } from './App';
import { AppProvider } from './state/AppContext';
import { ToastProvider } from './components/ToastProvider';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ToastProvider>
      <AppProvider>
        <App />
      </AppProvider>
    </ToastProvider>
  </React.StrictMode>,
);

// Service worker registration (UX-27). Only registers on HTTPS or
// localhost — `serviceWorker` is undefined on plain http origins and
// `navigator.serviceWorker` throws on insecure contexts otherwise.
if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch((err) => {
      // Non-fatal: the app still works without offline support.
      // eslint-disable-next-line no-console
      console.warn('SW registration failed:', err);
    });
  });
}
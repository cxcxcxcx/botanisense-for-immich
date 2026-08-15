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
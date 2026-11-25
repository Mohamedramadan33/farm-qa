import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';

if (typeof window !== 'undefined') {
  const win = window as unknown as { gm_authFailure?: () => void };
  const prevAuthFailure = win.gm_authFailure;
  win.gm_authFailure = () => {
    window.dispatchEvent(new CustomEvent('gmp-quota-exceeded'));
    if (typeof prevAuthFailure === 'function') prevAuthFailure();
  };

  const origErr = console.error;
  console.error = (...args: unknown[]) => {
    origErr.apply(console, args);
    const hasQuotaErr = args.some(
      (a) => typeof a === 'string' && /OverQuotaMapError|QuotaExceededError/.test(a)
    );
    if (hasQuotaErr) {
      window.dispatchEvent(new CustomEvent('gmp-quota-exceeded'));
    }
  };
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);


/**
 * Clipboard helper for a self-hosted app that is often served over plain
 * HTTP on a LAN (e.g. http://192.168.84.113:8770). `navigator.clipboard`
 * only exists in a secure context, so a bare `navigator.clipboard.writeText`
 * throws there and the old Copy button just said "Copy failed".
 *
 * Order:
 *   1. real Clipboard API when available;
 *   2. select the visible source element (so the user can press Ctrl+C /
 *      long-press → Copy) and try the legacy execCommand('copy');
 *   3. if there is no source element, show a selected readonly textarea.
 */
export type CopyResult = 'copied' | 'selected' | 'failed';

export function selectElementText(element: Element | null): boolean {
  if (!element) return false;
  if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
    element.focus();
    element.select();
    return element.selectionStart !== element.selectionEnd;
  }
  const selection = window.getSelection();
  if (!selection) return false;
  const range = document.createRange();
  range.selectNodeContents(element);
  selection.removeAllRanges();
  selection.addRange(range);
  return selection.toString().length > 0;
}

function selectedTextarea(text: string): boolean {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.readOnly = true;
  textarea.setAttribute('aria-label', 'Copy this text');
  textarea.style.position = 'fixed';
  textarea.style.left = '50%';
  textarea.style.bottom = '24px';
  textarea.style.transform = 'translateX(-50%)';
  textarea.style.width = 'min(90vw, 560px)';
  textarea.style.zIndex = '10000';
  textarea.style.padding = '10px';
  textarea.style.fontFamily = 'ui-monospace, monospace';
  textarea.style.fontSize = '12px';
  textarea.style.background = 'var(--bg-1, #18181b)';
  textarea.style.color = 'var(--fg, #fafafa)';
  textarea.style.border = '1px solid var(--line2, #52525b)';
  textarea.style.borderRadius = '6px';
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();

  const cleanup = () => {
    textarea.remove();
    document.removeEventListener('keydown', onKey);
  };
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape' || event.key === 'Enter') cleanup();
  };
  document.addEventListener('keydown', onKey);
  // Keep it on screen long enough to long-press / Ctrl+C. A click outside
  // is also a reasonable dismissal.
  textarea.addEventListener('blur', () => window.setTimeout(cleanup, 8000), { once: true });
  return true;
}

export async function copyTextOrSelect(
  text: string,
  source?: Element | null,
): Promise<CopyResult> {
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return 'copied';
    } catch {
      // Fall through: permission denied or a browser that exposes the API
      // but blocks it on this origin.
    }
  }

  const selected = selectElementText(source ?? null);
  try {
    if (document.execCommand('copy')) return 'copied';
  } catch {
    // execCommand is deprecated and may be gone; the selection still helps.
  }
  if (selected) return 'selected';
  return selectedTextarea(text) ? 'selected' : 'failed';
}

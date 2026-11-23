import { useEffect, useState } from 'react';
import { onAuthStateChanged, signInWithCustomToken, signOut, type User } from 'firebase/auth';
import {
  botanisenseAuth,
  BOTANISENSE_WEB_URL,
} from '../lib/botanisense';
import {
  fetchBotanisenseMe,
  pollBotanisenseDeviceToken,
  revokeBotanisenseIntegration,
  startBotanisenseDeviceCode,
  type BotanisenseMe,
  type DeviceCodeStart,
} from '../api/botanisense';
import { useToasts } from './ToastProvider';

interface Props {
  compact?: boolean;
}

/**
 * Connect this instance to the user's own BotaniSense account.
 *
 * The user approves a code on botanisense.app; we then receive a Firebase
 * custom token for *their* uid. From that point the browser has a normal
 * BotaniSense session and can read Firestore directly (owner rules).
 */
export function BotanisenseConnect({ compact = false }: Props) {
  const push = useToasts();
  const [user, setUser] = useState<User | null>(botanisenseAuth.currentUser);
  const [me, setMe] = useState<BotanisenseMe | null>(null);
  const [flow, setFlow] = useState<DeviceCodeStart | null>(null);
  const [starting, setStarting] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [disconnecting, setDisconnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return onAuthStateChanged(botanisenseAuth, next => {
      setUser(next);
      if (!next) setMe(null);
    });
  }, []);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    fetchBotanisenseMe(user)
      .then(result => {
        if (!cancelled) setMe(result);
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [user]);

  // Poll the device-code endpoint while the user approves on botanisense.app.
  useEffect(() => {
    if (!flow || !waiting) return;
    let cancelled = false;
    let timer: number | undefined;

    const tick = async () => {
      try {
        const result = await pollBotanisenseDeviceToken(flow.device_code);
        if (cancelled) return;
        if (result.status === 'approved') {
          await signInWithCustomToken(botanisenseAuth, result.customToken);
          setFlow(null);
          setWaiting(false);
          setError(null);
          push(
            {
              kind: 'success',
              message: result.email
                ? `Connected to BotaniSense as ${result.email}.`
                : 'Connected to BotaniSense.',
            },
            3500,
          );
          return;
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setFlow(null);
        setWaiting(false);
        return;
      }
      timer = window.setTimeout(tick, Math.max(3, flow.interval || 5) * 1000);
    };

    timer = window.setTimeout(tick, Math.max(3, flow.interval || 5) * 1000);
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [flow, waiting, push]);

  const start = async () => {
    setStarting(true);
    setError(null);
    try {
      const next = await startBotanisenseDeviceCode();
      setFlow(next);
      setWaiting(true);
      const approvalUrl = `${next.verification_uri}?code=${encodeURIComponent(next.user_code)}`;
      window.open(approvalUrl, '_blank', 'noopener,noreferrer');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  };

  const disconnect = async () => {
    const current = botanisenseAuth.currentUser;
    setDisconnecting(true);
    setError(null);
    try {
      if (current) await revokeBotanisenseIntegration(current);
    } catch (err) {
      // A revoked/expired token must not trap the user in a connected UI.
      push(
        {
          kind: 'error',
          message: `Could not revoke on the server: ${
            err instanceof Error ? err.message : String(err)
          }. Signing out locally.`,
        },
        5000,
      );
    } finally {
      await signOut(botanisenseAuth);
      setMe(null);
      setDisconnecting(false);
    }
  };

  const approvalUrl = flow
    ? `${flow.verification_uri}?code=${encodeURIComponent(flow.user_code)}`
    : `${BOTANISENSE_WEB_URL}/tabs`;

  return (
    <section
      className={`rounded-lg border border-line2 bg-bg-1 ${compact ? 'p-3' : 'p-4'}`}
      aria-label="BotaniSense account"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="m-0 text-[0.95rem] font-semibold text-fg">BotaniSense account</h3>
          <p className="mt-1 mb-0 text-[0.8rem] leading-relaxed text-fg-1">
            Sign in with your own BotaniSense account to read your plant diary here and
            send selected photos to it.
          </p>
        </div>
        {user ? (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/50 px-2 py-0.5 text-[0.7rem] uppercase tracking-wider text-accent">
            Connected
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-line2 px-2 py-0.5 text-[0.7rem] uppercase tracking-wider text-fg-2">
            Not connected
          </span>
        )}
      </div>

      {error && (
        <div
          role="alert"
          className="mt-3 rounded border border-[#a44] bg-[#3a1414] px-3 py-2 text-[0.8rem] text-red-200"
        >
          {error}
        </div>
      )}

      {user ? (
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <div className="min-w-0">
            <div className="truncate text-[0.85rem] font-medium text-fg">
              {user.displayName || user.email || 'BotaniSense user'}
            </div>
            <div className="truncate text-[0.75rem] text-fg-2">
              {user.email || user.uid}
              {me?.grant ? ` · ${me.grant.importedPhotos} imported` : ''}
            </div>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <a
              className="rounded border border-line2 px-2.5 py-1 text-[0.8rem] text-fg-1 no-underline hover:border-accent hover:text-fg"
              href={`${BOTANISENSE_WEB_URL}/tabs`}
              target="_blank"
              rel="noreferrer"
            >
              Open BotaniSense
            </a>
            <button
              type="button"
              disabled={disconnecting}
              className="rounded border border-[#a44] px-2.5 py-1 text-[0.8rem] text-red-300 hover:bg-[#3a1414] disabled:opacity-50"
              onClick={disconnect}
            >
              {disconnecting ? 'Disconnecting…' : 'Disconnect'}
            </button>
          </div>
        </div>
      ) : flow ? (
        <div className="mt-3">
          <div className="text-[0.8rem] text-fg-1">
            Open the approval page and confirm this code:
          </div>
          <div className="my-2 flex flex-wrap items-center gap-3">
            <code className="rounded border border-line2 bg-bg px-3 py-1.5 text-[1.3rem] font-bold tracking-[0.2em] text-fg">
              {flow.user_code}
            </code>
            <a
              className="text-[0.8rem] text-accent underline"
              href={approvalUrl}
              target="_blank"
              rel="noreferrer"
            >
              Open approval page
            </a>
          </div>
          <div className="flex items-center gap-2 text-[0.8rem] text-fg-2">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-accent" />
            Waiting for approval…
            <button
              type="button"
              className="ml-2 rounded border border-line2 px-2 py-0.5 text-[0.75rem] text-fg-1 hover:border-accent"
              onClick={() => {
                setFlow(null);
                setWaiting(false);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={starting}
            className="rounded bg-accent-dark border border-accent px-3 py-1.5 text-[0.85rem] font-medium text-white hover:bg-accent-hover disabled:opacity-50"
            onClick={start}
          >
            {starting ? 'Starting…' : 'Connect BotaniSense'}
          </button>
          <span className="text-[0.75rem] text-fg-2">
            You will sign in on botanisense.app; this instance never sees your password.
          </span>
        </div>
      )}
    </section>
  );
}

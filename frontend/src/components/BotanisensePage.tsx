import { useEffect, useMemo, useState } from 'react';
import { onAuthStateChanged, type User } from 'firebase/auth';
import {
  collection,
  doc,
  limit,
  onSnapshot,
  orderBy,
  query,
} from 'firebase/firestore';
import {
  botanisenseAuth,
  botanisenseDb,
  botanisenseImageUrl,
  BOTANISENSE_WEB_URL,
} from '../lib/botanisense';
import { BotanisenseConnect } from './BotanisenseConnect';

interface DiaryTag {
  type: 'plant' | 'container' | 'location';
  id: string;
  name: string;
  plantName?: string;
  instanceName?: string;
}

interface DiaryEntry {
  id: string;
  notes?: string;
  tags?: DiaryTag[];
  photoDocIds?: string[];
  photoDownloadURLs?: string[];
  timestamp?: unknown;
  kind?: string;
}

function toDate(value: unknown): Date | null {
  if (!value) return null;
  if (typeof value === 'object') {
    const maybe = value as { toDate?: () => Date; seconds?: number };
    if (typeof maybe.toDate === 'function') return maybe.toDate();
    if (typeof maybe.seconds === 'number') return new Date(maybe.seconds * 1000);
  }
  if (typeof value === 'string' || typeof value === 'number') {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }
  return null;
}

function formatDate(date: Date | null): string {
  if (!date) return 'No date';
  return new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function tagLabel(tag: DiaryTag): string {
  if (tag.type === 'plant') return tag.instanceName || tag.plantName || tag.name;
  return tag.name;
}

export function BotanisensePage() {
  const [user, setUser] = useState<User | null>(botanisenseAuth.currentUser);
  const [entries, setEntries] = useState<DiaryEntry[]>([]);
  const [tags, setTags] = useState<DiaryTag[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return onAuthStateChanged(botanisenseAuth, next => {
      setUser(next);
      if (!next) {
        setEntries([]);
        setTags([]);
        setError(null);
      }
    });
  }, []);

  useEffect(() => {
    if (!user) return;
    setLoading(true);
    setError(null);

    const unsubscribeEntries = onSnapshot(
      query(
        collection(botanisenseDb, 'users', user.uid, 'plantDiaryEntries'),
        orderBy('timestamp', 'desc'),
        limit(100),
      ),
      snapshot => {
        setEntries(
          snapshot.docs.map(item => ({ id: item.id, ...(item.data() as Omit<DiaryEntry, 'id'>) })),
        );
        setLoading(false);
      },
      err => {
        setError(err.message);
        setLoading(false);
      },
    );

    const unsubscribeTags = onSnapshot(
      doc(botanisenseDb, 'users', user.uid, 'diaryTags', 'data'),
      snapshot => {
        const list = (snapshot.data()?.tags ?? []) as DiaryTag[];
        setTags(Array.isArray(list) ? list : []);
      },
      () => {
        // A missing tags doc is normal for a fresh account.
        setTags([]);
      },
    );

    return () => {
      unsubscribeEntries();
      unsubscribeTags();
    };
  }, [user]);

  const tagIndex = useMemo(() => {
    const map = new Map<string, DiaryTag>();
    for (const tag of tags) map.set(`${tag.type}:${tag.id}`, tag);
    return map;
  }, [tags]);

  return (
    <section className="mx-auto max-w-[1100px] px-3 py-4 sm:px-6">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="m-0 text-lg font-semibold text-fg">BotaniSense diary</h2>
          <p className="mt-1 mb-0 text-[0.8rem] text-fg-1">
            Live from your BotaniSense account. Edits made in the app appear here
            automatically.
          </p>
        </div>
        <a
          className="text-[0.8rem] text-accent underline"
          href={`${BOTANISENSE_WEB_URL}/tabs`}
          target="_blank"
          rel="noreferrer"
        >
          Open in BotaniSense
        </a>
      </div>

      <BotanisenseConnect />

      {!user ? (
        <div className="mt-4 rounded-lg border border-line2 bg-bg-1 p-4 text-[0.85rem] text-fg-1">
          Connect your account above to see your plant diary here.
        </div>
      ) : (
        <div className="mt-4">
          <div className="mb-2 flex items-center justify-between text-[0.75rem] uppercase tracking-wider text-fg-2">
            <span>{loading ? 'Loading…' : `${entries.length} entries`}</span>
            <span>Live</span>
          </div>

          {error && (
            <div
              role="alert"
              className="mb-3 rounded border border-[#a44] bg-[#3a1414] px-3 py-2 text-[0.8rem] text-red-200"
            >
              {error}
            </div>
          )}

          {!loading && entries.length === 0 && !error && (
            <div className="rounded-lg border border-line2 bg-bg-1 p-4 text-[0.85rem] text-fg-1">
              No diary entries yet. Add one in BotaniSense, or send photos from the
              Browse tab.
            </div>
          )}

          <div className="grid gap-3">
            {entries.map(entry => {
              const date = toDate(entry.timestamp);
              const entryTags = (entry.tags ?? []).map(tag => {
                const resolved = tagIndex.get(`${tag.type}:${tag.id}`) ?? tag;
                return resolved;
              });
              const photos = entry.photoDownloadURLs ?? [];
              return (
                <article
                  key={entry.id}
                  className="rounded-lg border border-line2 bg-bg-1 p-3 sm:p-4"
                >
                  <header className="flex flex-wrap items-center gap-2">
                    <time className="text-[0.8rem] font-medium text-fg">
                      {formatDate(date)}
                    </time>
                    {entryTags.map(tag => (
                      <span
                        key={`${tag.type}:${tag.id}`}
                        className="rounded-full border border-line2 px-2 py-[1px] text-[0.7rem] text-fg-1"
                        title={tag.plantName || tag.name}
                      >
                        {tag.type === 'plant' ? '🌱 ' : tag.type === 'location' ? '📍 ' : '🪴 '}
                        {tagLabel(tag)}
                      </span>
                    ))}
                  </header>

                  {entry.notes && (
                    <p className="mt-2 mb-0 whitespace-pre-wrap text-[0.85rem] leading-relaxed text-fg-1">
                      {entry.notes}
                    </p>
                  )}

                  {photos.length > 0 && (
                    <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-4">
                      {photos.map((path, index) => (
                        <img
                          key={`${entry.id}:${index}`}
                          src={botanisenseImageUrl(path)}
                          alt=""
                          loading="lazy"
                          className="aspect-square w-full rounded object-cover"
                        />
                      ))}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

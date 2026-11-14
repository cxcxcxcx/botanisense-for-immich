// Map view — geotagged photos grouped into "places" by distance alone.
//
// The grouping itself is server-side (`map_clusters.py` + /api/map/clusters)
// because the DB is the source of truth and the browser should not download
// every row just to draw the map. This component is the presentation half:
// OpenStreetMap tiles via Leaflet, one circle marker per place sized by
// photo count. Clicking a marker opens a bottom sheet with that place's
// photos *without leaving the map*; "Open in grid" is the escape hatch that
// drops the place into the normal browse filters for the full grid.
//
// Stack matches the Solar app this was ported from: Leaflet + OSM tiles,
// no API key.

import { useEffect, useMemo, useRef, useState } from 'react';
import { CircleMarker, TileLayer, Tooltip, useMap } from 'react-leaflet';
import type { Map as LeafletMap } from 'leaflet';
import 'leaflet/dist/leaflet.css';

import SafeMapContainer from './SafeMapContainer';
import {
  applyLocationGuesses,
  clearLocationGuesses,
  getLocationGuesses,
  getMapClusters,
  getResults,
} from '../api/endpoints';
import type {
  LocationGuessStats,
  MapCluster,
  MapClustersResponse,
  ResultRow,
  ResultsResponse,
} from '../api/types';
import { useApp, type Filters } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { navigate } from '../hooks/useRoute';
import { thumbUrl } from '../lib/thumb';

const OSM_TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>';

/** World view used until the first payload arrives (or when there is no GPS). */
const FALLBACK_CENTER: [number, number] = [25, 0];
const FALLBACK_ZOOM = 2;

/**
 * Marker radius, log-scaled against the largest place in the result. A
 * linear scale would turn every small place into an indistinguishable dot
 * next to a 2,000-photo one.
 */
const MIN_RADIUS = 6;
const MAX_RADIUS = 24;

/** Photos loaded per page in the place sheet. */
const PLACE_PAGE_SIZE = 60;

/** Viewport + granularity survive tab switches (per browser tab). */
const MAP_VIEW_KEY = 'botanisense:map-view';
const MAP_GRAN_KEY = 'botanisense:map-granularity';

export type MapGranularity = 'fine' | 'wide' | 'region';

interface SavedView {
  lat: number;
  lng: number;
  zoom: number;
}

function readSession<T>(key: string): T | null {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function writeSession(key: string, value: unknown): void {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode / quota — persistence is optional */
  }
}

function radiusFor(count: number, maxCount: number): number {
  if (maxCount <= 1) return MIN_RADIUS;
  const t = Math.min(1, Math.log1p(count) / Math.log1p(Math.max(2, maxCount)));
  return MIN_RADIUS + t * (MAX_RADIUS - MIN_RADIUS);
}

function fmtDate(iso: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString();
}

/**
 * Frame the markers on the **first** data load, then leave the viewport
 * alone.
 *
 * The old version re-fit whenever `points` changed, which meant switching
 * the cluster radius (150 m → 1 km → 10 km) threw the user back to the world
 * view. Fitting is now a one-shot event; the "Fit" button in the panel is
 * the explicit way to re-frame. A saved viewport from a previous mount is
 * restored instead of fitting, so tab switches keep your place too.
 *
 * Also keeps Leaflet's cached container size honest: the map can mount
 * mid-layout (modal open animation, mobile header wrap), so the first
 * measurement may be too narrow. A ResizeObserver invalidates the size and
 * retries the one-shot fit until it has data and a real box.
 */
function FitToPlaces({
  points,
  singlePointZoom,
  initialView,
}: {
  points: Array<[number, number]>;
  singlePointZoom: number;
  initialView: SavedView | null;
}) {
  const map = useMap();
  const pointsRef = useRef(points);
  pointsRef.current = points;
  const fittedRef = useRef(false);
  const key = points.map((p) => p.join(',')).join(';');

  const fitOnce = () => {
    if (fittedRef.current) return;
    const pts = pointsRef.current;
    if (pts.length === 0) return;
    fittedRef.current = true;
    if (initialView) {
      map.setView([initialView.lat, initialView.lng], initialView.zoom, {
        animate: false,
      });
      return;
    }
    const only = pts[0];
    if (pts.length === 1 && only) {
      map.setView(only, singlePointZoom, { animate: false });
      return;
    }
    // maxZoom keeps a compact library (all shots in one park) from zooming
    // to street level and losing the surrounding context.
    map.fitBounds(pts, { padding: [32, 32], maxZoom: 14, animate: false });
  };

  useEffect(() => {
    const observer = new ResizeObserver(() => {
      map.invalidateSize({ animate: false });
      fitOnce();
    });
    observer.observe(map.getContainer());
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map]);

  useEffect(() => {
    fitOnce();
    // Only the first non-empty data load fits; later point-set changes
    // (granularity, scope) deliberately keep the current viewport.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, map]);

  return null;
}

/** Persist the current viewport so remounting the tab restores it. */
function PersistView() {
  const map = useMap();
  useEffect(() => {
    const save = () => {
      const center = map.getCenter();
      writeSession(MAP_VIEW_KEY, {
        lat: center.lat,
        lng: center.lng,
        zoom: map.getZoom(),
      });
    };
    map.on('moveend', save);
    map.on('zoomend', save);
    return () => {
      map.off('moveend', save);
      map.off('zoomend', save);
    };
  }, [map]);
  return null;
}

/**
 * Bring a selected place into view above the photo sheet. The sheet covers
 * roughly the bottom half, so centering the marker would hide it; project
 * the point, push the map centre below it, and unproject.
 */
function FocusPlace({ place }: { place: MapCluster | null }) {
  const map = useMap();
  const id = place?.id ?? '';
  useEffect(() => {
    if (!place) return;
    const zoom = Math.max(map.getZoom(), 13);
    const target = map
      .project([place.latitude, place.longitude], zoom)
      .add([0, 140]);
    map.setView(map.unproject(target, zoom), zoom, { animate: true });
    // Re-run only when a different place is selected; panning the map is
    // the user's business in between.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, map]);
  return null;
}

/** Browse facets a scoped map (the cluster modal) may apply. */
export type MapScope = Pick<
  Filters,
  | 'cluster_id'
  | 'supergroup'
  | 'species'
  | 'genus'
  | 'family'
  | 'has_person'
  | 'plantae'
  | 'include_non_plants'
>;

export interface LocationMapViewProps {
  /**
   * Facets to scope the map to. Omit on the independent Map tab — that view
   * must not inherit whatever cluster/species the Browse tab happens to be
   * filtered to.
   */
  scope?: MapScope;
  /** 'page' fills the viewport; 'modal' fits inside the Grid's map dialog. */
  variant?: 'page' | 'modal';
  /** Read-only description of the scope, shown in the panel. */
  scopeLabel?: string;
  /**
   * "Open in grid" handler. The scoped modal uses this to close itself and
   * land on the place's photos; the page variant defaults to URL filters +
   * navigation.
   */
  onOpenInGrid?: (
    place: MapCluster,
    granularity: MapGranularity,
    minPhotos: number,
  ) => void;
}

export function LocationMapView({
  scope,
  variant = 'page',
  scopeLabel,
  onOpenInGrid,
}: LocationMapViewProps = {}) {
  const { dispatch } = useApp();
  const setFilters = useUrlFilters();

  const [granularity, setGranularity] = useState<MapGranularity>(() =>
    variant === 'page' ? (readSession<MapGranularity>(MAP_GRAN_KEY) ?? 'fine') : 'fine',
  );
  const [showSingles, setShowSingles] = useState(false);
  useEffect(() => {
    if (variant === 'page') writeSession(MAP_GRAN_KEY, granularity);
  }, [granularity, variant]);
  const initialView = useMemo(
    () => (variant === 'page' ? readSession<SavedView>(MAP_VIEW_KEY) : null),
    [variant],
  );
  const mapRef = useRef<LeafletMap | null>(null);
  const [data, setData] = useState<MapClustersResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Locally inferred coordinates (location_guesses). `includeGuessed` is a
  // view toggle, not a data change; the guesses themselves are recomputed
  // explicitly via the action row.
  const [includeGuessed, setIncludeGuessed] = useState(true);
  const [guessStats, setGuessStats] = useState<LocationGuessStats | null>(null);
  const [guessBusy, setGuessBusy] = useState(false);
  const [guessError, setGuessError] = useState<string | null>(null);
  const [guessVersion, setGuessVersion] = useState(0);

  // The place whose photos are open in the bottom sheet. Keeping this
  // separate from the URL filters is the whole point: browsing a place no
  // longer unmounts the map.
  const [selected, setSelected] = useState<MapCluster | null>(null);
  const [placeRows, setPlaceRows] = useState<ResultRow[]>([]);
  const [placeTotal, setPlaceTotal] = useState(0);
  const [placeLoading, setPlaceLoading] = useState(false);
  const [placeLoadingMore, setPlaceLoadingMore] = useState(false);
  const [placeError, setPlaceError] = useState<string | null>(null);
  // The controls panel is huge on a phone; start collapsed there and let the
  // compact summary pill expand it. Desktop starts open.
  const [panelOpen, setPanelOpen] = useState(() =>
    typeof window === 'undefined' ? true : window.innerWidth >= 640,
  );

  // Only the facets the map endpoint understands. On the Map tab `scope` is
  // undefined, so none of the Browse tab's cluster/taxon filters leak in.
  const {
    cluster_id,
    supergroup,
    species,
    genus,
    family,
    has_person,
    plantae,
    include_non_plants,
  } = scope ?? {};

  // Same filter surface the map used, so the sheet's membership is the
  // place that was clicked. `place_id` resolves server-side against the
  // same partition (granularity + min_photos must travel with it).
  const placeQuery = (page: number) => ({
    page,
    page_size: PLACE_PAGE_SIZE,
    cluster_id,
    cluster_group: supergroup,
    species,
    genus,
    family,
    has_person,
    plantae,
    include_non_plants,
    place_id: selected?.id,
    place_granularity: granularity,
    place_min_photos: showSingles ? 1 : 2,
    sort: 'time' as const,
  });

  useEffect(() => {
    let cancelled = false;
    getLocationGuesses()
      .then((r) => { if (!cancelled) setGuessStats(r); })
      .catch(() => { /* the toggle simply shows no count */ });
    return () => { cancelled = true; };
  }, [guessVersion]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getMapClusters({
      granularity,
      min_photos: showSingles ? 1 : 2,
      guessed: includeGuessed ? 'include' : 'exclude',
      cluster_id,
      cluster_group: supergroup,
      species,
      genus,
      family,
      has_person,
      plantae,
      include_non_plants,
    })
      .then((r) => {
        if (cancelled) return;
        setData(r);
        setError(null);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : 'Could not load map places');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [
    granularity,
    showSingles,
    cluster_id,
    supergroup,
    species,
    genus,
    family,
    has_person,
    plantae,
    include_non_plants,
    includeGuessed,
    guessVersion,
  ]);

  // A different filter/partition invalidates the selected place (its id is
  // an index into that partition). Close the sheet rather than showing a
  // stale list.
  useEffect(() => {
    setSelected(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    granularity,
    showSingles,
    includeGuessed,
    cluster_id,
    supergroup,
    species,
    genus,
    family,
    has_person,
    plantae,
    include_non_plants,
  ]);

  // Sheet contents follow the selected place (and the same filters that
  // produced the marker). Page 0 only; "Load more" appends.
  useEffect(() => {
    if (!selected) {
      setPlaceRows([]);
      setPlaceTotal(0);
      setPlaceError(null);
      return;
    }
    let cancelled = false;
    setPlaceLoading(true);
    setPlaceError(null);
    setPlaceRows([]);
    setPlaceTotal(0);
    getResults(placeQuery(0))
      .then((r: ResultsResponse) => {
        if (cancelled) return;
        setPlaceRows(r.results);
        setPlaceTotal(r.pagination.total);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setPlaceError(e instanceof Error ? e.message : 'Could not load this place');
      })
      .finally(() => {
        if (!cancelled) setPlaceLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    selected?.id,
    granularity,
    showSingles,
    cluster_id,
    supergroup,
    species,
    genus,
    family,
    has_person,
    plantae,
    include_non_plants,
  ]);

  const places = data?.clusters ?? [];
  const points = useMemo(
    () => places.map((p) => [p.latitude, p.longitude] as [number, number]),
    [places],
  );
  const maxCount = useMemo(
    () => places.reduce((max, p) => Math.max(max, p.count), 0),
    [places],
  );

  /** Explicit re-frame, compensating for the one-shot auto-fit. */
  const fitPlaces = () => {
    const map = mapRef.current;
    if (!map || points.length === 0) return;
    const only = points[0];
    if (points.length === 1 && only) {
      map.setView(only, 13, { animate: true });
      return;
    }
    map.fitBounds(points, { padding: [32, 32], maxZoom: 14, animate: true });
  };

  /** Escape hatch: open this place in the full grid (filters + navigate). */
  const openInGrid = () => {
    if (!selected) return;
    if (onOpenInGrid) {
      onOpenInGrid(selected, granularity, showSingles ? 1 : 2);
      return;
    }
    setFilters({
      // The Map tab is independent: opening a place is a fresh browse
      // scope, so drop whatever cluster/taxon filter was left in the URL.
      // The scoped modal keeps its scope by using `onOpenInGrid` instead.
      cluster_id: undefined,
      supergroup: undefined,
      species: undefined,
      genus: undefined,
      family: undefined,
      has_person: undefined,
      plantae: undefined,
      include_non_plants: false,
      near_lat: selected.latitude,
      near_lng: selected.longitude,
      // The radius drives the human-readable title; the exact membership
      // comes from place_id, so a neighbouring place inside this disk
      // cannot leak into the grid.
      near_radius: Math.ceil(selected.radius_m * 1.05),
      place_id: selected.id,
      place_granularity: granularity,
      place_min_photos: showSingles ? 1 : 2,
      page: 0,
    });
    navigate('browse-content');
  };

  const loadMorePlace = async () => {
    if (!selected || placeLoadingMore || placeRows.length >= placeTotal) return;
    setPlaceLoadingMore(true);
    try {
      const page = Math.floor(placeRows.length / PLACE_PAGE_SIZE);
      const r = await getResults(placeQuery(page));
      setPlaceRows((prev) => [...prev, ...r.results]);
      setPlaceTotal(r.pagination.total);
    } catch (e: unknown) {
      setPlaceError(e instanceof Error ? e.message : 'Could not load more photos');
    } finally {
      setPlaceLoadingMore(false);
    }
  };

  const runGuesses = async () => {
    setGuessBusy(true);
    setGuessError(null);
    try {
      const result = await applyLocationGuesses();
      setGuessStats(result);
      setIncludeGuessed(true);
      // Refetch both the stats and the map (the guess set changed).
      setGuessVersion((v) => v + 1);
    } catch (e: unknown) {
      setGuessError(e instanceof Error ? e.message : 'Could not infer locations');
    } finally {
      setGuessBusy(false);
    }
  };

  const clearGuesses = async () => {
    if (!window.confirm('Remove all inferred locations? Real EXIF locations are not affected.')) {
      return;
    }
    setGuessBusy(true);
    setGuessError(null);
    try {
      await clearLocationGuesses();
      setGuessVersion((v) => v + 1);
    } catch (e: unknown) {
      setGuessError(e instanceof Error ? e.message : 'Could not clear inferred locations');
    } finally {
      setGuessBusy(false);
    }
  };

  const initialCenter: [number, number] = initialView
    ? [initialView.lat, initialView.lng]
    : points[0] ?? FALLBACK_CENTER;
  const initialZoom = initialView
    ? initialView.zoom
    : points.length
      ? 12
      : FALLBACK_ZOOM;

  return (
    <section className={variant === 'modal' ? '' : 'px-3 sm:px-4 py-3'} aria-label="Photo map">
      <div className="relative rounded-xl overflow-hidden border border-line2 bg-bg-1">
        <div
          className="relative"
          style={
            variant === 'modal'
              ? { height: 'min(62vh, 640px)', minHeight: 360 }
              : { height: 'calc(100vh - 148px)', minHeight: 420 }
          }
        >
          <SafeMapContainer
            ref={mapRef}
            center={initialCenter}
            zoom={initialZoom}
            scrollWheelZoom
            className="h-full w-full"
            attributionControl
          >
            <TileLayer attribution={OSM_ATTRIBUTION} url={OSM_TILE_URL} />
            <FitToPlaces points={points} singlePointZoom={13} initialView={initialView} />
            {variant === 'page' && <PersistView />}
            <FocusPlace place={selected} />
            {places.map((place) => {
              const isSelected = selected?.id === place.id;
              return (
              <CircleMarker
                key={place.id}
                center={[place.latitude, place.longitude]}
                radius={radiusFor(place.count, maxCount) + (isSelected ? 3 : 0)}
                pathOptions={{
                  color: '#2f7d4f',
                  weight: isSelected ? 4 : 2,
                  fillColor: '#3fae6f',
                  fillOpacity: isSelected ? 0.85 : 0.55,
                }}
                eventHandlers={{ click: () => setSelected(place) }}
              >
                <Tooltip direction="top">
                  <div className="font-sans text-[12px] leading-snug">
                    <div className="font-semibold">
                      {place.count.toLocaleString()} photo{place.count === 1 ? '' : 's'}
                    </div>
                    {(place.start || place.end) && (
                      <div className="text-fg-1">
                        {fmtDate(place.start)}
                        {place.end && place.end !== place.start ? ` – ${fmtDate(place.end)}` : ''}
                      </div>
                    )}
                    {place.visits > 1 && <div className="text-fg-1">{place.visits} visits</div>}
                    <div className="text-fg-2">
                      {isSelected ? 'Selected' : 'Click to see photos'}
                    </div>
                  </div>
                </Tooltip>
              </CircleMarker>
              );
            })}
          </SafeMapContainer>

          {/* Controls + counts. Collapsed to a compact pill on phones —
              the full panel would cover most of the map. Desktop starts
              open and can be collapsed too. */}
          {!panelOpen && (
            <button
              type="button"
              className="absolute top-3 right-3 z-[550] flex items-center gap-1.5 rounded-lg border border-line2 bg-bg/95 backdrop-blur px-2.5 py-1.5 text-[11px] shadow-lg"
              onClick={() => setPanelOpen(true)}
              aria-label="Show map controls"
            >
              <span className="font-semibold text-fg">Places</span>
              {data && (
                <span className="text-fg-1">
                  {data.cluster_count.toLocaleString()} · {data.photo_count.toLocaleString()}
                </span>
              )}
              <span aria-hidden="true" className="text-fg-2">▾</span>
            </button>
          )}
          {panelOpen && (
          <div className="absolute z-[500] left-3 right-3 top-20 sm:left-auto sm:top-3 sm:right-3 sm:max-w-sm rounded-lg border border-line2 bg-bg/95 backdrop-blur px-3 py-2.5 shadow-lg">
            <div className="flex items-baseline gap-2">
              <h2 className="text-fg text-sm font-semibold">Places</h2>
              {loading && <span className="text-fg-2 text-[11px]">loading…</span>}
              <button
                type="button"
                className="ml-auto text-fg-2 hover:text-fg text-base leading-none px-1"
                onClick={() => setPanelOpen(false)}
                aria-label="Collapse map controls"
                title="Collapse"
              >
                −
              </button>
            </div>
            {error ? (
              <p className="text-score-red text-xs mt-1">{error}</p>
            ) : (
              <p className="text-fg-1 text-xs mt-1">
                {data ? (
                  <>
                    <strong className="text-fg-0">{data.cluster_count.toLocaleString()}</strong>{' '}
                    place{data.cluster_count === 1 ? '' : 's'} ·{' '}
                    <strong className="text-fg-0">{data.photo_count.toLocaleString()}</strong>{' '}
                    photo{data.photo_count === 1 ? '' : 's'}
                    {data.guessed_photos > 0 && includeGuessed && (
                      <span className="text-fg-2">
                        {' '}· {data.guessed_photos.toLocaleString()} positions inferred
                      </span>
                    )}
                    {data.unlocated > 0 && (
                      <span className="text-fg-2">
                        {' '}· {data.unlocated.toLocaleString()} without GPS
                      </span>
                    )}
                  </>
                ) : (
                  'Grouping geotagged photos…'
                )}
              </p>
            )}

            <div className="mt-2 flex flex-wrap items-center gap-2">
              <div className="inline-flex p-[2px] bg-bg-2 border border-line2 rounded-md" role="group" aria-label="Place size">
                {([
                  ['fine', '150 m', 'Venue radius (~150 m)'],
                  ['wide', '1 km', 'Area radius (~1 km)'],
                  ['region', '10 km', 'Region radius (~10 km)'],
                ] as const).map(([value, label, title]) => (
                  <button
                    key={value}
                    type="button"
                    aria-pressed={granularity === value}
                    title={title}
                    className={`px-2 py-1 rounded text-[11px] border border-transparent cursor-pointer ${
                      granularity === value
                        ? 'bg-accent text-bg'
                        : 'text-fg-1 hover:text-fg hover:border-[#444]'
                    }`}
                    onClick={() => setGranularity(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <button
                type="button"
                className="px-2 py-1 rounded text-[11px] border border-line2 text-fg-1 hover:text-fg hover:border-accent"
                onClick={fitPlaces}
                title="Fit the map to every place"
              >
                Fit
              </button>

              <label className="inline-flex items-center gap-1.5 text-[11px] text-fg-1 cursor-pointer">
                <input
                  type="checkbox"
                  className="accent-[rgb(var(--c-accent))]"
                  checked={showSingles}
                  onChange={(e) => setShowSingles(e.target.checked)}
                />
                Single-photo places
                {data && data.singleton_places > 0 && (
                  <span className="text-fg-2">({data.singleton_places.toLocaleString()})</span>
                )}
              </label>
            </div>

            <div className="mt-2 pt-2 border-t border-line flex flex-wrap items-center gap-2 text-[11px]">
              <label className="inline-flex items-center gap-1.5 text-fg-1 cursor-pointer">
                <input
                  type="checkbox"
                  className="accent-[rgb(var(--c-accent))]"
                  checked={includeGuessed}
                  onChange={(e) => setIncludeGuessed(e.target.checked)}
                />
                Inferred locations
                {guessStats && guessStats.total > 0 && (
                  <span className="text-fg-2">({guessStats.total.toLocaleString()})</span>
                )}
              </label>
              <div className="ml-auto flex items-center gap-2">
                {guessError ? (
                  <span className="text-score-red">{guessError}</span>
                ) : guessBusy ? (
                  <span className="text-fg-2">working…</span>
                ) : (
                  <>
                    <button
                      type="button"
                      className="text-accent hover:text-accent-hover underline disabled:opacity-50"
                      onClick={runGuesses}
                      disabled={guessBusy}
                      title="Interpolate coordinates for photos without GPS from GPS-tagged photos within ±15 minutes and 1 km. Local only — Immich is not modified."
                    >
                      {guessStats && guessStats.total > 0 ? 'Recompute' : 'Infer from nearby GPS'}
                    </button>
                    {guessStats && guessStats.total > 0 && (
                      <button
                        type="button"
                        className="text-fg-2 hover:text-fg underline"
                        onClick={clearGuesses}
                        disabled={guessBusy}
                        title="Delete every inferred coordinate. Real EXIF locations are untouched."
                      >
                        Clear
                      </button>
                    )}
                  </>
                )}
              </div>
            </div>

            {scopeLabel && (
              <div className="mt-2 pt-2 border-t border-line text-[11px] text-fg-1">
                Showing: <span className="text-fg-0">{scopeLabel}</span>
              </div>
            )}
          </div>
          )}

          {!loading && !error && places.length === 0 && (
            <div className="absolute inset-0 z-[400] flex items-center justify-center pointer-events-none">
              <div className="rounded-lg border border-line2 bg-bg/95 backdrop-blur px-4 py-3 text-center max-w-sm">
                <p className="text-fg text-sm font-semibold">No geotagged photos</p>
                <p className="text-fg-1 text-xs mt-1">
                  The map needs EXIF GPS on the photos. Run “Backfill EXIF” on the
                  Run page to pull coordinates from Immich.
                </p>
              </div>
            </div>
          )}

          {/* Place sheet — photos open here, over the map. The full grid is
              still one click away ("Open in grid"). */}
          {selected && (
            <div className="absolute inset-x-0 bottom-6 z-[600] max-h-[40vh] sm:max-h-[47vh] flex flex-col rounded-t-xl border border-line2 bg-bg/95 backdrop-blur shadow-2xl">
              <div className="flex items-start gap-3 px-3 pt-2 pb-2 border-b border-line shrink-0">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-baseline gap-x-2">
                    <span className="text-fg text-sm font-semibold">
                      {selected.count.toLocaleString()} photo
                      {selected.count === 1 ? '' : 's'}
                    </span>

                  </div>
                  <div className="text-[11px] text-fg-2">
                    {fmtDate(selected.start)}
                    {selected.end && selected.end !== selected.start
                      ? ` – ${fmtDate(selected.end)}`
                      : ''}
                    {selected.visits > 1 ? ` · ${selected.visits} visits` : ''}
                    {placeLoading ? ' · loading…' : ''}
                  </div>
                </div>
                <div className="ml-auto flex items-center gap-2 shrink-0">
                  <button
                    type="button"
                    className="text-[11px] text-accent hover:text-accent-hover underline whitespace-nowrap"
                    onClick={openInGrid}
                  >
                    Open in grid →
                  </button>
                  <button
                    type="button"
                    aria-label="Close place"
                    title="Close"
                    className="text-fg-1 hover:text-fg text-lg leading-none px-1"
                    onClick={() => setSelected(null)}
                  >
                    ×
                  </button>
                </div>
              </div>

              {placeError ? (
                <p className="px-3 py-4 text-xs text-score-red">{placeError}</p>
              ) : placeRows.length === 0 && !placeLoading ? (
                <p className="px-3 py-4 text-xs text-fg-2">No photos match this place.</p>
              ) : (
                <>
                  {/* Filmstrip on phones (one row, swipe sideways), grid
                      on larger screens. Same nodes either way, so no
                      duplicated image requests. */}
                  <div className="flex gap-1 p-2 overflow-x-auto sm:grid sm:grid-cols-6 lg:grid-cols-8 sm:overflow-y-auto sm:overflow-x-hidden">
                    {placeRows.map((row) => (
                      <button
                        key={row.asset_id}
                        type="button"
                        title={
                          row.location_guessed
                            ? `${row.original_file_name} · location inferred from nearby GPS photos`
                            : row.original_file_name
                        }
                        className="relative h-24 w-24 shrink-0 overflow-hidden rounded bg-bg-2 border border-line2/60 focus:outline-none focus:ring-2 focus:ring-accent sm:h-auto sm:w-auto sm:aspect-square"
                        onClick={() =>
                          dispatch({ type: 'OPEN_MODAL', assetId: row.asset_id })
                        }
                      >
                        <img
                          src={thumbUrl(row.asset_id, 'thumbnail')}
                          alt=""
                          loading="lazy"
                          className="h-full w-full object-cover"
                        />
                        {row.location_guessed ? (
                          <span
                            className="absolute top-1 right-1 h-2 w-2 rounded-full bg-[#f59e0b] ring-1 ring-black/50"
                            aria-hidden="true"
                          />
                        ) : null}
                      </button>
                    ))}
                  </div>
                  {placeRows.length < placeTotal && (
                    <div className="px-2 py-2 text-center border-t border-line shrink-0">
                      <button
                        type="button"
                        className="text-[11px] text-accent hover:text-accent-hover underline disabled:opacity-50"
                        onClick={loadMorePlace}
                        disabled={placeLoadingMore}
                      >
                        {placeLoadingMore
                          ? 'Loading…'
                          : `Load more (${(placeTotal - placeRows.length).toLocaleString()} remaining)`}
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {variant === 'page' && (
      <p className="hidden sm:block text-fg-2 text-[11px] mt-2">
        Photos are grouped by distance only — the same place visited in different
        months stays one marker. Click a marker to open its photos.
        {' '}A small amber dot in the photo strip marks individual photos whose
        position was inferred from nearby GPS shots; a real EXIF fix always wins.
      </p>
      )}
    </section>
  );
}

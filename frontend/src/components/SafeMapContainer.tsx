import React, { forwardRef, useEffect, useRef } from 'react';
import { MapContainer, type MapContainerProps } from 'react-leaflet';
import type { Map as LeafletMap } from 'leaflet';

/**
 * A `MapContainer` that survives being unmounted while a zoom animation is
 * running. Ported from the Solar app (`SafeMapContainer.tsx`), where this
 * was a real Sentry crash: Leaflet 1.9.4's `_animateZoom` schedules a
 * 250 ms `_onZoomTransitionEnd` timer, but `Map.remove()` only cancels
 * flyTo/pan — not zoom animations. If the user taps a marker and we
 * navigate away within that window, the timer fires after `_mapPane` is
 * deleted and throws `Cannot read properties of undefined
 * (reading '_leaflet_pos')`.
 *
 * Finishing the transition before Leaflet tears the map down turns the
 * timer into a no-op. It has to happen from *outside* `MapContainer`:
 * React runs unmount cleanups parent-first, so a child of the map would
 * already find `_mapPane` deleted.
 */
interface LeafletZoomInternals {
  _animatingZoom?: boolean;
  _onZoomTransitionEnd?: () => void;
  _mapPane?: HTMLElement;
}

export const SafeMapContainer = forwardRef<LeafletMap | null, MapContainerProps>(
  function SafeMapContainer(props, forwardedRef) {
    // React detaches refs *before* running unmount cleanups, so a plain
    // `ref` is already null by the time the effect below fires.
    const instanceRef = useRef<LeafletMap | null>(null);

    useEffect(
      () => () => {
        const map = instanceRef.current as unknown as LeafletZoomInternals | null;
        if (map?._animatingZoom && map._mapPane) {
          map._onZoomTransitionEnd?.();
        }
      },
      [],
    );

    return (
      <MapContainer
        {...props}
        ref={(map: LeafletMap | null) => {
          // `null` is the detach, not a new map — it must not clobber the
          // instance the cleanup needs.
          if (map) instanceRef.current = map;
          if (typeof forwardedRef === 'function') {
            forwardedRef(map);
          } else if (forwardedRef) {
            (forwardedRef as React.MutableRefObject<LeafletMap | null>).current = map;
          }
        }}
      />
    );
  },
);

export default SafeMapContainer;

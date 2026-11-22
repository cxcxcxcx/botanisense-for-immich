import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import type { Detection } from '../api/types';
import { mergeDetectionsForDisplay } from '../lib/bbox';

// Color-code by label so the user can tell at a glance what YOLO
// saw. Person-flavored boxes (person / face / human) get the accent
// color (high signal); hands get yellow (visible-but-not-person);
// plants/flowers default to green.
const COLOR_FOR: Record<string, string> = {
  person: '#ff7a45',
  face:   '#5fa8ff',
  human:  '#5fc8c8',
  hand:   '#ffd84d',
  plant:  '#5f5',
  leaf:   '#5f5',
  flower: '#5f5',
  tree:   '#5f5',
  shrub:  '#5f5',
  succulent: '#5f5',
  cactus: '#5f5',
  herb: '#5f5',
  grass: '#5f5',
  stem: '#5f5',
  branch: '#5f5',
  bloom: '#5f5',
  petal: '#5f5',
  seedling: '#5f5',
};
const DEFAULT_COLOR = '#888';

interface Props {
  // Image URL (same-origin or CORS-enabled; we don't read pixels).
  src: string;
  // YOLO boxes with x/y/w/h as 0..1 fractions of the ORIGINAL image.
  detections?: Detection[] | null;
  // 'contain' shows the whole image (letterboxed) — used by the
  // asset-detail modal. 'cover' fills the element and crops — used by
  // the 4:5 grid cards. Either way the image and boxes are drawn on
  // the same canvas in the image's coordinate space, so they always
  // line up.
  fit: 'contain' | 'cover';
  // For fit="contain": cap the display height in CSS px.
  maxHeight?: number;
  // Hint for the browser to defer offscreen images.
  lazy?: boolean;
  className?: string;
  ariaLabel?: string;
  // Fires if the image fails to decode (so callers can swap to a
  // fallback URL, e.g. fullsize → preview).
  onError?: () => void;
  // Boxes whose `id` is in the set are always drawn, regardless of
  // score. This is the "plants in this image" signal: the asset-detail
  // modal passes the detection IDs that storage.compute_plant_attribution
  // accepted as justified, so plausibly-low-YOLO boxes are still drawn
  // when the whole-image plantae score confirms the bbox.
  highlightDetectionIds?: number[] | null;
  // Draw boxes with score >= this threshold (default 0.50). The asset
  // detail lowers it to 0.25 for the "show all boxes" mode.
  minScore?: number;
  // Click a box on the canvas to select it (asset detail → Similar).
  onDetectionClick?: (detectionId: number) => void;
}

// A <canvas> that draws BOTH the photo and its YOLO detection boxes.
// Drawing them in one pass on the same surface is the robust fix for
// the old "box doesn't line up with the image" bug: there is no
// separate overlay to position, no letterbox coordinate math, and no
// two-element alignment to get wrong — the image and the boxes share
// one coordinate system.
export function BboxImage({
  src,
  detections,
  fit,
  maxHeight = 220,
  lazy = false,
  className,
  ariaLabel,
  onError,
  highlightDetectionIds,
  minScore,
  onDetectionClick,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const onErrorRef = useRef(onError);
  useLayoutEffect(() => { onErrorRef.current = onError; });

  // Load the image. Keep the decoded image in a ref; put its natural
  // dimensions in state so the layout/draw effects run.
  useEffect(() => {
    setNatural(null);
    imgRef.current = null;
    const img = new Image();
    if (lazy) img.loading = 'lazy';
    img.onload = () => {
      imgRef.current = img;
      setNatural({ w: img.naturalWidth, h: img.naturalHeight });
    };
    img.onerror = () => { onErrorRef.current?.(); };
    img.src = src;
    return () => { img.onload = null; img.onerror = null; };
  }, [src, lazy]);

  // CONTAIN mode: set the canvas's intrinsic (width/height attributes)
  // to the image's natural size. The browser then lays it out exactly
  // like an <img> — `max-h`/`max-w-full` scale it down while keeping
  // the aspect ratio. We must NOT change canvas.width/height after
  // this (that would alter the intrinsic size and cause layout
  // feedback). The backing store is the natural resolution, which is
  // already high-DPI when displayed small, so lines stay crisp. COVER
  // mode leaves sizing to CSS (w-full h-full) and the draw effect
  // sizes the backing store to the displayed rectangle.
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !natural || fit !== 'contain') return;
    if (canvas.width !== natural.w || canvas.height !== natural.h) {
      canvas.width = natural.w;
      canvas.height = natural.h;
    }
  }, [natural, fit]);

  // Draw the image + boxes, and keep them redrawn on resize.
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !natural) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      // In cover mode the canvas fills its parent; size the backing
      // store to the displayed rect × devicePixelRatio and draw in
      // CSS pixels. In contain mode the backing store is the natural
      // image size (set above); we draw in image pixels and scale
      // line/text sizes by natural.h/rect.height so they are a
      // constant thickness in CSS px.
      const dpr = window.devicePixelRatio || 1;
      let s: number;          // scale from "drawing units" to backing px
      let cw: number, ch: number;  // backing store size
      if (fit === 'cover') {
        cw = Math.round(rect.width * dpr);
        ch = Math.round(rect.height * dpr);
        if (canvas.width !== cw) canvas.width = cw;
        if (canvas.height !== ch) canvas.height = ch;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        s = 1;  // draw in CSS px
      } else {
        cw = natural.w;
        ch = natural.h;
        // canvas.width/height already natural from the layout effect.
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        // 1 drawing unit = 1 image pixel. Scale factor for things we
        // want constant in CSS px (strokes, text): image px per CSS
        // px. rect.height is the displayed CSS height.
        s = natural.h / rect.height;
      }

      ctx.clearRect(0, 0, fit === 'cover' ? rect.width : cw, fit === 'cover' ? rect.height : ch);

      // Draw the image.
      //   cover  → scale to fill the displayed rect, center-crop the
      //            overflow; draw coords are CSS px (dpr transform set).
      //   contain→ the backing store IS the natural image, so draw at
      //            (0,0,natural.w,natural.h). The browser scales the
      //            whole canvas element down via CSS (intrinsic aspect
      //            ratio + max-h/max-w), so there are no letterbox
      //            bars inside the canvas and no offset math.
      let offX: number, offY: number, drawW: number, drawH: number;
      if (fit === 'cover') {
        const scaleImg = Math.max(rect.width / natural.w, rect.height / natural.h);
        drawW = natural.w * scaleImg;
        drawH = natural.h * scaleImg;
        offX = (rect.width - drawW) / 2;
        offY = (rect.height - drawH) / 2;
      } else {
        drawW = natural.w;
        drawH = natural.h;
        offX = 0;
        offY = 0;
      }
      ctx.drawImage(img, offX, offY, drawW, drawH);

      // Map a 0..1 fraction of the ORIGINAL image to drawing units.
      // For cover this places the box over the (possibly cropped)
      // visible image; boxes in the cropped-away region fall off the
      // canvas and are clipped. For contain it's just px*natural.
      const mapX = (px: number) => offX + px * drawW;
      const mapY = (py: number) => offY + py * drawH;

      const threshold = minScore ?? 0.50;
      const highlight = highlightDetectionIds?.length
        ? new Set(highlightDetectionIds)
        : null;
      const visible = (detections ?? []).filter(
        d => (highlight?.has(d.id) ?? false) || d.score >= threshold,
      );
      // A bouquet often produces one YOLO box per flower head. Draw
      // adjacent same-label boxes as one union region, with the member
      // count in the label. The raw detections remain in the debug list.
      const display = mergeDetectionsForDisplay(visible);
      const stroke = 2 * s;
      const fontSize = 10 * s;
      const pad = 2 * s;
      ctx.lineWidth = stroke;
      ctx.font = `600 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.textBaseline = 'top';

      for (const d of display) {
        const color = COLOR_FOR[d.label] ?? DEFAULT_COLOR;
        const bx = mapX(d.x);
        const by = mapY(d.y);
        const bw = d.w * drawW;
        const bh = d.h * drawH;
        ctx.strokeStyle = color;
        ctx.strokeRect(bx, by, bw, bh);

        const count = d.group_count > 1 ? ` ×${d.group_count}` : '';
        const label = `${d.label}${count} ${Math.round(d.score * 100)}%`;
        const tw = ctx.measureText(label).width;
        const chipH = fontSize + pad * 2;
        let cx = Math.max(0, bx);
        let cy = Math.max(0, by);
        const maxW = (fit === 'cover' ? rect.width : cw);
        let cwChip = tw + pad * 2;
        if (cx + cwChip > maxW) cwChip = Math.max(0, maxW - cx);
        ctx.fillStyle = color;
        ctx.fillRect(cx, cy, cwChip, chipH);
        ctx.fillStyle = '#000';
        ctx.fillText(label, cx + pad, cy + pad);
      }
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [natural, detections, fit, highlightDetectionIds, minScore]);

  const handleClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (!onDetectionClick || !natural) return;
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    let nx: number;
    let ny: number;
    if (fit === 'cover') {
      const scale = Math.max(rect.width / natural.w, rect.height / natural.h);
      const drawW = natural.w * scale;
      const drawH = natural.h * scale;
      const offX = (rect.width - drawW) / 2;
      const offY = (rect.height - drawH) / 2;
      nx = (px - offX) / drawW;
      ny = (py - offY) / drawH;
    } else {
      nx = px / rect.width;
      ny = py / rect.height;
    }
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
    const threshold = minScore ?? 0.50;
    const highlight = highlightDetectionIds?.length
      ? new Set(highlightDetectionIds)
      : null;
    const visible = (detections ?? []).filter(
      d => (highlight?.has(d.id) ?? false) || d.score >= threshold,
    );
    const hits = mergeDetectionsForDisplay(visible).filter(
      d => nx >= d.x && nx <= d.x + d.w
        && ny >= d.y && ny <= d.y + d.h,
    );
    if (hits.length === 0) return;
    hits.sort((a, b) => a.w * a.h - b.w * b.h);
    const first = hits[0];
    if (first) onDetectionClick(first.id);
  };

  const baseClass = fit === 'cover'
    ? 'block w-full h-full'
    : 'block max-w-full h-auto w-auto';
  const style = fit === 'contain' ? { maxHeight: `${maxHeight}px` } : undefined;

  return (
    <canvas
      ref={canvasRef}
      // Hide until the image loads so the default 300×150 canvas never
      // flashes (contain mode has no intrinsic size before then).
      className={`${baseClass}${natural ? '' : ' opacity-0'}${className ? ` ${className}` : ''}`}
      style={onDetectionClick ? { ...style, cursor: 'pointer' } : style}
      role="img"
      aria-label={ariaLabel}
      onClick={handleClick}
    />
  );
}

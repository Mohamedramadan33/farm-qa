import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  APIProvider,
  AdvancedMarker,
  AdvancedMarkerAnchorPoint,
  CollisionBehavior,
  InfoWindow,
  Map as GoogleMap,
  Pin,
  useAdvancedMarkerRef,
  useMap,
} from '@vis.gl/react-google-maps';
import { MarkerClusterer } from '@googlemaps/markerclusterer';
import { Layers, MapPin, Plus, Trash2 } from 'lucide-react';
import { bufferPolygonMeters, polygonCentroid, shiftPolygonMeters } from '../lib/geometry';
import { worldPxToLonLat } from '../services/satellite/tileFetcher';
import type {
  PointLonLat,
  SatelliteAnalysisResult,
  TreeClassName,
  TreeDetectionBox,
  TreeHealthStatus,
} from '../types/qa';

const GMP_ATTRIBUTION_IDS = ['gmp_mcp_codeassist_v1_aistudio'];
const DEMO_MAP_ID = 'DEMO_MAP_ID';

interface GoogleMapsTreeInspectorProps {
  farmId: string;
  poly: PointLonLat[];
  offsetX_m: number;
  offsetY_m: number;
  inwardBufferM: number;
  satRes?: SatelliteAnalysisResult;
  boxes: TreeDetectionBox[];
  editTreesMode: boolean;
  onUpdateTree?: (
    treeId: number,
    patch: Partial<Pick<TreeDetectionBox, 'className' | 'classId' | 'health'>>
  ) => void;
  onDeleteTree?: (treeId: number) => void;
  onAddTreeAtLatLng?: (lat: number, lng: number) => void;
}

interface TreePinMarkerProps {
  tree: TreeDetectionBox;
  position: { lat: number; lng: number };
  zIndex: number;
  selected: boolean;
  onSelect: (tree: TreeDetectionBox, marker: google.maps.marker.AdvancedMarkerElement) => void;
  onRegisterMarker: (
    treeId: number,
    marker: google.maps.marker.AdvancedMarkerElement | null
  ) => void;
}

const TreePinMarker: React.FC<TreePinMarkerProps> = ({
  tree,
  position,
  zIndex,
  selected,
  onSelect,
  onRegisterMarker,
}) => {
  const [markerRef, marker] = useAdvancedMarkerRef();

  useEffect(() => {
    onRegisterMarker(tree.id, marker);
    return () => onRegisterMarker(tree.id, null);
  }, [tree.id, marker, onRegisterMarker]);

  const pinPalette =
    tree.health === 'dead'
      ? { background: '#dc2626', borderColor: '#7f1d1d', glyphColor: '#fef2f2' }
      : tree.health === 'stressed'
        ? { background: '#d97706', borderColor: '#78350f', glyphColor: '#fffbeb' }
        : tree.className === 'palm_tree'
          ? { background: '#f59e0b', borderColor: '#92400e', glyphColor: '#ffffff' }
          : tree.className === 'windbreak_tree'
            ? { background: '#0891b2', borderColor: '#164e63', glyphColor: '#ffffff' }
            : { background: '#059669', borderColor: '#064e3b', glyphColor: '#ffffff' };

  return (
    <AdvancedMarker
      ref={markerRef}
      position={position}
      zIndex={selected ? 2000 : zIndex}
      anchorPoint={AdvancedMarkerAnchorPoint.BOTTOM}
      collisionBehavior={CollisionBehavior.OPTIONAL_AND_HIDES_LOWER_PRIORITY}
      title={`Tree #${tree.id} (${tree.className}, ${Math.round(tree.score * 100)}%)`}
      onClick={() => {
        if (marker) onSelect(tree, marker);
      }}
    >
      <Pin
        background={selected ? '#4f46e5' : pinPalette.background}
        borderColor={selected ? '#ffffff' : pinPalette.borderColor}
        glyphColor={pinPalette.glyphColor}
        scale={selected ? 1.1 : 0.78}
      />
    </AdvancedMarker>
  );
};

interface FarmBoundaryPolygonsProps {
  rawPoly: PointLonLat[];
  calibratedPoly: PointLonLat[];
  clippedPoly: PointLonLat[];
  hasCalibrationShift: boolean;
  hasInwardBuffer: boolean;
}

const FarmBoundaryPolygons: React.FC<FarmBoundaryPolygonsProps> = ({
  rawPoly,
  calibratedPoly,
  clippedPoly,
  hasCalibrationShift,
  hasInwardBuffer,
}) => {
  const map = useMap();

  useEffect(() => {
    if (!map || rawPoly.length < 3) return;

    const overlays: google.maps.Polygon[] = [];
    const bounds = new google.maps.LatLngBounds();

    const toLatLngPath = (ring: PointLonLat[]) =>
      ring.map(([lng, lat]) => {
        const pt = { lat, lng };
        bounds.extend(pt);
        return pt;
      });

    // 1. Raw Kobo GPS Boundary (Amber outline)
    const rawOverlay = new google.maps.Polygon({
      map,
      paths: toLatLngPath(rawPoly),
      strokeColor: '#f59e0b',
      strokeOpacity: hasCalibrationShift || hasInwardBuffer ? 0.65 : 0.95,
      strokeWeight: hasCalibrationShift || hasInwardBuffer ? 1.8 : 2.6,
      fillColor: '#f59e0b',
      fillOpacity: hasCalibrationShift || hasInwardBuffer ? 0.04 : 0.1,
      clickable: false,
    });
    overlays.push(rawOverlay);

    // 2. Calibrated / Shifted Boundary (Cyan) if user shifted X/Y meters
    if (hasCalibrationShift) {
      const calOverlay = new google.maps.Polygon({
        map,
        paths: toLatLngPath(calibratedPoly),
        strokeColor: '#06b6d4',
        strokeOpacity: 0.9,
        strokeWeight: 2.2,
        fillOpacity: 0,
        clickable: false,
      });
      overlays.push(calOverlay);
    }

    // 3. Inward-Buffered Orchard Mask Boundary (Emerald)
    if (hasInwardBuffer) {
      const clipOverlay = new google.maps.Polygon({
        map,
        paths: toLatLngPath(clippedPoly),
        strokeColor: '#10b981',
        strokeOpacity: 0.95,
        strokeWeight: 2.4,
        fillColor: '#10b981',
        fillOpacity: 0.08,
        clickable: false,
      });
      overlays.push(clipOverlay);
    }

    if (!bounds.isEmpty()) {
      map.fitBounds(bounds, { top: 28, right: 28, bottom: 28, left: 28 });
    }

    return () => {
      for (const polyOverlay of overlays) {
        polyOverlay.setMap(null);
      }
    };
  }, [map, rawPoly, calibratedPoly, clippedPoly, hasCalibrationShift, hasInwardBuffer]);

  return null;
};

interface MarkerClusterControllerProps {
  enabled: boolean;
  markerRegistry: Record<number, google.maps.marker.AdvancedMarkerElement>;
  markerCount: number;
}

const MarkerClusterController: React.FC<MarkerClusterControllerProps> = ({
  enabled,
  markerRegistry,
  markerCount,
}) => {
  const map = useMap();

  useEffect(() => {
    if (!map || !enabled) return;
    const instances = Object.values(markerRegistry);
    if (instances.length === 0) return;

    const clusterer = new MarkerClusterer({ map, markers: instances });
    return () => {
      clusterer.clearMarkers();
    };
  }, [map, enabled, markerRegistry, markerCount]);

  return null;
};

export const GoogleMapsTreeInspector: React.FC<GoogleMapsTreeInspectorProps> = ({
  farmId,
  poly,
  offsetX_m,
  offsetY_m,
  inwardBufferM,
  satRes,
  boxes,
  editTreesMode,
  onUpdateTree,
  onDeleteTree,
  onAddTreeAtLatLng,
}) => {
  const apiKey = (import.meta.env.VITE_GOOGLE_MAPS_API_KEY || '').trim();
  const [mapTypeId, setMapTypeId] = useState<'hybrid' | 'satellite'>('hybrid');
  const [clusterEnabled, setClusterEnabled] = useState<boolean>(false);
  const [selectedTreeId, setSelectedTreeId] = useState<number | null>(null);
  const [selectedMarker, setSelectedMarker] =
    useState<google.maps.marker.AdvancedMarkerElement | null>(null);

  // Registry uses Record<number, ...> rather than JS Map so the name Map is never shadowed
  const [markerRegistry, setMarkerRegistry] = useState<
    Record<number, google.maps.marker.AdvancedMarkerElement>
  >({});

  useEffect(() => {
    setSelectedTreeId(null);
    setSelectedMarker(null);
  }, [farmId]);

  useEffect(() => {
    setClusterEnabled(boxes.length > 75);
  }, [farmId, boxes.length]);

  const calibratedPoly = useMemo(
    () => shiftPolygonMeters(poly, offsetX_m, offsetY_m),
    [poly, offsetX_m, offsetY_m]
  );
  const clippedPoly = useMemo(
    () =>
      Math.abs(inwardBufferM) >= 0.05
        ? bufferPolygonMeters(calibratedPoly, inwardBufferM)
        : calibratedPoly,
    [calibratedPoly, inwardBufferM]
  );

  const [cLng, cLat] = useMemo(() => polygonCentroid(calibratedPoly), [calibratedPoly]);
  const defaultCenter = useMemo(() => ({ lat: cLat, lng: cLng }), [cLat, cLng]);

  // Project any pixel-only boxes onto lat/lng using satRes geotransform
  const geoTrees = useMemo(() => {
    const projected = boxes
      .map((tree) => {
        if (tree.lat !== undefined && tree.lon !== undefined) {
          return { tree, lat: tree.lat, lng: tree.lon };
        }
        if (satRes?.worldOriginX !== undefined && satRes?.worldOriginY !== undefined) {
          const [lon, lat] = worldPxToLonLat(
            satRes.worldOriginX + tree.cx,
            satRes.worldOriginY + tree.cy,
            satRes.zoomUsed
          );
          return { tree, lat, lng: lon };
        }
        return null;
      })
      .filter((item): item is { tree: TreeDetectionBox; lat: number; lng: number } => item !== null);

    // Sort northernmost first -> southern markers receive higher base zIndex
    projected.sort((a, b) => b.lat - a.lat);
    return projected;
  }, [boxes, satRes?.worldOriginX, satRes?.worldOriginY, satRes?.zoomUsed]);

  const selectedTree = useMemo(
    () => boxes.find((b) => b.id === selectedTreeId) || null,
    [boxes, selectedTreeId]
  );

  const handleRegisterMarker = useCallback(
    (treeId: number, marker: google.maps.marker.AdvancedMarkerElement | null) => {
      setMarkerRegistry((prev) => {
        if (marker) {
          if (prev[treeId] === marker) return prev;
          return { ...prev, [treeId]: marker };
        }
        if (!(treeId in prev)) return prev;
        const copy = { ...prev };
        delete copy[treeId];
        return copy;
      });
    },
    []
  );

  const handleSelectTree = useCallback(
    (tree: TreeDetectionBox, marker: google.maps.marker.AdvancedMarkerElement) => {
      setSelectedTreeId(tree.id);
      setSelectedMarker(marker);
    },
    []
  );

  const classLabelAr: Record<TreeClassName, string> = {
    orchard_tree: 'شجرة بستان (Orchard)',
    palm_tree: 'نخيل (Palm)',
    windbreak_tree: 'مصدات رياح (Windbreak)',
  };

  if (!apiKey) {
    return (
      <div className="min-h-[340px] rounded-lg border border-amber-200 bg-amber-50 p-6 flex flex-col items-center justify-center text-center space-y-2">
        <MapPin className="w-6 h-6 text-amber-700" />
        <p className="text-xs font-semibold text-amber-950">
          Google Maps Platform API key غير مهيأ بعد في بيئة المتصفح (VITE_GOOGLE_MAPS_API_KEY).
        </p>
        <p className="text-[11px] text-amber-800 max-w-md">
          يمكنك استمرار التدقيق عبر أوضاع Google / Esri / Side-by-Side أو إضافة المفتاح وإعادة تشغيل الخادم.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Map Controls Strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
        <div className="inline-flex items-center gap-1 p-0.5 bg-slate-100 rounded-md">
          <button
            type="button"
            onClick={() => setMapTypeId('hybrid')}
            className={`inline-flex items-center gap-1 px-2 py-1 rounded cursor-pointer ${
              mapTypeId === 'hybrid'
                ? 'bg-white text-slate-900 font-semibold shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-3 h-3" />
            <span>Hybrid (Labels)</span>
          </button>
          <button
            type="button"
            onClick={() => setMapTypeId('satellite')}
            className={`px-2 py-1 rounded cursor-pointer ${
              mapTypeId === 'satellite'
                ? 'bg-white text-slate-900 font-semibold shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <span>Satellite Clean</span>
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-slate-600">
          {boxes.length > 25 && (
            <label className="inline-flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={clusterEnabled}
                onChange={(e) => setClusterEnabled(e.target.checked)}
                className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
              />
              <span>تجميع العلامات ({boxes.length})</span>
            </label>
          )}
          <span className="inline-flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-xs border-2 border-amber-500 bg-amber-500/20" />
            Kobo GPS
          </span>
          {Math.abs(inwardBufferM) >= 0.05 && (
            <span className="inline-flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-xs border-2 border-emerald-500 bg-emerald-500/20" />
              Buffer ({inwardBufferM}m)
            </span>
          )}
        </div>
      </div>

      {editTreesMode && (
        <div className="px-2.5 py-1.5 rounded-md bg-indigo-50 border border-indigo-200 text-[11px] text-indigo-950 flex items-center gap-1.5">
          <Plus className="w-3.5 h-3.5 text-indigo-700 shrink-0" />
          <span>وضع تعديل الخريطة التفاعلية: اضغط على أي شجرة على Google Maps لإضافتها أو تعديلها.</span>
        </div>
      )}

      {/* Explicit container dimensions required by Google Maps Platform */}
      <div className="relative h-[400px] w-full rounded-lg overflow-hidden border border-slate-300">
        <APIProvider apiKey={apiKey} libraries={['marker']}>
          <GoogleMap
            mapId={DEMO_MAP_ID}
            defaultCenter={defaultCenter}
            defaultZoom={18}
            mapTypeId={mapTypeId}
            gestureHandling="greedy"
            tilt={0}
            streetViewControl={false}
            internalUsageAttributionIds={GMP_ATTRIBUTION_IDS}
            onClick={(ev) => {
              if (!editTreesMode || !onAddTreeAtLatLng) return;
              const latLng = ev.detail.latLng;
              if (latLng) {
                onAddTreeAtLatLng(latLng.lat, latLng.lng);
              }
            }}
          >
            <FarmBoundaryPolygons
              rawPoly={poly}
              calibratedPoly={calibratedPoly}
              clippedPoly={clippedPoly}
              hasCalibrationShift={Math.abs(offsetX_m) >= 0.05 || Math.abs(offsetY_m) >= 0.05}
              hasInwardBuffer={Math.abs(inwardBufferM) >= 0.05}
            />

            {geoTrees.map(({ tree, lat, lng }, idx) => (
              <TreePinMarker
                key={tree.id}
                tree={tree}
                position={{ lat, lng }}
                zIndex={idx + 1}
                selected={selectedTreeId === tree.id}
                onSelect={handleSelectTree}
                onRegisterMarker={handleRegisterMarker}
              />
            ))}

            <MarkerClusterController
              enabled={clusterEnabled}
              markerRegistry={markerRegistry}
              markerCount={geoTrees.length}
            />

            {selectedTree && selectedMarker && (
              <InfoWindow
                anchor={selectedMarker}
                pixelOffset={[0, -4]}
                onCloseClick={() => {
                  setSelectedTreeId(null);
                  setSelectedMarker(null);
                }}
              >
                <div className="min-w-[215px] text-slate-900 space-y-1.5 text-[11px]">
                  <div className="flex items-center justify-between gap-2 font-bold">
                    <span className="font-mono-num text-emerald-800">
                      Tree #{selectedTree.id} {selectedTree.source === 'manual' ? '(Manual)' : ''}
                    </span>
                    <span className="font-mono-num text-sky-700">
                      {Math.round(selectedTree.score * 100)}%
                    </span>
                  </div>

                  <div className="text-slate-700">{classLabelAr[selectedTree.className]}</div>

                  <div className="text-[10px] font-mono-num text-slate-500">
                    Crown: {selectedTree.crownDiameterM ?? '—'}m · Health:{' '}
                    {selectedTree.health || 'healthy'}
                  </div>

                  {selectedTree.lat !== undefined && selectedTree.lon !== undefined && (
                    <div className="text-[10px] font-mono-num text-slate-500">
                      {selectedTree.lat.toFixed(6)}, {selectedTree.lon.toFixed(6)}
                    </div>
                  )}

                  {onUpdateTree && (
                    <div className="pt-1.5 border-t border-slate-200 space-y-1.5">
                      <div className="grid grid-cols-3 gap-1 text-[10px]">
                        {(
                          [
                            { id: 'orchard_tree', lbl: 'بستان' },
                            { id: 'palm_tree', lbl: 'نخيل' },
                            { id: 'windbreak_tree', lbl: 'مصدات' },
                          ] as Array<{ id: TreeClassName; lbl: string }>
                        ).map((cOpt) => (
                          <button
                            key={cOpt.id}
                            type="button"
                            onClick={() => onUpdateTree(selectedTree.id, { className: cOpt.id })}
                            className={`px-1 py-0.5 rounded border cursor-pointer ${
                              selectedTree.className === cOpt.id
                                ? 'bg-emerald-700 border-emerald-700 text-white'
                                : 'bg-slate-100 border-slate-300 text-slate-700 hover:bg-slate-200'
                            }`}
                          >
                            {cOpt.lbl}
                          </button>
                        ))}
                      </div>

                      <div className="grid grid-cols-3 gap-1 text-[10px]">
                        {(
                          [
                            { id: 'healthy', lbl: 'سليمة' },
                            { id: 'stressed', lbl: 'مجهدة' },
                            { id: 'dead', lbl: 'تالفة' },
                          ] as Array<{ id: TreeHealthStatus; lbl: string }>
                        ).map((hOpt) => (
                          <button
                            key={hOpt.id}
                            type="button"
                            onClick={() => onUpdateTree(selectedTree.id, { health: hOpt.id })}
                            className={`px-1 py-0.5 rounded border cursor-pointer ${
                              (selectedTree.health || 'healthy') === hOpt.id
                                ? 'bg-indigo-700 border-indigo-700 text-white'
                                : 'bg-slate-100 border-slate-300 text-slate-700 hover:bg-slate-200'
                            }`}
                          >
                            {hOpt.lbl}
                          </button>
                        ))}
                      </div>

                      {onDeleteTree && (
                        <button
                          type="button"
                          onClick={() => {
                            onDeleteTree(selectedTree.id);
                            setSelectedTreeId(null);
                            setSelectedMarker(null);
                          }}
                          className="w-full inline-flex items-center justify-center gap-1 py-1 rounded bg-red-600 hover:bg-red-700 text-white text-[10px] font-semibold cursor-pointer"
                        >
                          <Trash2 className="w-3 h-3" />
                          <span>حذف الشجرة</span>
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </InfoWindow>
            )}
          </GoogleMap>
        </APIProvider>
      </div>
    </div>
  );
};

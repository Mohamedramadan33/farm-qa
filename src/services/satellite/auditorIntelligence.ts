import { getDetectionAuthHeaders } from './config';
import type {
  FarmAuditRecord,
  OrchardSpatialPattern,
  SatelliteAnalysisResult,
  SmartTreeAuditReport,
  SpatialOrchardMetrics,
  TreeDetectionBox,
} from '../../types/qa';

function percentileSorted(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = (sorted.length - 1) * Math.max(0, Math.min(1, p));
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function haversineMeters(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371000;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function distanceMetersBetweenBoxes(
  a: TreeDetectionBox,
  b: TreeDetectionBox,
  mpp: number
): number {
  if (
    a.lat !== undefined &&
    a.lon !== undefined &&
    b.lat !== undefined &&
    b.lon !== undefined
  ) {
    return haversineMeters(a.lat, a.lon, b.lat, b.lon);
  }
  return Math.hypot(a.cx - b.cx, a.cy - b.cy) * Math.max(mpp, 0.1);
}

/**
 * Deterministic Geospatial & Agronomic Tree Intelligence Analyzer:
 * Evaluates tree-to-tree spacing regularity, orchard spatial pattern,
 * oversized/merged-crown anomalies, missing orchard grid gaps, understory crops,
 * and Engineer-vs-AI count discrepancy hypotheses.
 */
export function computeSpatialOrchardMetrics(params: {
  boxes: TreeDetectionBox[];
  mpp: number;
  vegPct: number | null;
  engineerCount: number | null;
  aiCount: number | null;
  forecast: number | null;
  inwardBufferM: number;
}): SpatialOrchardMetrics {
  const { boxes, mpp, vegPct, engineerCount, aiCount, forecast, inwardBufferM } = params;

  const totalTrees = boxes.length;
  const boundaryTrees = boxes.filter((b) => b.onBoundary).length;
  const interiorTrees = boxes.filter((b) => !b.onBoundary).length;

  const crownDiams = boxes
    .map((b) => b.crownDiameterM ?? Math.sqrt((b.xmax - b.xmin) * (b.ymax - b.ymin)) * mpp)
    .filter((d) => Number.isFinite(d) && d > 0)
    .sort((a, b) => a - b);

  const medianCrownM =
    crownDiams.length > 0 ? Number(percentileSorted(crownDiams, 0.5).toFixed(2)) : null;

  const oversizedThreshold = medianCrownM !== null ? Math.max(4.8, medianCrownM * 1.75) : 5.5;
  const oversizedCrownIds = boxes
    .filter((b) => {
      const d = b.crownDiameterM ?? Math.sqrt((b.xmax - b.xmin) * (b.ymax - b.ymin)) * mpp;
      return d >= oversizedThreshold;
    })
    .map((b) => b.id);

  // Nearest-neighbor distances in meters
  const nnDists: number[] = [];
  if (boxes.length >= 2) {
    for (let i = 0; i < boxes.length; i++) {
      let best = Infinity;
      for (let j = 0; j < boxes.length; j++) {
        if (i === j) continue;
        const d = distanceMetersBetweenBoxes(boxes[i], boxes[j], mpp);
        if (d < best) best = d;
      }
      if (Number.isFinite(best) && best > 0.2 && best < 60) {
        nnDists.push(best);
      }
    }
  }
  nnDists.sort((a, b) => a - b);

  const medianSpacingM =
    nnDists.length > 0 ? Number(percentileSorted(nnDists, 0.5).toFixed(2)) : null;
  const p25SpacingM =
    nnDists.length > 0 ? Number(percentileSorted(nnDists, 0.25).toFixed(2)) : null;
  const p75SpacingM =
    nnDists.length > 0 ? Number(percentileSorted(nnDists, 0.75).toFixed(2)) : null;

  let spacingRegularityPct: number | null = null;
  if (medianSpacingM && medianSpacingM > 0.2 && p25SpacingM !== null && p75SpacingM !== null) {
    const iqrRatio = (p75SpacingM - p25SpacingM) / medianSpacingM;
    spacingRegularityPct = Math.max(0, Math.min(100, Math.round((1 - iqrRatio * 0.75) * 100)));
  }

  const windbreakCount = boxes.filter((b) => b.className === 'windbreak_tree').length;
  let pattern: OrchardSpatialPattern = 'sparse_or_none';
  if (totalTrees < 4) {
    pattern = 'sparse_or_none';
  } else if (
    (totalTrees > 0 && boundaryTrees / totalTrees >= 0.65) ||
    windbreakCount / totalTrees >= 0.6
  ) {
    pattern = 'perimeter_windbreak';
  } else if (interiorTrees >= 6 && (spacingRegularityPct ?? 0) >= 60) {
    pattern = 'row_grid';
  } else {
    pattern = 'mixed_orchard';
  }

  // Estimate missing orchard row gaps when a structured grid exists
  let estimatedMissingGapCount = 0;
  if (
    (pattern === 'row_grid' || pattern === 'mixed_orchard') &&
    medianSpacingM !== null &&
    medianSpacingM >= 2.2 &&
    medianSpacingM <= 10.5
  ) {
    const gapThresh = medianSpacingM * 1.85;
    for (const d of nnDists) {
      if (d > gapThresh && d <= medianSpacingM * 3.5) {
        estimatedMissingGapCount += 1;
      }
    }
  }

  const denseFlatCropDetected = Boolean(
    vegPct !== null && vegPct >= 35 && totalTrees <= 25
  );

  const discrepancyRatio =
    engineerCount !== null && engineerCount > 0 && aiCount !== null
      ? Number((aiCount / engineerCount).toFixed(2))
      : null;

  const diagnosticsAr: string[] = [];

  if (denseFlatCropDetected) {
    diagnosticsAr.push(
      `تم رصد غطاء نباتي أرضي كثيف (${vegPct}%) مع عدد أشجار محدود (${totalTrees}) — يشير ذلك غالباً لمحاصيل مسطحة (برسيم/خضار) تم استبعادها بنجاح من عد الأشجار.`
    );
  }

  if (oversizedCrownIds.length > 0) {
    diagnosticsAr.push(
      `تنبيه تيجان متداخلة: يوجد ${oversizedCrownIds.length} تاج أكبر من المعتاد (قد يمثل شجرتين متلاصقتين، مثل #${oversizedCrownIds
        .slice(0, 5)
        .join(', #')}).`
    );
  }

  if (estimatedMissingGapCount >= 2) {
    diagnosticsAr.push(
      `فجوات محتملة في صفوف الغرس: رُصدت ~${estimatedMissingGapCount} مسافات مفقودة مقارنة بمتوسط المسافة البينية (${medianSpacingM}م).`
    );
  }

  if (
    engineerCount !== null &&
    engineerCount > 0 &&
    aiCount !== null &&
    inwardBufferM <= -1.0 &&
    boundaryTrees > 0 &&
    aiCount < engineerCount * 0.7
  ) {
    diagnosticsAr.push(
      `معامل الإزاحة للداخل (${inwardBufferM}م) يستبعد مصدات الحدود (${boundaryTrees} شجرة مرصودة على الحافة)، مما قد يفسر ارتفاع رقم المهندس مقارنة بالعد الصافي الداخلي.`
    );
  }

  if (
    forecast !== null &&
    forecast > 0 &&
    engineerCount !== null &&
    aiCount !== null &&
    aiCount < engineerCount * 0.45 &&
    engineerCount > forecast * 1.25
  ) {
    diagnosticsAr.push(
      `عد المهندس (${engineerCount}) أعلى بكثير من كلٍ من توقع المحيط (${forecast}) والذكاء الاصطناعي (${aiCount}) — يُرجّح وجود شتلات صغيرة جداً أو عد محاصيل/مصدات جوار.`
    );
  }

  if (diagnosticsAr.length === 0 && totalTrees > 0) {
    diagnosticsAr.push(
      `توزيع الأشجار متسق (${totalTrees} شجرة بمتوسط قطر تاج ${medianCrownM ?? '—'}م${
        medianSpacingM ? ` ومسافة بينية ${medianSpacingM}م` : ''
      }).`
    );
  }

  return {
    totalTrees,
    interiorTrees,
    boundaryTrees,
    medianCrownM,
    medianSpacingM,
    p25SpacingM,
    p75SpacingM,
    spacingRegularityPct,
    pattern,
    oversizedCrownIds,
    estimatedMissingGapCount,
    denseFlatCropDetected,
    discrepancyRatio,
    diagnosticsAr,
  };
}

export async function requestSmartAiTreeAudit(params: {
  record: FarmAuditRecord;
  satResult?: SatelliteAnalysisResult;
  spatialMetrics: SpatialOrchardMetrics;
  signal?: AbortSignal;
}): Promise<SmartTreeAuditReport> {
  const { record, satResult, spatialMetrics, signal } = params;

  const classBreakdown = (satResult?.boxes || []).reduce(
    (acc, b) => {
      acc[b.className] = (acc[b.className] || 0) + 1;
      return acc;
    },
    { orchard_tree: 0, palm_tree: 0, windbreak_tree: 0 }
  );
  const healthBreakdown = (satResult?.boxes || []).reduce(
    (acc, b) => {
      const h = b.health || 'healthy';
      acc[h] = (acc[h] || 0) + 1;
      return acc;
    },
    { healthy: 0, stressed: 0, dead: 0 }
  );

  const resp = await fetch('/api/audit-trees', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getDetectionAuthHeaders(),
    },
    signal,
    body: JSON.stringify({
      farmId: record.farm_id,
      farmName: record.farm_name,
      region: record.region,
      species: record.species,
      cropType: record.crop_type,
      treeDetails: record.tree_details,
      areaEntered: record.area_entered,
      areaGps: record.area_gps,
      perimeterM: record.perimeter_m,
      engineerCount: record.engineer_count,
      aiCount: record.ai_count,
      aiLow: record.ai_low,
      aiHigh: record.ai_high,
      forecast: record.forecast,
      boundarySpacingM: record.boundary_spacing_m,
      vegPct: record.veg_pct,
      imageryDate: record.imagery_date,
      imgAge: record.img_age,
      verdict: record.verdict,
      gpsWarnings: record.gps_warnings,
      provider: satResult?.provider ?? null,
      zoomUsed: satResult?.zoomUsed ?? record.zoom_used,
      inwardBufferM: satResult?.inwardBufferM ?? -1.5,
      classBreakdown,
      healthBreakdown,
      spatialMetrics,
    }),
  });

  const data = (await resp.json().catch(() => ({}))) as {
    ok?: boolean;
    report?: SmartTreeAuditReport;
    error?: string;
  };
  if (!resp.ok || !data.ok || !data.report) {
    throw new Error(data.error || `Smart Audit failed (${resp.status})`);
  }
  return data.report;
}

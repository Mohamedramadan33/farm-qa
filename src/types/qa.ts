export type TeamMemberKey = 'mohamed' | 'mansy' | 'haidy' | 'islam';

export interface TeamMember {
  key: TeamMemberKey;
  email: string;
  name: string;
}

export const TEAM: Record<TeamMemberKey, TeamMember> = {
  mohamed: { key: 'mohamed', email: 'mohamed@ebda.earth', name: 'Mohamed' },
  mansy: { key: 'mansy', email: 'mansy.ayman@ebda.earth', name: 'Mansy' },
  haidy: { key: 'haidy', email: 'haidy.abdelnasser@ebda.earth', name: 'Haidy' },
  islam: { key: 'islam', email: 'eslam.ashraf@ebda.earth', name: 'Islam' },
};

export const TEAM_KEYS: TeamMemberKey[] = ['mohamed', 'mansy', 'haidy', 'islam'];

export const STATUS_PENDING = 'Pending Review';
export const STATUS_APPROVED = 'Approved Entire Form';
export const STATUS_CONDITIONAL = 'Conditional (Modifications Required)';
export const STATUS_REJECTED = 'REJECTED ENTIRE FORM';

export type FormStatus =
  | typeof STATUS_PENDING
  | typeof STATUS_APPROVED
  | typeof STATUS_CONDITIONAL
  | typeof STATUS_REJECTED;

export const FORM_STATUS_CHOICES: FormStatus[] = [
  STATUS_PENDING,
  STATUS_APPROVED,
  STATUS_CONDITIONAL,
  STATUS_REJECTED,
];

export type ItemDecision = 'OK' | 'Reject' | 'Modify' | 'Clarify';
export const ITEM_DECISIONS: ItemDecision[] = ['OK', 'Reject', 'Modify', 'Clarify'];

export interface UserSession {
  key: TeamMemberKey;
  name: string;
  email: string;
  mustChangePassword: boolean;
}

export interface KoboAttachmentRaw {
  filename?: string;
  mimetype?: string;
  download_url?: string;
  download_medium_url?: string;
  download_small_url?: string;
}

export interface RawKoboSubmission {
  _id: number | string;
  _submission_time?: string;
  _validation_status?: {
    uid?: string;
    label?: string;
  };
  _attachments?: KoboAttachmentRaw[];
  [key: string]: unknown;
}

export type FlattenedRow = Record<string, unknown>;

export interface AttachmentInfo {
  previewUrl: string;
  fullUrl: string;
  mimetype: string;
  filename: string;
}

export type LabelMap = Record<string, [string, string]>;

export type PointLonLat = [number, number];
export type PointLatLon = [number, number];

export type TreeClassName = 'orchard_tree' | 'palm_tree' | 'windbreak_tree';
export type TreeHealthStatus = 'healthy' | 'stressed' | 'dead';
export type DetectorMode = 'gemini' | 'yolo' | 'morphological';
export type BasemapProvider = 'google' | 'esri';
export type BasemapViewMode = 'google' | 'esri' | 'compare' | 'gmaps';

export type OrchardSpatialPattern =
  | 'row_grid'
  | 'perimeter_windbreak'
  | 'mixed_orchard'
  | 'sparse_or_none';

export interface SpatialOrchardMetrics {
  totalTrees: number;
  interiorTrees: number;
  boundaryTrees: number;
  medianCrownM: number | null;
  medianSpacingM: number | null;
  p25SpacingM: number | null;
  p75SpacingM: number | null;
  spacingRegularityPct: number | null;
  pattern: OrchardSpatialPattern;
  oversizedCrownIds: number[];
  estimatedMissingGapCount: number;
  denseFlatCropDetected: boolean;
  discrepancyRatio: number | null;
  diagnosticsAr: string[];
}

export interface SmartTreeAuditReport {
  alignmentVerdict:
    | 'aligned'
    | 'under_counted_by_ai'
    | 'over_counted_by_engineer'
    | 'boundary_mismatch'
    | 'crop_confusion_risk';
  confidencePct: number;
  discrepancyCauseAr: string;
  canopyQualitySummaryAr: string;
  fieldCheckRecommendationsAr: string[];
  suggestedAuditorComment: string;
  generatedAt: string;
}

export interface SatelliteCalibrationOptions {
  providerMode: BasemapViewMode;
  offsetX_m: number;
  offsetY_m: number;
  inwardBufferM: number;
  highRes?: boolean;
  highAccuracy?: boolean;
  confThresh?: number;
}

export interface SatelliteProgress {
  stage: 'tiles' | 'detect' | 'postprocess';
  loaded: number;
  total: number;
  message?: string;
}

export interface TreeDetectionBox {
  id: number;
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  cx: number;
  cy: number;
  radiusPx: number;
  score: number;
  classId: 0 | 1 | 2;
  className: TreeClassName;
  health?: TreeHealthStatus;
  onBoundary: boolean;
  shadowDrop: number;
  laplacianVar: number;
  circularity?: number;
  crownDiameterM?: number;
  lat?: number;
  lon?: number;
  source?: 'ai' | 'manual';
}

export type LibraryEngineId =
  | 'gemini_vision'
  | 'yolov8_onnx'
  | 'morphological_watershed'
  | 'multiscale_dog'
  | 'shadow_texture_pair'
  | 'turf_consensus_ensemble';

export interface LibraryDetectionSummary {
  id: LibraryEngineId;
  name: string;
  nameAr: string;
  libraryBadge: string;
  category: 'ai_vision' | 'deep_learning' | 'cv_classical' | 'geospatial_ensemble';
  status: 'ran' | 'unavailable' | 'error';
  unavailableReason?: string;
  count: number;
  interiorCount: number;
  boundaryCount: number;
  avgConfidence: number;
  medianCrownM: number | null;
  durationMs: number;
  boxes: TreeDetectionBox[];
  notesAr: string;
}

export interface AiLibraryArbitration {
  winnerEngineId: LibraryEngineId;
  recommendedCount: number;
  confidencePct: number;
  explanationAr: string;
  cropRiskWarningAr?: string;
  generatedAt: string;
}

export interface MultiLibraryComparisonReport {
  libraries: LibraryDetectionSummary[];
  activeEngineId: LibraryEngineId;
  recommendedEngineId: LibraryEngineId;
  recommendedCount: number;
  consensusCount: number;
  medianCount: number;
  agreementPct: number;
  aiArbitration?: AiLibraryArbitration;
}

export interface SatelliteAnalysisResult {
  aiCount: number;
  aiLow: number;
  aiHigh: number;
  aiInteriorTrees: number;
  aiRaw: number;
  aiFactor: number;
  vegPct: number;
  crownM: number | null;
  imageryDate: string | null;
  imgAge: number | null;
  pseudoNdvi: number | null;
  zoomUsed: number;
  mpp: number;
  provider: string;
  providerId: BasemapProvider;
  detectorMode: DetectorMode;
  detectorEngine: string;
  aiVisionNotice?: string;
  inwardBufferM: number;
  offsetX_m: number;
  offsetY_m: number;
  confThreshUsed?: number;
  canvasWidth: number;
  canvasHeight: number;
  worldOriginX?: number;
  worldOriginY?: number;
  ringPx: Array<[number, number]>;
  clippedRingPx: Array<[number, number]>;
  boxes: TreeDetectionBox[];
  rawImageUrl?: string;
  cleanImageUrl: string;
  maskedImageUrl?: string;
  annotatedImageUrl: string;
  compareImageUrl?: string | null;
  compareProvider?: string | null;
  compareZoomUsed?: number | null;
  stage1ActivePct?: number;
  stage1AuditRejectedAi?: number;
  stage1CellsKept?: number;
  stage1CellsTotal?: number;
  highResMode?: boolean;
  sahiTileSizeUsed?: number;
  sahiOverlapUsed?: number;
  sahiTilesCount?: number;
  xyzTilesSavedByPolyFilter?: number;
  aiError?: string;
  libraryComparison?: MultiLibraryComparisonReport;
}

export interface FarmAuditRecord {
  _dfIndex: number;
  _submissionId: string;
  _poly: PointLonLat[] | null;
  farm_id: string;
  farm_name: string;
  engineer: string;
  farmer: string;
  region: string;
  submission_time: string;
  visit_date: string;
  audit_type: string;
  season: string;
  plot_owner: string;
  owner_phone: string;
  ownership_type: string;
  soil_type: string;
  irrig_type: string;
  irrig_amnt: string;
  crop_type: string;
  animal_type: string;
  animal_count: string;
  non_compliance: string;
  corrective_act: string;
  auditor_recom: string;
  tree_details: string;
  missing: string;
  gps_warnings: string;
  area_entered: number | null;
  area_gps: number | null;
  perimeter_m: number | null;
  area_check: string;
  engineer_count: number | null;
  forecast: number | null;
  boundary_spacing_m: number;
  species: string;
  density_note: string;
  ai_count: number | null;
  ai_low: number | null;
  ai_high: number | null;
  ai_interior_trees: number | null;
  ai_raw: number | null;
  ai_factor: number | null;
  veg_pct: number | null;
  crown_m: number | null;
  imagery_date: string | null;
  img_age: number | null;
  pseudo_ndvi: number | null;
  zoom_used: number | null;
  ai_error: string;
  verdict: string;
  confidence: 'High' | 'Medium' | 'Low';
  flags: string[];
  Form_Status: FormStatus;
  Rejected_Items_Count: number;
  Consolidated_Comments: string;
  General_Auditor_Note: string;
  Auditor_Name: string;
  Reviewed_At: string;
}

export interface FieldDecisionRow {
  farm_id: string;
  Field_Key: string;
  Question_Title: string;
  Submitted_Answer: string;
  Item_Decision: ItemDecision;
  Suggested_Correction_or_Comment: string;
  Attach_to_Summary: boolean;
}

export interface MasterReviewRecord {
  farm_id: string;
  submission_id: string;
  Form_Status: FormStatus;
  Rejected_Items_Count: number;
  Consolidated_Comments: string;
  General_Auditor_Note: string;
  Auditor_Name: string;
  Reviewed_At: string;
}

export interface MediaQuestionItem {
  Field_Key: string;
  Question_Title: string;
  Filename: string;
  Type: 'image' | 'video';
  PreviewUrl: string;
  FullUrl: string;
}

export interface AssignmentRecord {
  to: TeamMemberKey;
  by: string;
  at: string;
  farmId?: string;
}

export interface WorklistItem {
  idx: number;
  subId: string;
  farmId: string;
  farmName: string;
  farmer: string;
  region: string;
  engineer: string;
  submitted: string;
  assignedKey: TeamMemberKey | '';
  assignedName: string;
}

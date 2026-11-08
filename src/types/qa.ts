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

export interface TreeDetectionBox {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  score: number;
  onBoundary: boolean;
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
  ndviRecent: number | null;
  zoomUsed: number;
  provider: string;
  annotatedImageUrl: string;
  aiError?: string;
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
  ndvi_recent: number | null;
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

# EBDA & Kobo — Farm QA Audit

A full-stack React + TypeScript + Express web application for agricultural Participatory Guarantee System (PGS) form auditing, GPS polygon geometry validation, satellite perimeter tree crown analysis, and fair auditor worklist distribution.

## Core Features

- **Team Authentication & Workload Distribution**: Supports EBDA audit team members (`mohamed`, `mansy`, `haidy`, `islam`), first-time password enforcement, manual farm reassignment, equal redistribution, and governorate-balanced fair work division (`max_gov_ratio = 1.4`).
- **Geospatial Polygon & UTM Metrics**: Parses Kobo GPS traces, auto-untangles self-intersecting ("bowtie") point sequences, projects WGS84 coordinates to UTM to compute farm area in Feddan (`4200.83 m²`) and perimeter in meters, evaluates GPS quality (compactness & capture accuracy), and flags cross-farm overlapping polygons (`IoU >= 80%`).
- **Satellite Boundary & Canopy Audit**: Stitches high-resolution satellite tiles (Google Earth / ESRI World Imagery), retrieves capture dates (`SRC_DATE2`) and Sentinel-2 STAC metadata, computes Excess Green Index (`ExG`) vegetation cover along an 8-meter perimeter corridor, detects boundary vs. interior tree crowns, and compares against engineer species counts and perimeter spacing forecasts.
- **PGS Administrative & Agronomic Rules Engine**: Validates Egyptian mobile/landline phone numbers, minimum soil sample counts (`>= 4`) and polygon containment (`5m` edge tolerance), visit dates, 3-member audit committee photos, group farmer basin sums, mandatory document attachments (Tenure, Biodiversity, Non-GMO, Farm Ledger), and current/previous crop Feddan/Qirat/Sahm totals.
- **Interactive Media & Field QA**: Photo/video audit cards, searchable field-level verdict editor, consolidated summary builder, `.xlsx` Excel export with formula-injection protection, Kobo validation status sync, and Airtable record upsert.

## Development & Configuration

Copy `.env.example` to `.env` to connect live KoboToolbox and Airtable credentials:

```bash
npm install
npm run dev
```

The application runs on `http://0.0.0.0:3000`.
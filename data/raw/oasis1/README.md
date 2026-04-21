# OASIS-1 (Cross-sectional) — raw downloads

This folder contains the OASIS-1 cross-sectional dataset artifacts exactly as downloaded.

## Files

- `oasis_cross-sectional_disc1.tar.gz` / `oasis_cross-sectional_disc2.tar.gz` / `oasis_cross-sectional_disc3.tar.gz`
  - Compressed archives containing per-session MRI data organised by session directory (e.g. `OAS1_0012_MR1/`).
  - Each session typically includes:
    - `RAW/` (individual T1 scan repetitions)
    - `PROCESSED/` (averaged image + atlas-registered variants)
    - `FSL_SEG/` (tissue segmentation derived from the masked atlas image)
    - Session metadata in `*.xml` and `*.txt`

- `oasis_cross-sectional-5708aa0a98d82080.xlsx`
  - Spreadsheet of demographic/clinical/derived measures for the cross-sectional cohort.
  - Sheet: `oasis_cross-sectional`
  - Header columns observed:
    - `ID`: session ID (e.g. `OAS1_0001_MR1`)
    - `M/F`: sex
    - `Hand`: handedness
    - `Age`: age (years)
    - `Educ`: education code
    - `SES`: socioeconomic status code
    - `MMSE`: Mini-Mental State Examination score
    - `CDR`: Clinical Dementia Rating
    - `eTIV`: estimated total intracranial volume
    - `nWBV`: normalised whole brain volume
    - `ASF`: atlas scaling factor
    - `Delay`: scan delay field (often `N/A` for single-session entries)

- `oasis_cross-sectional-reliability-063c8642b909ee76.xlsx`
  - Spreadsheet for the small “acquisition reliability” subset (repeat sessions for some nondemented subjects).
  - Sheet: `oasis_cross-sectional-reliabili`
  - Header columns observed: `ID`, `Delay`, `eTIV`, `nWBV`, `ASF`

- `oasis_cross-sectional_facts-bcc7a002dfb104f4.pdf`
  - OASIS fact sheet describing the dataset, directory structure, and variable definitions.

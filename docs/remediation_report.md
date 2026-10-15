# CareerSense AI Remediation Report

**Completed:** 2026-10-15
**Baseline:** `docs/project_audit_report.md`
**Scope:** correctness, evidence semantics, security boundaries, model reproducibility, roadmap coverage, privacy, automated tests, documentation, and a full Streamlit interface redesign.

## Executive result

The repository is now a substantially safer and more coherent **career pathway exploration prototype**. The implementation no longer presents synthetic performance as proof of career suitability, no longer allows a selected target to boost its own ranking, distinguishes self-reported from advisor-reviewed evidence, abstains when evidence is insufficient or the leading scores are too close, and limits each role to an appropriate data view.

The interface has been rebuilt around five student tasks—pathway review, learning plan, evidence, academic record, and report—with parallel professional workspaces for advisors, coordinators, and administrators. The design is responsive, keyboard-conscious, consistent in typography and spacing, and intentionally avoids decorative dashboard clutter.

This remediation does **not** make the product a validated career predictor. Real student outcomes, multi-cohort external validation, protected-attribute fairness review, institutional policy approval, and production operations remain prerequisites for consequential use.

## Audit finding disposition

| Original finding | Status | Remediation |
|---|---|---|
| Label-first synthetic generator created a closed evaluation loop | Resolved in code; external validity still open | Profiles now originate from continuous latent aptitudes with noise; the simulated label is chosen last. All model copy calls the result a synthetic benchmark. |
| Deployed composite differed from the evaluated model | Resolved | The exact `70% K-NN competency + 20% interests + 10% degree alignment` formula is evaluated on the held-out synthetic split and displayed in admin diagnostics. |
| Interests and target aspiration were double-counted | Resolved | The 31-feature competency model excludes interests. Registration no longer increases an interest rating, and `target_career` only marks the selected row. |
| “Pure ML competency” included interests | Resolved | Competency features now contain grades and technical skills only. Interests are a separate, visible advisory dimension. |
| Always-on recommendation despite weak evidence | Resolved | The classifier abstains below two academic/skill evidence items and withholds a single top recommendation when the leading margin is under five points. |
| No subgroup diagnostics | Partially resolved | Admin diagnostics now report degree-programme error differences on the synthetic holdout. The UI explicitly says this is not a real-world fairness audit and protected attributes are not available. |
| Self-reported evidence was treated as verified | Resolved | Skills, projects, and document metadata have evidence states. Advisors can review assigned students and verify individual claims. |
| Uploaded document bytes were not parsed but marked verified | Resolved | The system stores metadata, MIME type, and SHA-256 fingerprint only; it says bytes are not retained or parsed and defaults to self-reported. |
| Roadmap completion automatically promoted skills | Resolved | Completion records activity progress only. Skill levels change only through an explicit self-assessment or advisor review. |
| Readiness claimed job/internship readiness | Resolved | The result is a prototype readiness index with evidence-weighted inputs and an explicit non-employability limitation. |
| CV readiness formula could subtract points | Resolved | Non-CV document counts are clamped and legacy filenames do not silently prove evidence. |
| Serialized model/version drift | Resolved | Python and libraries are pinned, artifacts were rebuilt, schema and exact dependency provenance are checked before unpickling, and saves are atomic. |
| Missing learning-graph competency coverage | Resolved | Activities were added for OOP, Git, SQL, secure coding, monitoring/logging, design/typography, and frontend awareness. All five zero-evidence pathway tests now return full graph coverage. |
| “Optimal” roadmap claim was too broad | Resolved in product language | The UI states that A* minimizes declared activity hours only; weekly scheduling happens after search. Uncovered gaps remain explicit. |
| Hard-coded coordinator insights | Resolved | Insights are computed from live aggregate data. All analytics are withheld below a five-student reporting threshold. |
| Coordinator privacy exposure | Resolved for the prototype | Coordinator queries strip names, registration numbers, and database IDs; the UI exposes aggregates only. |
| SQLite foreign keys disabled | Resolved | Every connection enables foreign keys and a busy timeout; regression tests verify enforcement. |
| UI-only authorization and universal advisor access | Substantially resolved | Advisor assignments are modeled, advisor reads and verification are scoped, roadmap updates require both record and owner IDs, and demo switching exists only on the explicit demo screen. A networked deployment should still add a dedicated service/API authorization layer. |
| Downloadable report allowed HTML injection | Resolved | Dynamic text is escaped and numeric style values are bounded. |
| Tests missed audit-specific behavior | Resolved for repository-known cases | Regression coverage now includes abstention, ambiguous results, target invariance, feature separation, CV scoring, graph coverage, foreign keys, owner scoping, advisor assignment, coordinator identifier removal, small-cohort suppression, artifact diagnostics, and report escaping. |

## Model and evidence changes

The current score is an **uncalibrated exploration index**, not a probability:

```text
70% competency-neighbour distribution
+ 20% normalized stated interests
+ 10% degree-curriculum alignment
```

The selected pathway does not enter this equation. K-NN provides the competency distribution; the decision tree remains an interpretable diagnostic baseline. With insufficient evidence the system does not show the ranking as actionable. With a top-two margin below five percentage points, it preserves the exploratory ranking but declines to name one recommended pathway.

Current synthetic held-out results:

| Evaluation | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Exact production composite | 43.53% | 43.98% | 43.72% |
| K-NN competency model | 44.71% | 44.57% | 44.72% |
| Dummy baseline | 24.12% | 7.77% | 9.37% |

The lower metrics are more credible than the former 88.82% because the generator no longer constructs features from a preselected answer. They still measure agreement with a simulation, not career success.

## Interface rework

The application now uses one shared visual system:

- neutral light workspace with a high-contrast navigation rail;
- Manrope with system-font fallbacks, restrained typography, and consistent spacing;
- shared page headers, section headers, status badges, cards, empty states, and focus styling;
- plain-language labels with no emoji navigation or decorative icon noise;
- responsive behavior, 16px mobile inputs, reduced-motion support, and explicit transitions;
- task-based student navigation and evidence-first advisor workflows;
- small-cohort privacy states and transparent administrator diagnostics;
- current Streamlit full-width APIs, avoiding the deprecated `use_container_width` parameter.

Browser QA covered login, student pathway and evidence views, assigned-advisor review, coordinator suppression, administrator diagnostics, and a narrow mobile viewport.

## Verification

- **Automated tests:** 85 passed.
- **Syntax/import compilation:** all application, AI, database, script, and test modules compile.
- **Model artifacts:** compatible with Python 3.11.17, scikit-learn 1.6.1, NumPy 2.0.2, and pandas 2.3.3.
- **Feature schema:** 31 competency features; no interest feature is present in the K-NN/decision-tree inputs.
- **Roadmap coverage:** all five configured pathways report full zero-evidence graph coverage.
- **Browser QA:** a fresh role-flow session reported zero application JavaScript errors and zero warnings; Streamlit emitted two verbose browser notes about its generated password-field structure.
- **CI:** a Python 3.11 workflow installs exact requirements and runs unittest discovery.

## Remaining prerequisites before real deployment

These items cannot be solved honestly with repository changes alone:

1. Define a governance-approved decision and outcome. Exploration quality, course planning, internship preparation, and role prediction are different products.
2. Collect consented, de-identified, multi-cohort data and reserve a temporal or institution-level external test set.
3. Run real calibration, subgroup, accessibility, and harm studies with students and advisors; include protected attributes only under an approved governance process.
4. Version the competency rubric and learning resources with owners, review dates, availability, price, modality, and accessibility metadata.
5. Add production identity, server-side sessions, audit logging, secrets management, backup/restore, retention rules, incident response, and an authorization-aware API/service layer.
6. Decide whether documents should ever be retained. If so, establish encryption, malware scanning, access logging, deletion, consent, and retention controls before storing bytes.

## Recommendation

The revised repository is suitable for demonstration, supervised academic evaluation, and further user research. It should remain clearly labeled as an exploration prototype until external validation and institutional governance are complete. The next investment should be real evidence and service-layer security—not a more complex classifier.

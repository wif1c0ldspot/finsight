# Evaluation dataset

The corpus contains authored summaries of 11 dated primary-source announcements
and articles, retrieved for this revision on 3 October 2026. Each Markdown file has
an adjacent `.metadata.json` sidecar with its source URL, publication date and
retrieval date. Ingestion records a SHA256 revision of the local summary. That
revision identifies the text indexed here; it is not a claim to archive the full
publisher page or detect later changes on that website.

These historical snapshots replace the previous unsourced company overviews.
They are deliberately bounded research fixtures, not current company profiles.
Questions about a publication date must distinguish it from the event or reporting
period discussed in that publication. Missing current figures should remain
unanswered even if a model remembers a possible value from training.

## Sources

- [Airwallex US launch, 26 August 2021](https://www.airwallex.com/global/newsroom/airwallex-furthers-global-expansion-with-launch-into-north-america).
- [Airwallex Series F, 21 May 2025](https://www.airwallex.com/global/blog/New-Chapter-Series-F).
- [Stripe employee tender offer, 28 February 2024](https://stripe.com/newsroom/news/employee-liquidity-feb-2024).
- [Stripe Series I, 15 March 2023](https://stripe.com/newsroom/news/stripe-series-i-employee-liquidity).
- [Stripe business survey, 28 March 2023](https://stripe.com/newsroom/news/2023-insights-report).
- [Wise name change, 22 February 2021](https://wise.com/ca/blog/world-meet-wise).
- [Wise direct listing, 7 July 2021](https://wise.com/gb/blog/the-only-number-that-matters).
- [DBS founding history, 16 July 2018](https://www.dbs.com/media/features/a-bank-is-born.page).
- [DBS 2023 results briefing, 7 February 2024](https://www.dbs.com/iwov-resources/images/investors/quarterly-financials/2023/4Q23_media_briefing_transcript.pdf?pid=sg-group-pweb-investors-pdf-4Q23_media_briefing_transcript), especially pages 1–2. Net profit before and after one-time items must be distinguished.
- [Razorpay Series F, 20 December 2021](https://razorpay.com/newsroom/razorpay-raises-375-mn-led-by-lone-pine-capital-alkeon-capital-and-tcv-valuation-increases-to-7-5-bn/).
- [Razorpay Series E, 19 April 2021](https://razorpay.com/blog/announcing-razorpays-160-million-series-e-funding-valuation-triples-to-3-billion/). This dated blog is used instead of a newsroom copy whose page date differs from its dateline.

## Golden cases

The 40 authored cases cover single facts, numerical questions, multi-document
comparisons, temporal distinctions, misleading premises, unavailable facts and
instructions to fabricate answers. Each case has a category, an explicit
`development` or `held_out` partition, expected document IDs, and a reference
answer when answerable. Arithmetic references specify calculations where needed.

The held-out partition is a suggested comparison set, not a secret or independently
collected benchmark: both questions and labels are public, share the same corpus,
and were authored during development. Do not tune against it and then describe its
score as an unbiased estimate. Labels received code-level consistency checks and
agent review; human adjudication and broader independent data remain desirable.

No live-model score is bundled or claimed. Automated model doubles validate the
harness and control flow only. Before claiming retrieval or answer improvements,
record baseline and candidate runs with identical corpus/golden hashes, model
identities and budgets, examine category coverage, and inspect disputed answers.
Adversarial questions test evaluation coverage; they do not establish a general
prompt-injection defense.

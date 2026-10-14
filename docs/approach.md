# User intent classifier: engineering and evaluation

## Task and routing
The user intent router chooses between retrieval of videos (`video_search`), summarization, video question-answering, editing operations, general conversation, and a safe `clarify` fallback. It processes complete text queries, without forcing an utterance into a video-specific class. The mapping from label to downstream service is explicit and inspectable.

## Pipeline
Unicode string -> normalized whitespace/case -> TF-IDF word unigrams/bigrams and character 3-5 grams -> balanced logistic regression -> calibrated-like probability ranking -> confidence/margin-based abstention -> route key.

This is a CPU-friendly sparse baseline that can be trained on domain-specific queries and does not require any model download or API credentials. The threshold prevents low-confidence and ambiguous requests from silently dispatching actions. **Logistic-regression softmax values are not guaranteed calibrated**; fit a held-out calibration model before relying on confidence as probability.

## Evaluation
`python -m src.main --evaluate` performs seeded stratified 5-fold cross-validation and outputs accuracy, macro-F1, per-label confusion matrix. **The provided phrases are hand-written prototypes** and the resulting score cannot be interpreted as production effectiveness. Replace with human-annotated representative queries, stratified by intent and source/user, avoid near-duplicate leakage, quantify unknown/ambiguous intent abstention rate, and tune thresholds using a separate validation set. Record p50/p95 latency and memory on the target CPU for scale testing.

## Latency and scaling
Fit once at startup, not per request. Sparse features and linear classifier make single-request inference inexpensive; batch requests for throughput. For many intents or multilingual users, use hierarchical routing (broad category then domain-specific sub-router). Cache normalized duplicate requests only if privacy rules permit. Trigger explicit clarification instead of risky edits when intent is uncertain.

## Limitations and next steps
Current seed training does not provide robust out-of-domain coverage, multilingual robustness, or true calibrated probabilities. Improve with collected anonymized queries, adversarial ambiguous examples, open-set evaluation, explicit user consent for consequential editing actions and continuous drift monitoring. Validate latency and accuracy against a small transformer baseline under identical workloads.

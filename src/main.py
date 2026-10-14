"""Fast, transparent hierarchical intent router using sparse character/word features."""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# Separate command/demo intents from information retrieval. Trainable, configurable router.
SEEDS = {
    "video_search": [
        "find footage of a dog running", "show me videos with waterfalls", "search for clips featuring bicycles",
        "retrieve a video of people dancing", "find the moment when the ball enters the goal",
        "look up surveillance footage of a red car", "show videos with a person holding a phone",
        "which clips contain a meeting", "locate recordings where a person falls", "get clips of cats",
    ],
    "video_summary": [
        "summarize this video", "what happens in this recording", "give a short recap of the clip",
        "describe the events in the video", "explain what happens during this footage",
        "tell me the story in the clip", "summarize the surveillance recording", "video overview please",
    ],
    "video_question": [
        "what color is the car in this video", "how many people appear in the scene",
        "when does the person enter the room", "who is speaking in this clip",
        "what is the man holding", "is there a dog in this video", "what happens after the door opens",
        "where did the cyclist go",
    ],
    "editing": [
        "trim the first ten seconds", "crop the recording to a square", "blur the face",
        "add subtitles to the clip", "export the video as mp4", "cut this clip into two",
        "stabilize the shaky footage", "slow down the recording", "mute the audio",
    ],
    "general": [
        "hello", "thanks for your help", "how are you", "tell me a joke",
        "good morning", "what can you do", "please help", "bye for now",
    ],
}
ROUTES = {
    "video_search": "retrieval_index",
    "video_summary": "video_summarizer",
    "video_question": "video_qa",
    "editing": "video_editor",
    "general": "general_assistant",
    "clarify": "clarification_prompt",
}

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def fit_router(samples=None):
    from sklearn.model_selection import train_test_split
    samples = samples or SEEDS
    x = [normalize(s) for label, phrases in samples.items() for s in phrases]
    y = [label for label, phrases in samples.items() for _ in phrases]
    if len(set(y)) < 2:
        raise ValueError("At least two intent classes required")
    model = Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(max_iter=800, class_weight="balanced")),
    ])
    model.fit(x, y)
    return model

def predict(model, query, threshold=0.40, margin=0.06):
    query = normalize(query)
    if not query:
        return {"query": query, "intent": "clarify", "route": ROUTES["clarify"], "confidence": 0.0}
    probabilities = model.predict_proba([query])[0]
    labels = model.classes_
    ranked = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)
    label, confidence = ranked[0]
    if confidence < threshold or (len(ranked) > 1 and confidence-ranked[1][1] < margin):
        label = "clarify"
    return {"query": query, "intent": label, "route": ROUTES[label],
            "confidence": round(float(confidence), 4),
            "candidates": [{"intent": name, "probability": round(float(score), 4)} for name, score in ranked[:3]]}

def evaluate(model, samples=SEEDS):
    """Leave-one-example-out avoids reporting training-set accuracy as held-out."""
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    x = [normalize(s) for label, phrases in samples.items() for s in phrases]
    y = [label for label, phrases in samples.items() for _ in phrases]
    # Deterministic stratified 5-fold CV; includes only known-intent classification.
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = sorted(samples)
    predicted = cross_val_predict(Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(max_iter=800, class_weight="balanced")),
    ]), x, y, cv=cv)
    return {"samples": len(x), "accuracy": accuracy_score(y, predicted),
            "macro_f1": classification_report(y, predicted, output_dict=True, zero_division=0)["macro avg"]["f1-score"],
            "labels": labels, "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist(),
            "warning": "Small hand-written seed dataset: CV estimates are illustrative, not real-world generalization."}

def main():
    parser = argparse.ArgumentParser(description="Lightweight user-intent router")
    parser.add_argument("--query", help="Single user query")
    parser.add_argument("--input", help="JSONL with a text or query key, one query per line")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0,1]")
    start = time.perf_counter()
    model = fit_router()
    queries = []
    if args.query: queries.append(args.query)
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    queries.append(obj if isinstance(obj, str) else obj.get("query", obj.get("text", "")))
    if not queries and not args.evaluate:
        queries = ["find clips showing a person running", "trim the last 5 seconds",
                   "what color is the car?", "summarize the meeting recording"]
    output = {"model": "sparse_tfidf_logistic_regression", "predictions": []}
    for query in queries:
        begin = time.perf_counter()
        result = predict(model, query, threshold=args.threshold)
        result["inference_ms"] = round((time.perf_counter()-begin)*1000, 3)
        output["predictions"].append(result)
        print(json.dumps(result))
    output["training_seconds"] = round(time.perf_counter()-start, 3)
    if args.evaluate:
        output["evaluation"] = evaluate(model)
        print(json.dumps(output["evaluation"]))
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

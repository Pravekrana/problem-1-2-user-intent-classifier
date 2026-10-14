from src.main import fit_router, predict, normalize, evaluate
def test_normalize():
    assert normalize("  FIND   Clips  ") == "find clips"
def test_empty_query_abstains():
    assert predict(fit_router(), "   ")["intent"] == "clarify"
def test_routing_contract():
    r = predict(fit_router(), "find video of a dog running", threshold=0)
    assert "route" in r and 0 <= r["confidence"] <= 1
def test_evaluation():
    r = evaluate(fit_router())
    assert r["samples"] >= 30
    assert 0 <= r["macro_f1"] <= 1
    assert len(r["confusion_matrix"]) == 5

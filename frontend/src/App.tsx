import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileSearch,
  History,
  Info,
  ShieldCheck,
  Sparkles,
  Target,
} from "lucide-react";
import "./App.css";

type Prediction = {
  label: string;
  deception_probability: number;
  truth_probability: number;
  risk_level: string;
};

type CalculationData = {
  base_probability?: number;
  calibrated_probability?: number;
  classification_threshold?: number;
  risk_thresholds?: {
    low?: number;
    high?: number;
  };
  formula?: string;
  risk_level?: string;
};

type TermContribution = {
  term: string;
  contribution: number;
};

type AnalysisResult = {
  prediction: Prediction;
  calculations: CalculationData;
  linguistic_features?: Record<string, number>;
  signals?: string[];
  term_contributions?: {
    deceptive?: TermContribution[];
    truthful?: TermContribution[];
  };
};

type HistoryItem = {
  id: number;
  review: string;
  prediction: string;
  probability: number;
  risk: string;
  date: string;
};

type Page = "analyzer" | "dashboard" | "history" | "insights";

function App() {
  const [activePage, setActivePage] = useState<Page>("analyzer");

  const [review, setReview] = useState("");
  const [verifiedPurchase, setVerifiedPurchase] = useState("Unknown");

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [showCalculation, setShowCalculation] = useState(false);

  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    const savedHistory = localStorage.getItem("truthlens_history");

    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);

        if (Array.isArray(parsed)) {
          setHistory(parsed);
        } else {
          setHistory([]);
        }
      } catch {
        setHistory([]);
      }
    }
  }, []);

  const saveHistory = (newItem: HistoryItem) => {
    setHistory((currentHistory) => {
      const updatedHistory = [newItem, ...currentHistory].slice(0, 50);

      localStorage.setItem(
        "truthlens_history",
        JSON.stringify(updatedHistory)
      );

      return updatedHistory;
    });
  };

  const getRiskFromProbability = (probability: number) => {
    if (probability >= 70) {
      return "HIGH";
    }

    if (probability >= 30) {
      return "MEDIUM";
    }

    return "LOW";
  };

  const normalizeAnalysisResponse = (data: any): AnalysisResult => {
    /*
      Your backend currently returns:

      {
        prediction: "Deceptive" / "Truthful",
        calculations: {...},
        linguistic_features: {...},
        signals: [...],
        term_contributions: {...}
      }

      The frontend previously expected prediction to be an object.

      This function supports both formats so the UI does not crash.
    */

    const rawPrediction = data?.prediction;

    let label = "Unknown";
    let deceptionProbability = 0;
    let truthProbability = 0;
    let riskLevel = "LOW";

    if (typeof rawPrediction === "string") {
      label = rawPrediction;
    } else if (rawPrediction && typeof rawPrediction === "object") {
      label = rawPrediction.label ?? "Unknown";
      deceptionProbability = Number(
        rawPrediction.deception_probability ?? 0
      );
      truthProbability = Number(rawPrediction.truth_probability ?? 0);
      riskLevel = String(rawPrediction.risk_level ?? "LOW");
    }

    const calculations = data?.calculations ?? {};

    /*
      If the backend returns calibrated_probability as a fraction
      such as 0.7742, convert it to 77.42%.
    */
    if (deceptionProbability === 0) {
      const calibrated = Number(
        calculations.calibrated_probability ?? 0
      );

      if (calibrated <= 1) {
        deceptionProbability = calibrated * 100;
      } else {
        deceptionProbability = calibrated;
      }
    }

    if (truthProbability === 0) {
      truthProbability = Math.max(0, 100 - deceptionProbability);
    }

    if (
      !rawPrediction ||
      typeof rawPrediction === "string" ||
      !rawPrediction.risk_level
    ) {
      riskLevel = calculations.risk_level
        ? String(calculations.risk_level).toUpperCase()
        : getRiskFromProbability(deceptionProbability);
    }

    const signals = Array.isArray(data?.signals)
      ? data.signals
      : [];

    const termContributions = data?.term_contributions ?? {};

    return {
      prediction: {
        label,
        deception_probability: deceptionProbability,
        truth_probability: truthProbability,
        risk_level: riskLevel,
      },

      calculations: {
        base_probability:
          typeof calculations.base_probability === "number"
            ? calculations.base_probability
            : undefined,

        calibrated_probability:
          typeof calculations.calibrated_probability === "number"
            ? calculations.calibrated_probability
            : deceptionProbability / 100,

        classification_threshold:
          typeof calculations.classification_threshold === "number"
            ? calculations.classification_threshold
            : 0.35,

        risk_thresholds:
          calculations.risk_thresholds ?? {
            low: 0.3,
            high: 0.7,
          },

        formula:
          calculations.formula ??
          "P_calibrated = 1 / (1 + exp(-(A * P_base + B)))",

        risk_level: riskLevel,
      },

      linguistic_features:
        data?.linguistic_features &&
        typeof data.linguistic_features === "object"
          ? data.linguistic_features
          : {},

      signals,

      term_contributions: {
        deceptive: Array.isArray(termContributions.deceptive)
          ? termContributions.deceptive
          : [],

        truthful: Array.isArray(termContributions.truthful)
          ? termContributions.truthful
          : [],
      },
    };
  };

  const analyzeReview = async () => {
    if (!review.trim()) {
      setError("Please enter a review before analyzing.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/analyze-review",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            review_text: review.trim(),
            verified_purchase:
              verifiedPurchase === "Unknown"
                ? null
                : verifiedPurchase,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);

        throw new Error(
          errorData?.detail ||
            "Unable to analyze the review."
        );
      }

      const rawData = await response.json();

      console.log("TruthLens API response:", rawData);

      const normalizedResult =
        normalizeAnalysisResponse(rawData);

      setResult(normalizedResult);

      const historyItem: HistoryItem = {
        id: Date.now(),
        review: review.trim(),
        prediction: normalizedResult.prediction.label,
        probability:
          normalizedResult.prediction.deception_probability,
        risk: normalizedResult.prediction.risk_level,
        date: new Date().toLocaleString(),
      };

      saveHistory(historyItem);
    } catch (err) {
      console.error("TruthLens analysis error:", err);

      setError(
        err instanceof Error
          ? err.message
          : "Could not analyze the review."
      );
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("truthlens_history");
  };

  const getRiskClass = (risk: string) => {
    const normalized = String(risk).toLowerCase();

    if (normalized === "medium") {
      return "medium";
    }

    if (normalized === "high") {
      return "high";
    }

    return "low";
  };

  const totalAnalyzed = history.length;

  const deceptiveCount = history.filter(
    (item) => item.prediction === "Deceptive"
  ).length;

  const truthfulCount = history.filter(
    (item) => item.prediction === "Truthful"
  ).length;

  const averageProbability =
    history.length > 0
      ? history.reduce(
          (sum, item) => sum + item.probability,
          0
        ) / history.length
      : 0;

  const lowCount = history.filter(
    (item) => item.risk === "LOW"
  ).length;

  const mediumCount = history.filter(
    (item) => item.risk === "MEDIUM"
  ).length;

  const highCount = history.filter(
    (item) => item.risk === "HIGH"
  ).length;

  const pageTitles: Record<Page, string> = {
    analyzer: "Review Analyzer",
    dashboard: "Dashboard",
    history: "Analysis History",
    insights: "Model Insights",
  };

  const pageDescriptions: Record<Page, string> = {
    analyzer:
      "Evaluate a review using NLP, linguistic analysis and the TruthLens machine-learning model.",

    dashboard:
      "Monitor your review-analysis activity and detection results.",

    history:
      "Review previously analyzed reviews and their model predictions.",

    insights:
      "Understand the model, features, calibration and detection methodology.",
  };

  const navigate = (page: Page) => {
    setActivePage(page);
  };

  return (
    <div className="app">

      {/* SIDEBAR */}

      <aside className="sidebar">

        <div className="brand">
          <div className="brand-icon">
            <ShieldCheck
              size={23}
              strokeWidth={2.2}
            />
          </div>

          <div className="brand-text">
            <h1>TruthLens</h1>
            <span>Review Intelligence</span>
          </div>
        </div>

        <div className="nav-section">
          <p className="nav-heading">
            WORKSPACE
          </p>

          <button
            className={`nav-item ${
              activePage === "analyzer"
                ? "active"
                : ""
            }`}
            onClick={() =>
              navigate("analyzer")
            }
          >
            <FileSearch size={18} />
            <span>Review Analyzer</span>
          </button>

          <button
            className={`nav-item ${
              activePage === "dashboard"
                ? "active"
                : ""
            }`}
            onClick={() =>
              navigate("dashboard")
            }
          >
            <BarChart3 size={18} />
            <span>Dashboard</span>
          </button>

          <button
            className={`nav-item ${
              activePage === "history"
                ? "active"
                : ""
            }`}
            onClick={() =>
              navigate("history")
            }
          >
            <History size={18} />
            <span>Analysis History</span>
          </button>
        </div>

        <div className="nav-section">
          <p className="nav-heading">
            SYSTEM
          </p>

          <button
            className={`nav-item ${
              activePage === "insights"
                ? "active"
                : ""
            }`}
            onClick={() =>
              navigate("insights")
            }
          >
            <Activity size={18} />
            <span>Model Insights</span>
          </button>
        </div>

        <div className="sidebar-bottom">

          <div className="system-status">
            <span className="status-dot" />

            <div>
              <strong>Model online</strong>
              <span>
                TruthLens detection engine
              </span>
            </div>
          </div>

          <div className="version-text">
            TruthLens v1.0
          </div>

        </div>

      </aside>

      {/* MAIN CONTENT */}

      <main className="main-content">

        <header className="topbar">

          <div>
            <p className="eyebrow">
              AI-POWERED REVIEW INTELLIGENCE
            </p>

            <h2>
              {pageTitles[activePage]}
            </h2>

            <p className="subtitle">
              {pageDescriptions[activePage]}
            </p>
          </div>

          <div className="topbar-status">
            <span className="status-dot" />
            API Connected
          </div>

        </header>

        {/* ================= REVIEW ANALYZER ================= */}

        {activePage === "analyzer" && (

          <section className="page-content analyzer-page">

            <div className="analyzer-grid">

              {/* INPUT */}

              <div className="panel input-panel">

                <div className="panel-header">

                  <div>
                    <h3>
                      Analyze a review
                    </h3>

                    <p>
                      Enter review content to
                      evaluate potential
                      deceptive patterns.
                    </p>
                  </div>

                  <div className="header-icon">
                    <FileSearch size={19} />
                  </div>

                </div>

                <label className="field-label">
                  Review text
                </label>

                <textarea
                  className="review-input"
                  value={review}
                  onChange={(e) =>
                    setReview(e.target.value)
                  }
                  placeholder="Paste or type a customer review here..."
                />

                <div className="input-meta">

                  <span>
                    {review.length} characters
                  </span>

                  <div className="verified-field">

                    <label>
                      Verified purchase
                    </label>

                    <select
                      value={verifiedPurchase}
                      onChange={(e) =>
                        setVerifiedPurchase(
                          e.target.value
                        )
                      }
                    >
                      <option value="Unknown">
                        Unknown
                      </option>

                      <option value="Y">
                        Yes
                      </option>

                      <option value="N">
                        No
                      </option>
                    </select>

                  </div>

                </div>

                {error && (
                  <div className="error-message">

                    <AlertTriangle size={15} />

                    {error}

                  </div>
                )}

                <button
                  className="analyze-button"
                  onClick={analyzeReview}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="loading-spinner" />
                      Analyzing review...
                    </>
                  ) : (
                    <>
                      Analyze Review
                      <Sparkles size={17} />
                    </>
                  )}
                </button>

              </div>

              {/* RESULTS */}

              <div className="results-column">

                {!result ? (

                  <div className="panel empty-state">

                    <div className="empty-icon">
                      <Target size={27} />
                    </div>

                    <h3>
                      Ready to analyze
                    </h3>

                    <p>
                      Submit a review to see
                      its deception probability,
                      classification, risk level
                      and model-indicative
                      signals.
                    </p>

                    <div className="empty-features">

                      <span>
                        <CheckCircle2 size={14} />
                        NLP analysis
                      </span>

                      <span>
                        <CheckCircle2 size={14} />
                        Linguistic signals
                      </span>

                      <span>
                        <CheckCircle2 size={14} />
                        Probability calibration
                      </span>

                    </div>

                  </div>

                ) : (

                  <>

                    {/* RESULT CARD */}

                    <div className="panel result-card">

                      <div className="result-header">

                        <div>

                          <p className="section-label">
                            DETECTION RESULT
                          </p>

                          <div className="prediction-row">

                            {result.prediction.label ===
                            "Deceptive" ? (

                              <AlertTriangle
                                size={24}
                                className="prediction-icon deceptive"
                              />

                            ) : (

                              <CheckCircle2
                                size={24}
                                className="prediction-icon truthful"
                              />

                            )}

                            <h3>
                              {result.prediction.label}
                            </h3>

                          </div>

                        </div>

                        <span
                          className={`risk-badge ${getRiskClass(
                            result.prediction
                              .risk_level
                          )}`}
                        >
                          {result.prediction.risk_level}{" "}
                          RISK
                        </span>

                      </div>

                      <div className="probability-section">

                        <div className="probability-heading">

                          <div>

                            <span className="probability-value">

                              {result.prediction.deception_probability.toFixed(
                                1
                              )}
                              %

                            </span>

                            <span className="probability-label">
                              deception probability
                            </span>

                          </div>

                          <div className="truth-value">

                            {result.prediction.truth_probability.toFixed(
                              1
                            )}
                            % truthful

                          </div>

                        </div>

                        <div className="probability-track">

                          <div
                            className={`probability-fill ${getRiskClass(
                              result.prediction
                                .risk_level
                            )}`}
                            style={{
                              width: `${Math.min(
                                100,
                                Math.max(
                                  0,
                                  result.prediction
                                    .deception_probability
                                )
                              )}%`,
                            }}
                          />

                        </div>

                        <div className="probability-scale">
                          <span>0%</span>
                          <span>50%</span>
                          <span>100%</span>
                        </div>

                      </div>

                      <div className="result-stats">

                        <div className="stat-box">

                          <span>
                            Truth probability
                          </span>

                          <strong>
                            {result.prediction.truth_probability.toFixed(
                              1
                            )}
                            %
                          </strong>

                        </div>

                        <div className="stat-box">

                          <span>
                            Detection threshold
                          </span>

                          <strong>
                            {(
                              (result.calculations
                                .classification_threshold ??
                                0.35) *
                              100
                            ).toFixed(0)}
                            %
                          </strong>

                        </div>

                        <div className="stat-box">

                          <span>
                            Verified purchase
                          </span>

                          <strong>
                            {verifiedPurchase ===
                            "Unknown"
                              ? "Unknown"
                              : verifiedPurchase ===
                                "Y"
                              ? "Yes"
                              : "No"}
                          </strong>

                        </div>

                      </div>

                    </div>

                    {/* EXPLANATION */}

                    <div className="panel explanation-panel">

                      <div className="panel-header">

                        <div>

                          <h3>
                            Why was this review flagged?
                          </h3>

                          <p>
                            Model-indicative patterns
                            identified during analysis.
                          </p>

                        </div>

                        <Info
                          size={18}
                          className="panel-icon"
                        />

                      </div>

                      {(result.signals ?? [])
                        .length > 0 ? (

                        <div className="signals-list">

                          {(result.signals ?? []).map(
                            (signal, index) => (

                              <div
                                className="signal-item"
                                key={index}
                              >

                                <span className="signal-marker" />

                                {signal}

                              </div>

                            )
                          )}

                        </div>

                      ) : (

                        <div className="no-signals">
                          No strong linguistic
                          signals were detected.
                        </div>

                      )}

                      <div className="contributions">

                        <div>

                          <h4>
                            Deception-associated terms
                          </h4>

                          {(result.term_contributions
                            ?.deceptive ?? []).length ===
                          0 ? (

                            <div className="no-contributions">
                              No significant
                              deception-associated
                              terms were identified.
                            </div>

                          ) : (

                            <div className="term-list">

                              {(
                                result
                                  .term_contributions
                                  ?.deceptive ?? []
                              ).map(
                                (item, index) => (

                                  <div
                                    className="term-row"
                                    key={`${item.term}-${index}`}
                                  >

                                    <span>
                                      {item.term}
                                    </span>

                                    <strong>
                                      +
                                      {Number(
                                        item.contribution
                                      ).toFixed(4)}
                                    </strong>

                                  </div>

                                )
                              )}

                            </div>

                          )}

                        </div>

                        <div>

                          <h4>
                            Truth-associated terms
                          </h4>

                          {(result.term_contributions
                            ?.truthful ?? []).length ===
                          0 ? (

                            <div className="no-contributions">
                              No significant
                              truth-associated terms
                              were identified.
                            </div>

                          ) : (

                            <div className="term-list">

                              {(
                                result
                                  .term_contributions
                                  ?.truthful ?? []
                              ).map(
                                (item, index) => (

                                  <div
                                    className="term-row"
                                    key={`${item.term}-${index}`}
                                  >

                                    <span>
                                      {item.term}
                                    </span>

                                    <strong className="truth-contribution">
                                      {item.contribution >=
                                      0
                                        ? "+"
                                        : ""}
                                      {Number(
                                        item.contribution
                                      ).toFixed(4)}
                                    </strong>

                                  </div>

                                )
                              )}

                            </div>

                          )}

                        </div>

                      </div>

                      {/* CALCULATION */}

                      <button
                        className="calculation-toggle"
                        onClick={() =>
                          setShowCalculation(
                            !showCalculation
                          )
                        }
                      >

                        <span>
                          How was this score
                          calculated?
                        </span>

                        {showCalculation ? (
                          <ChevronUp size={17} />
                        ) : (
                          <ChevronDown size={17} />
                        )}

                      </button>

                      {showCalculation && (

                        <div className="calculation-details">

                          <div className="calculation-row">

                            <span>
                              Base probability
                            </span>

                            <strong>
                              {typeof result
                                .calculations
                                .base_probability ===
                              "number"
                                ? `${(
                                    result
                                      .calculations
                                      .base_probability *
                                    100
                                  ).toFixed(2)}%`
                                : "Not available"}
                            </strong>

                          </div>

                          <div className="calculation-row">

                            <span>
                              Calibrated probability
                            </span>

                            <strong>
                              {(
                                result
                                  .prediction
                                  .deception_probability
                              ).toFixed(2)}
                              %
                            </strong>

                          </div>

                          <div className="calculation-row">

                            <span>
                              Classification threshold
                            </span>

                            <strong>
                              {(
                                (result
                                  .calculations
                                  .classification_threshold ??
                                  0.35) *
                                100
                              ).toFixed(0)}
                              %
                            </strong>

                          </div>

                          <div className="calculation-row">

                            <span>
                              Risk band
                            </span>

                            <strong>
                              {result.prediction
                                .risk_level}
                            </strong>

                          </div>

                          {result.calculations
                            .formula && (

                            <div className="formula-box">

                              <span>
                                Calibration formula
                              </span>

                              <code>
                                {
                                  result
                                    .calculations
                                    .formula
                                }
                              </code>

                            </div>

                          )}

                        </div>

                      )}

                    </div>

                    <div className="disclaimer-box">

                      <Info size={16} />

                      <p>
                        Model-indicative signals
                        describe patterns associated
                        with the prediction. They are
                        not proof that a review is fake
                        or deceptive.
                      </p>

                    </div>

                  </>

                )}

              </div>

            </div>

          </section>

        )}

        {/* ================= DASHBOARD ================= */}

        {activePage === "dashboard" && (

          <section className="page-content">

            <div className="dashboard-grid">

              <div className="dashboard-stat">

                <span>
                  Total analyzed
                </span>

                <strong>
                  {totalAnalyzed}
                </strong>

                <small>
                  Reviews analyzed in this browser
                </small>

              </div>

              <div className="dashboard-stat">

                <span>
                  Deceptive detected
                </span>

                <strong>
                  {deceptiveCount}
                </strong>

                <small>
                  Classified above the detection
                  threshold
                </small>

              </div>

              <div className="dashboard-stat">

                <span>
                  Truthful detected
                </span>

                <strong>
                  {truthfulCount}
                </strong>

                <small>
                  Classified below the detection
                  threshold
                </small>

              </div>

              <div className="dashboard-stat">

                <span>
                  Average deception probability
                </span>

                <strong>
                  {averageProbability.toFixed(1)}%
                </strong>

                <small>
                  Across analyzed reviews
                </small>

              </div>

            </div>

            <div className="dashboard-columns">

              <div className="panel dashboard-panel">

                <div className="panel-header">

                  <div>
                    <h3>
                      Risk distribution
                    </h3>

                    <p>
                      Distribution of analyzed reviews
                      by model risk band.
                    </p>
                  </div>

                  <BarChart3 size={20} />

                </div>

                <div className="risk-distribution">

                  <div className="risk-distribution-row">

                    <span>
                      <i className="risk-dot low" />
                      Low
                    </span>

                    <strong>
                      {lowCount}
                    </strong>

                  </div>

                  <div className="risk-distribution-row">

                    <span>
                      <i className="risk-dot medium" />
                      Medium
                    </span>

                    <strong>
                      {mediumCount}
                    </strong>

                  </div>

                  <div className="risk-distribution-row">

                    <span>
                      <i className="risk-dot high" />
                      High
                    </span>

                    <strong>
                      {highCount}
                    </strong>

                  </div>

                </div>

              </div>

              <div className="panel dashboard-panel">

                <div className="panel-header">

                  <div>
                    <h3>
                      Model status
                    </h3>

                    <p>
                      Current TruthLens inference
                      service.
                    </p>
                  </div>

                  <ShieldCheck size={20} />

                </div>

                <div className="model-status-card">

                  <div className="model-status-icon">
                    <CheckCircle2 size={22} />
                  </div>

                  <div>
                    <strong>
                      Detection engine online
                    </strong>

                    <span>
                      Calibrated NLP classification
                      pipeline available through
                      FastAPI.
                    </span>
                  </div>

                </div>

                <button
                  className="secondary-button"
                  onClick={() =>
                    navigate("analyzer")
                  }
                >
                  Analyze another review
                </button>

              </div>

            </div>

          </section>

        )}

        {/* ================= HISTORY ================= */}

        {activePage === "history" && (

          <section className="page-content">

            <div className="panel history-panel">

              <div className="panel-header">

                <div>

                  <h3>
                    Analysis history
                  </h3>

                  <p>
                    Previous analyses stored locally
                    in this browser.
                  </p>

                </div>

                {history.length > 0 && (

                  <button
                    className="clear-button"
                    onClick={clearHistory}
                  >
                    Clear history
                  </button>

                )}

              </div>

              {history.length === 0 ? (

                <div className="history-empty">

                  <History size={30} />

                  <h3>
                    No analyses yet
                  </h3>

                  <p>
                    Reviews you analyze will appear
                    here.
                  </p>

                  <button
                    className="secondary-button"
                    onClick={() =>
                      navigate("analyzer")
                    }
                  >
                    Analyze a review
                  </button>

                </div>

              ) : (

                <div className="history-list">

                  {history.map((item) => (

                    <div
                      className="history-item"
                      key={item.id}
                    >

                      <div className="history-main">

                        <div className="history-item-top">

                          <span
                            className={`history-prediction ${getRiskClass(
                              item.risk
                            )}`}
                          >
                            {item.prediction}
                          </span>

                          <span
                            className={`risk-badge small ${getRiskClass(
                              item.risk
                            )}`}
                          >
                            {item.risk}
                          </span>

                        </div>

                        <p>
                          {item.review}
                        </p>

                        <small>
                          {item.date}
                        </small>

                      </div>

                      <div className="history-probability">

                        <strong>
                          {item.probability.toFixed(
                            1
                          )}
                          %
                        </strong>

                        <span>
                          deception probability
                        </span>

                      </div>

                    </div>

                  ))}

                </div>

              )}

            </div>

          </section>

        )}

        {/* ================= MODEL INSIGHTS ================= */}

        {activePage === "insights" && (

          <section className="page-content">

            <div className="insights-grid">

              <div className="panel insight-panel">

                <div className="insight-icon">
                  <Target size={21} />
                </div>

                <h3>
                  Detection approach
                </h3>

                <p>
                  TruthLens combines word-level
                  and character-level TF-IDF
                  representations with linguistic
                  features and verified-purchase
                  information.
                </p>

              </div>

              <div className="panel insight-panel">

                <div className="insight-icon">
                  <Activity size={21} />
                </div>

                <h3>
                  Machine-learning model
                </h3>

                <p>
                  The production pipeline uses a
                  calibrated Logistic Regression
                  classifier to estimate the
                  probability that a review is
                  deceptive.
                </p>

              </div>

              <div className="panel insight-panel">

                <div className="insight-icon">
                  <BarChart3 size={21} />
                </div>

                <h3>
                  Model performance
                </h3>

                <div className="metric-list">

                  <div>
                    <span>
                      Accuracy
                    </span>
                    <strong>
                      82.43%
                    </strong>
                  </div>

                  <div>
                    <span>
                      Precision
                    </span>
                    <strong>
                      83.95%
                    </strong>
                  </div>

                  <div>
                    <span>
                      Recall
                    </span>
                    <strong>
                      80.19%
                    </strong>
                  </div>

                  <div>
                    <span>
                      F1 Score
                    </span>
                    <strong>
                      82.03%
                    </strong>
                  </div>

                  <div>
                    <span>
                      ROC-AUC
                    </span>
                    <strong>
                      88.09%
                    </strong>
                  </div>

                </div>

              </div>

              <div className="panel insight-panel">

                <div className="insight-icon">
                  <ShieldCheck size={21} />
                </div>

                <h3>
                  Probability calibration
                </h3>

                <p>
                  The model probability is calibrated
                  before being displayed to the user.
                  The current classification threshold
                  is 35%.
                </p>

                <div className="risk-band-list">

                  <div>
                    <span className="risk-dot low" />
                    <strong>
                      Low
                    </strong>
                    <span>
                      &lt; 30%
                    </span>
                  </div>

                  <div>
                    <span className="risk-dot medium" />
                    <strong>
                      Medium
                    </strong>
                    <span>
                      30% – &lt; 70%
                    </span>
                  </div>

                  <div>
                    <span className="risk-dot high" />
                    <strong>
                      High
                    </strong>
                    <span>
                      ≥ 70%
                    </span>
                  </div>

                </div>

              </div>

            </div>

            <div className="panel methodology-panel">

              <div className="panel-header">

                <div>
                  <h3>
                    How to interpret TruthLens
                  </h3>

                  <p>
                    Important limitations of
                    automated review detection.
                  </p>
                </div>

                <Info size={19} />

              </div>

              <div className="methodology-content">

                <div>
                  <strong>
                    Deception probability
                  </strong>

                  <p>
                    This represents the model's
                    estimated probability that the
                    review belongs to the deceptive
                    class based on patterns learned
                    from the training data.
                  </p>
                </div>

                <div>
                  <strong>
                    Risk level
                  </strong>

                  <p>
                    Risk bands are UI categories based
                    on the deception probability. They
                    should not be interpreted as proof
                    that a reviewer is fraudulent.
                  </p>
                </div>

                <div>
                  <strong>
                    Model signals
                  </strong>

                  <p>
                    Linguistic patterns and important
                    terms are model-indicative
                    features. They help explain the
                    prediction but do not establish
                    causation.
                  </p>
                </div>

              </div>

            </div>

          </section>

        )}

      </main>

    </div>
  );
}

export default App;
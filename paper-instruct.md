# Instruction for Coding Agent — Rewrite and Elevate the Research Paper

## Objective

You are a senior research-writing and quantitative-finance coding agent.

Your task is to completely rewrite, restructure, and elevate the research paper into a publication-quality empirical economics / computational finance paper suitable for:
- an undergraduate thesis,
- conference submission,
- or journal-style working paper.

The rewritten paper must NOT simply paraphrase existing content.  
It must:
- improve academic rigor,
- strengthen argument flow,
- tighten methodology explanation,
- improve literature synthesis,
- clarify empirical contribution,
- and follow proper economics research writing standards.

The paper should read like a serious empirical finance research paper written in clear academic English.

---

# Core Research Topic

## Working Title

Can Vietnamese Financial News Sentiment Improve Volatility Forecasting of the VN-Index Beyond a Pure GARCH Baseline? Evidence from a Two-Stage GARCH–LSTM Hybrid Model Using PhoBERT-Based Sentiment Signals

---

# Central Research Question

Does daily Vietnamese financial news sentiment, extracted using PhoBERT from CafeF articles, improve next-day realized volatility forecasts of the VN-Index beyond a pure GARCH(1,1) baseline when incorporated through a two-stage GARCH+LSTM hybrid framework?

---

# Secondary Research Questions

## 1. Sentiment Asymmetry

Does negative sentiment contribute more strongly to volatility forecasting than positive sentiment in the Vietnamese stock market?

Motivation:
- Vu et al. (2023) found that negative sentiment significantly affects variance while positive sentiment does not.

Key variables:
- negative_share
- positive_share
- sentiment polarity measures

---

## 2. Market Regime Dependence

Does the hybrid model outperform the baseline differently across:
- high-volatility regimes,
- crisis periods,
- and normal periods?

Potential regime periods:
- COVID-19 crash (2020)
- liquidity/manipulation shock (2022)

---

## 3. News Volume Effects

Does information intensity itself predict volatility?

Key variables:
- n_articles
- has_news
- sentiment_std

Motivation:
- Antweiler & Frank (2004)
- Bodilsen & Lunde (2025)

---

## 4. Nonlinear Sentiment–Variance Relationship

Is the relationship between sentiment and conditional variance nonlinear?

Motivation:
- Léber & Egyed (2025)

The paper should justify the LSTM architecture through:
- nonlinear Granger causality,
- residual nonlinear structure,
- or forecast improvement evidence.

---

# Positioning of the Paper

This paper sits at the intersection of:
- financial econometrics,
- NLP for finance,
- volatility forecasting,
- and emerging market behavioral finance.

The contribution is NOT merely applying machine learning.

The contribution is:
1. Building a Vietnamese financial-news sentiment pipeline using PhoBERT.
2. Combining econometric volatility modeling with deep learning residual correction.
3. Studying sentiment-driven volatility forecasting in an emerging retail-dominated market.
4. Evaluating whether sentiment adds incremental predictive value beyond GARCH.
5. Investigating asymmetry and nonlinear dynamics.

---

# Writing Standards (MANDATORY)

The rewritten paper MUST follow the principles of strong economics writing.

## Key Principles

### 1. Clarity Above All
- Prioritize clarity over sophistication.
- Avoid vague language.
- Avoid unnecessarily complex sentences.
- Avoid decorative writing.

### 2. Strong Logical Flow
Every section must answer:
- What is the problem?
- Why does it matter?
- What does the literature say?
- What gap remains?
- What does this paper do?
- Why is the methodology appropriate?
- What do the results imply?

### 3. Technical but Readable
The paper should sound:
- academic,
- empirical,
- rigorous,
- concise,
- and direct.

Avoid:
- marketing-style language,
- exaggerated claims,
- unsupported statements.

### 4. Active Voice Preferred
Use:
- “This paper estimates…”
instead of:
- “It is estimated…”

### 5. Remove Filler
Delete:
- redundant transitions,
- repeated explanations,
- generic AI-generated phrases,
- empty academic wording.

### 6. Maintain Consistent Tense
Generally:
- present tense for established literature,
- past tense for completed procedures,
- present tense for interpretation.

---

# Economics Research Writing Principles

The rewrite MUST follow these economics-writing principles:

## Research
Demonstrate:
- understanding of volatility literature,
- econometric forecasting,
- financial sentiment analysis,
- and Vietnamese market context.

## Analysis
Go beyond description:
- explain mechanisms,
- interpret findings economically,
- connect empirical evidence to theory.

## Organization
The paper must follow a coherent structure:
1. Introduction
2. Literature Review
3. Data
4. Methodology
5. Empirical Results
6. Robustness Checks
7. Discussion
8. Conclusion

## Clarity
Every paragraph should have:
- one central idea,
- a clear topic sentence,
- and logical progression.

---

# Required Paper Structure

# 1. Introduction

The introduction MUST:
- clearly state the research question,
- explain why volatility forecasting matters,
- explain why Vietnam is an interesting setting,
- explain why sentiment may matter,
- explain limitations of pure GARCH models,
- motivate hybrid econometric–deep learning methods,
- summarize the methodology,
- and state the contributions explicitly.

The introduction should answer:
- What is this paper about?
- Why is the question important?
- What gap exists?
- What does this paper contribute?
- What are the main findings?

The contribution paragraph should be extremely clear and explicit.

---

# 2. Literature Review

The literature review must be THEMATIC, not paper-by-paper summary dumping.

Organize the literature into sections such as:
1. Volatility modeling literature
2. Sentiment and financial markets
3. NLP and financial sentiment
4. Hybrid GARCH–deep learning models
5. Vietnamese market literature

Each subsection should:
- synthesize findings,
- compare methodologies,
- identify limitations,
- and motivate this paper.

The review must end with:
- a clear research gap,
- and explanation of how this paper fills it.

---

# 3. Data

Clearly describe:
- VN-Index data,
- CafeF scraping,
- article collection,
- preprocessing,
- sentiment labeling,
- PhoBERT inference,
- sentiment aggregation,
- realized volatility construction,
- sample period,
- train/validation/test split.

Define every major variable carefully.

Explain:
- why each variable matters,
- how it is constructed,
- and how it maps to theory.

---

# 4. Methodology

This section must be rigorous and mathematically structured.

Include:
- GARCH(1,1),
- realized volatility definition,
- residual construction,
- LSTM architecture,
- hybrid framework,
- evaluation metrics,
- forecasting protocol,
- benchmark comparisons.

The methodology section should:
- justify every modeling decision,
- explain why GARCH alone is insufficient,
- explain why nonlinear learning may help,
- explain why sentiment is theoretically informative.

Clearly distinguish:
- Stage 1 econometric modeling,
- Stage 2 residual learning.

---

# 5. Empirical Results

Results must NOT merely report numbers.

Interpret:
- economic meaning,
- forecasting implications,
- market behavior implications,
- asymmetry implications,
- nonlinear implications.

Include:
- benchmark comparisons,
- tables,
- rolling performance,
- regime analysis,
- error metrics,
- feature importance if available.

---

# 6. Robustness Checks

Potential robustness sections:
- alternative volatility definitions,
- excluding crisis periods,
- alternative sentiment aggregation,
- different LSTM windows,
- EGARCH/GJR-GARCH comparison,
- sentiment lag structures,
- news-volume-only models.

Robustness discussion should explain:
- whether results are stable,
- where they weaken,
- and why.

---

# 7. Discussion

This section should:
- connect findings to literature,
- explain implications for emerging markets,
- discuss retail-investor behavior,
- discuss limitations of sentiment inference,
- discuss data limitations,
- discuss model limitations,
- explain practical implications.

---

# 8. Conclusion

The conclusion must:
- directly answer the research question,
- summarize contributions,
- summarize major findings,
- acknowledge limitations,
- propose future research.

Avoid introducing new results.

---

# Style Constraints

## DO NOT
- use exaggerated claims,
- use “revolutionary”, “groundbreaking”, etc.
- overuse bullet points in the actual paper,
- write like a blog post,
- repeat identical ideas,
- generate shallow summaries.

## DO
- write in dense but readable academic prose,
- maintain coherence,
- provide transitions between ideas,
- explain intuition behind methods,
- interpret results economically.

---

# Methodological Expectations

The paper should clearly explain:

## Realized Volatility
Define mathematically.

## GARCH(1,1)
Explain:
- conditional variance,
- volatility clustering,
- persistence.

## Residual Learning
Explain:
- why GARCH residuals may contain nonlinear information.

## LSTM
Explain:
- sequence learning,
- nonlinear temporal dynamics,
- sentiment interaction effects.

## Hybrid Architecture
Clarify:
- GARCH captures linear conditional heteroskedasticity,
- LSTM captures remaining nonlinear structure.

---

# Important Writing Instructions

## The paper should feel:
- cohesive,
- deliberate,
- technically grounded,
- and human-written.

## Every section must:
- explicitly connect back to the research question.

## Every major modeling choice must:
- have theoretical or empirical justification.

## The literature review must:
- synthesize,
- compare,
- critique,
- and position the paper.

## The discussion section must:
- interpret results economically,
- not just statistically.

---

# Output Expectations

When rewriting:
- produce publication-quality academic prose,
- substantially improve organization,
- remove redundancy,
- tighten arguments,
- strengthen transitions,
- and improve methodological precision.

Do not merely lightly edit the original draft.

The final output should resemble:
- a serious empirical finance thesis,
- or an economics/financial econometrics conference paper.

---

# Additional Guidance from Economics Writing Principles

The rewrite should follow these principles derived from economics research-writing best practices:

1. A good paper asks a clear and manageable research question.
2. Writing should communicate reasoning, not merely describe procedures.
3. Organization matters as much as technical correctness.
4. Every section should serve the central argument.
5. Literature review should identify gaps, not summarize endlessly.
6. Methodology should justify choices, not merely list models.
7. Results should explain meaning, not only statistical significance.
8. Discussion should connect evidence to economic interpretation.
9. Revision should aggressively remove unnecessary wording.
10. Clarity and precision are more important than stylistic sophistication.

---

# Final Instruction

Rewrite the paper as a coherent, rigorous, publication-style empirical economics and computational finance paper that convincingly answers whether Vietnamese financial news sentiment improves VN-Index volatility forecasting beyond traditional econometric baselines.

Final Instruction

Rewrite the paper as a coherent, rigorous, publication-style empirical economics and computational finance paper that convincingly answers whether Vietnamese financial news sentiment improves VN-Index volatility forecasting beyond traditional econometric baselines.

To help you with the writing, please read the Base Papers' Summary to understand the motivation behind this paper.

# Base Papers' Summary

Antweiler, W., & Frank, M. Z. (2004). Is all that talk just noise? The information content of internet stock message boards. *Journal of Finance, 59*(3), 1259–1294.

Antweiler and Frank study the effect of more than 1.5 million messages posted on Yahoo! Finance and Raging Bull about 45 companies in the Dow Jones Industrial Average and the Dow Jones Internet Index, measuring bullishness using computational linguistics methods and using Wall Street Journal news stories as controls. They find that stock messages help predict market volatility, while their effect on stock returns is statistically significant but economically small. Consistent with Harris and Raviv (1993), disagreement among posted messages is associated with increased trading volume.

**Relevance:** Establishes the foundational empirical link between news/message volume and market volatility, directly motivating your use of `n_articles` and `sentiment_std` (disagreement proxy) as features in the GARCH+LSTM model. The finding that information volume predicts volatility independently of sentiment direction supports including `n_articles` as a standalone predictor alongside `mean_sentiment`.

---

Bodilsen, S. T., & Lunde, A. (2025). Exploiting news analytics for volatility forecasting. *Journal of Applied Econometrics, 40*(1), 18–36.

Bodilsen and Lunde investigate the potential of news sentiment in predicting stock market volatility by augmenting traditional time series models of realized volatility with the sentiment of macroeconomic and firm-specific news, relying on RavenPack News Analytics data for the US market. Their results demonstrate that incorporating domestic macroeconomic news sentiment significantly improves volatility predictions for individual stocks and the S&P 500 Index, with substantial enhancements particularly in long-horizon volatility predictions. In contrast, firm-specific news sentiment shows only modest predictive power in the general framework, though overnight news count for firm-specific news significantly improves one-period-ahead forecasts.

**Relevance:** The most methodologically proximate econometrics paper to your work. Three findings directly inform your design: macroeconomic sentiment matters more than firm-level sentiment (motivating your `macro_sentiment` feature from CafeF's Vĩ mô and Kinh tế categories), overnight news count has independent predictive power (motivating the `has_news` indicator and `n_articles` on post-close article days), and sentiment improves HAR-family baseline forecasts (analogous to your GARCH baseline improvement hypothesis). The HAR-RPNA framework is the closest Western-market precedent for your GARCH+LSTM setup.

---

Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics, 31*(3), 307–327.

Bollerslev generalizes Engle's (1982) ARCH model by allowing the conditional variance equation to depend on both past squared residuals and past conditional variances, producing the GARCH(p,q) specification. The simple GARCH(1,1) is shown to be sufficient for most empirical financial time series, capturing volatility clustering parsimoniously while remaining tractable for maximum likelihood estimation. Stationarity conditions, the autocorrelation structure of squared returns, and likelihood-based testing procedures are derived.

**Relevance:** The theoretical foundation for your Stage 1 baseline. The GARCH(1,1) conditional variance estimates and standardized residuals produced by this model are the direct inputs to your LSTM stage. Cite this when introducing the baseline specification and when defining `garch_conditional_vol`, `garch_forecast_vol`, and `garch_std_resid` in the methodology.

---

Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica, 50*(4), 987–1008.

Engle introduces ARCH processes as a new class of stochastic model with mean-zero, serially uncorrelated properties and nonconstant variances conditional on the past but constant unconditional variances, so that the recent past gives information about the one-period forecast variance. A regression model with ARCH disturbances is introduced, maximum likelihood estimators are derived, and a Lagrange multiplier test for ARCH effects is formulated.

**Relevance:** The originating paper for conditional heteroskedasticity modeling and the direct predecessor to Bollerslev (1986). Cite this alongside Bollerslev when introducing the GARCH framework. Also relevant when reporting your ARCH-LM test on GARCH residuals during baseline validation in Phase 5 of the workflow.

---

Léber, M., & Egyed, B. (2025). The sentiment augmented GARCH-LSTM hybrid model for value-at-risk forecasting. *Computational Economics*.

Léber and Egyed present a sentiment-augmented GARCH-LSTM hybrid model for forecasting volatility and Value-at-Risk of individual S&P 500 stocks using Bloomberg-sourced Twitter and news sentiment indices from 2019 to 2024, emphasizing the nonlinear relationship between variance and sentiment indices. The sentiment-augmented GARCH-LSTM model demonstrates superior performance compared to the traditional GARCH model for a considerable subset of equities, achieving the most accurate conditional volatility forecast and producing VaR estimations with the smallest expected shortfall while satisfying all statistical backtests. Nonlinear Granger causality between sentiment and conditional variance is shown to be more prevalent than linear causality, justifying the LSTM component over a simpler GARCH-X extension.

**Relevance:** The closest architectural precedent to your model. Key differentiators to state explicitly in your thesis: you use PhoBERT on scraped Vietnamese text rather than Bloomberg's proprietary sentiment index; you target next-day index-level realized volatility rather than VaR on individual S&P 500 stocks; and your LSTM learns the GARCH residual specifically rather than the full variance prediction. The nonlinear Granger causality result is a diagnostic test you should replicate on your own VN-Index data before committing to the LSTM architecture over a simpler GARCH-X specification.

---

Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. In *Findings of the Association for Computational Linguistics: EMNLP 2020* (pp. 1037–1042).

Nguyen and Nguyen present PhoBERT-base and PhoBERT-large as the first public large-scale monolingual language models pre-trained specifically for Vietnamese, built on the RoBERTa architecture and trained on a 20GB corpus of Vietnamese Wikipedia and news texts using masked language modeling. PhoBERT consistently outperforms the best pre-trained multilingual model XLM-R and improves state-of-the-art results on multiple Vietnamese NLP tasks including POS tagging, dependency parsing, named-entity recognition, and natural language inference. The maximum sequence length is 256 tokens, and input texts must be word-segmented prior to tokenization.

**Relevance:** The foundational model paper for your sentiment inference pipeline. Cite this when introducing PhoBERT as your chosen encoder and when justifying the choice of a Vietnamese-specific monolingual model over multilingual alternatives. The 256-token sequence length constraint directly determines your title + lead paragraph input construction strategy, and the news-text pretraining corpus composition informs why full-body input is inappropriate without controlled truncation.

---

Sontung. (2021). Stock article title sentiment-based classification using PhoBERT. *Proceedings of the 2nd International Conference on Human-centered Artificial Intelligence (Computing4Human 2021)*, Da Nang, Vietnam.

Sontung and colleagues fine-tune PhoBERT on a dataset of over 1,000 financial article titles from CafeF.vn, labeling them into three classes — negative, neutral, and positive — with expert annotation, resulting in 187 negative, 248 neutral, and 565 positive samples. The model achieves up to 93% accuracy on the classification task using title-only input after preprocessing with VnCoreNLP named entity recognition and RDRSegmenter word segmentation, with an 80/10/10 train/validation/test split.

**Relevance:** The most directly domain-matched precedent for your sentiment model — same source (CafeF), same language (Vietnamese), same model family (PhoBERT), same three-class schema. The 93% accuracy benchmark on title-only input is a useful comparison point for your own evaluation results. Their class distribution (skewed toward positive) is also informative for anticipating class imbalance in your ViFiC annotation sample and for setting weighted loss in fine-tuning. The labeled dataset availability at their GitHub repository should be checked as a potential supplementary training resource.

---

Vu, L. T., Pham, D. N., Kieu, H. T., & Le, T. T. (2023). Sentiments extracted from news and stock market reactions in Vietnam. *International Journal of Financial Studies, 11*(3), 101.

Vu and colleagues fine-tune PhoBERT with CNN layers on nearly 40,000 Vietnamese financial articles from CafeF, VnEconomy, and Stockbiz, using 2,738 manually labeled articles in a binary positive/negative schema, achieving over 81% accuracy. The study then investigates market reactions using variance ratio tests and two-sample t-tests over ±5 and ±30 day event windows around 50 positive and 50 negative news dates. Results show no statistically significant difference in stock return means before and after news releases, but negative news sentiment significantly alters the variance of market returns over the 30-day window, while positive news produces no significant variance change.

**Relevance:** The primary Vietnamese-market precedent for your entire study. Three specific contributions inform your work directly: it validates PhoBERT as the appropriate model for Vietnamese financial news sentiment; it establishes the negative-only variance effect in the Vietnamese market, which motivates your asymmetry secondary research question and the `negative_share` feature; and it uses an event-study design that your forecasting framework explicitly extends — where Vu et al. ask whether sentiment changes variance, you ask whether sentiment predicts the magnitude of next-day volatility. Cite this as the paper your study most directly builds upon.
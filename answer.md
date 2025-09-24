# Short Answers

**1) If you only had 200 labeled replies, how would you improve the model without collecting thousands more?**  
Use data augmentation (paraphrases/back-translation or LLM paraphrasing), weak supervision (labeling functions), and active learning to prioritize labeling high-value examples. Also prefer simple strong baselines (TF-IDF + logistic) and metric learning / few-shot techniques rather than immediately deploying a large transformer.

**2) How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?**  
Audit and clean training data for harmful or skewed samples, run fairness and subgroup performance checks, and add input sanitization and safety filters. Route low-confidence predictions to human review and log decisions for retraining.

**3) Prompt design strategies to generate personalized cold openers with an LLM?**  
Give concise structured context (recipient role, company, a recent specific fact), include explicit stylistic constraints (tone, length), and use few-shot examples plus a "do not" list to avoid generic phrasing. Post-filter with heuristics (detect buzzwords / vagueness) and prefer multiple candidate outputs for human selection.

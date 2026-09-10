# Research Overview

* **Problem:** Multi-objective scheduling requires policies to adapt to changing operational preferences, but repeatedly retraining RL policies is computationally expensive.
* **Central Theme / Study:** How can reinforcement-learning schedulers efficiently adapt to competing operational preferences under heterogeneous scheduling conditions?

---

### Research Questions

* **RQ1:** At what adaptation budget — measured in PPO updates / environment interaction steps — does continued optimisation justify its additional cost over model merging? Adaptation methods compared: model merging (zero-budget), fine-tuning from the merged checkpoint, and training from scratch.
* **RQ2:** To what extent does the response to model composition vary across scheduling instances, and how robust is this variation to specialist-training randomness? 
* **RQ3:** Under what forms of scheduling heterogeneity, if any, does lightweight instance-aware model composition provide meaningful gains over static merging without requiring continued policy optimisation? 

---

### Hypotheses

**Overall Hypothesis:**  
Efficient multi-objective adaptation in RL scheduling depends not only on the desired operational preference, but also on the available adaptation budget and the characteristics of the scheduling instance.

* **H1 — Performance–Efficiency Hypothesis:**  
  Model merging will provide the strongest benefit under low adaptation budgets because it requires little or no additional policy optimisation, while continued optimisation is expected to become more competitive as greater training budget is available. The scarce resource in this setting is **environment interaction**, not memory: the policy has ~29k parameters, so budget is measured in PPO updates / environment steps rather than in trainable parameters, and parameter-efficient adaptation methods (LoRA/QLoRA) are not applicable at this scale.

* **H2 — Instance Heterogeneity Hypothesis:**  
  Scheduling instances will exhibit systematic differences in their response to model composition, but the magnitude of this response will also be influenced by specialist-training randomness. 

* **H3 — Lightweight Adaptation Hypothesis:**  
  The benefit of lightweight instance-aware model composition will depend on the structure of instance heterogeneity: limited gains are expected when additional objectives are independently generated, while structured objective relationships may create greater exploitable headroom than static merging. 

---

### Thesis (Provisional)

> Multi-objective RL scheduling should not be treated as a simple choice between static model merging and repeated retraining. The appropriate adaptation strategy depends on the available optimisation budget, the robustness of the merging process to specialist-training variability, and the structure of scheduling heterogeneity. This study investigates when lightweight instance-aware adaptation provides meaningful value between static merging and continued optimisation. 
